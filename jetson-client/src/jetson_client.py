"""
Main Jetson AGX Orin Client Application.
Handles device registration, heartbeat, OTA updates, and headless formula execution.
"""
import os
import sys
import time
import logging
import signal
import threading
import json
import uuid
import socket
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Local imports
from system_metrics import JetsonMetrics
from api_client import BackendAPIClient
from model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/jetson-client.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class JetsonClient:
    """
    Main Jetson Client application.
    Runs as a daemon on the Jetson device.
    """

    def __init__(self, config_path: str = "/etc/jetson-client/config.json"):
        """
        Initialize Jetson Client.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        self.heartbeat_thread = None

        # Get or generate device ID
        self.device_id = self._get_device_id()

        # Initialize components
        self.metrics_collector = JetsonMetrics()
        self.api_client = BackendAPIClient(
            base_url=self.config['backend_url'],
            device_id=self.device_id,
            api_key=self.config.get('api_key')
        )
        self.model_manager = ModelManager(
            models_dir=self.config.get('models_dir', '/opt/ml-platform/models')
        )

        # State
        self.registered = False
        self.last_heartbeat = None
        self.heartbeat_interval = self.config.get('heartbeat_interval', 30)

        logger.info(f"Jetson Client initialized. Device ID: {self.device_id}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'backend_url': os.getenv('BACKEND_URL', 'http://localhost:8000'),
            'api_key': os.getenv('API_KEY'),
            'heartbeat_interval': int(os.getenv('HEARTBEAT_INTERVAL', '30')),
            'device_name': os.getenv('DEVICE_NAME', socket.gethostname()),
            'device_type': 'jetson_orin',
            'location': os.getenv('DEVICE_LOCATION', 'Unknown'),
            'models_dir': '/opt/ml-platform/models',
            'auto_update': True,
            'log_level': 'INFO'
        }

    def _get_device_id(self) -> str:
        """
        Get or generate unique device ID.
        Uses MAC address or generates UUID.
        """
        device_id_file = Path('/etc/jetson-client/device_id')

        # Try to load existing ID
        if device_id_file.exists():
            try:
                with open(device_id_file, 'r') as f:
                    device_id = f.read().strip()
                    if device_id:
                        return device_id
            except Exception as e:
                logger.warning(f"Error reading device ID file: {e}")

        # Generate new ID from MAC address
        try:
            # Get primary network interface MAC
            mac = uuid.getnode()
            device_id = f"jetson-{mac:012x}"
        except Exception:
            # Fallback to random UUID
            device_id = f"jetson-{uuid.uuid4().hex[:12]}"

        # Save for future use
        try:
            device_id_file.parent.mkdir(parents=True, exist_ok=True)
            with open(device_id_file, 'w') as f:
                f.write(device_id)
            logger.info(f"Device ID saved: {device_id}")
        except Exception as e:
            logger.warning(f"Could not save device ID: {e}")

        return device_id

    def _get_ip_address(self) -> Optional[str]:
        """Get device IP address."""
        try:
            # Connect to external host to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return None

    def _get_mac_address(self) -> Optional[str]:
        """Get device MAC address."""
        try:
            mac = uuid.getnode()
            mac_str = ':'.join(['{:02x}'.format((mac >> elements) & 0xff)
                                for elements in range(0, 48, 8)][::-1])
            return mac_str
        except Exception:
            return None

    def register(self) -> bool:
        """Register device with backend."""
        try:
            logger.info("Registering device with backend...")

            hw_info = self.metrics_collector.get_hardware_info()

            response = self.api_client.register_device(
                device_name=self.config.get('device_name', socket.gethostname()),
                device_type=self.config.get('device_type', 'jetson_orin'),
                hardware_info=hw_info,
                firmware_version=hw_info.get('l4t_version'),
                os_version=hw_info.get('os', 'Linux'),
                mac_address=self._get_mac_address(),
                location=self.config.get('location', 'Unknown')
            )

            if response:
                self.registered = True
                logger.info("Device registered successfully")
                return True
            else:
                logger.error("Device registration failed")
                return False

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def send_heartbeat(self) -> Optional[Dict]:
        """Send heartbeat with system metrics."""
        try:
            # Collect metrics
            metrics = self.metrics_collector.get_all_metrics()

            # Send heartbeat
            response = self.api_client.send_heartbeat(
                metrics=metrics,
                status="online",
                active_model_version=self.model_manager.active_model_version,
                ip_address=self._get_ip_address()
            )

            if response:
                self.last_heartbeat = datetime.utcnow()
                logger.debug(f"Heartbeat sent. Pending updates: {response.get('pending_updates', False)}")

                # Check for OTA updates
                if response.get('pending_updates') and self.config.get('auto_update', True):
                    deployment = response.get('available_deployment')
                    if deployment:
                        logger.info(f"OTA update available: {deployment.get('model_version')}")
                        self.handle_ota_update(deployment)

            return response

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return None

    def heartbeat_loop(self):
        """Continuous heartbeat loop (runs in separate thread)."""
        logger.info(f"Starting heartbeat loop (interval: {self.heartbeat_interval}s)")

        while self.running:
            try:
                self.send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)

    def handle_ota_update(self, deployment: Dict):
        """
        Handle OTA model update.

        Args:
            deployment: Deployment information from backend
        """
        try:
            deployment_id = deployment.get('deployment_id')
            formula_id = deployment.get('formula_id')
            model_version = deployment.get('model_version')
            download_url = deployment.get('download_url')

            logger.info(f"Starting OTA update: {model_version}")

            # Download model
            model_filename = f"model_{formula_id}_v{model_version}.pt"
            model_path = self.model_manager.models_dir / model_filename

            full_download_url = f"{self.api_client.base_url}{download_url}"
            success = self.api_client.download_model(full_download_url, str(model_path))

            if not success:
                logger.error("Model download failed")
                self.api_client.confirm_deployment(
                    deployment_id=deployment_id,
                    success=False,
                    error_message="Model download failed"
                )
                return

            # Load new model
            load_success = self.model_manager.load_model(
                model_path=str(model_path),
                model_id=formula_id,
                model_version=model_version
            )

            if load_success:
                logger.info(f"Model activated: {model_version}")

                # Save metadata
                self.model_manager.save_model_metadata({
                    'formula_id': formula_id,
                    'version': model_version,
                    'activated_at': datetime.utcnow().isoformat(),
                    'deployment_id': deployment_id
                })

                # Confirm deployment
                self.api_client.confirm_deployment(
                    deployment_id=deployment_id,
                    success=True
                )

                # Cleanup old models
                self.model_manager.cleanup_old_models(keep_latest=3)

                logger.info("OTA update completed successfully")
            else:
                logger.error("Model loading failed")
                self.api_client.confirm_deployment(
                    deployment_id=deployment_id,
                    success=False,
                    error_message="Model loading failed"
                )

        except Exception as e:
            logger.error(f"OTA update error: {e}")
            if 'deployment_id' in locals():
                self.api_client.confirm_deployment(
                    deployment_id=deployment_id,
                    success=False,
                    error_message=str(e)
                )

    def start(self):
        """Start the Jetson Client daemon."""
        logger.info("Starting Jetson Client...")

        # Check backend connectivity
        if not self.api_client.ping():
            logger.error("Cannot reach backend. Please check configuration.")
            return False

        logger.info("Backend is reachable")

        # Register device
        if not self.register():
            logger.error("Device registration failed")
            return False

        # Load existing model if available
        metadata = self.model_manager.load_model_metadata()
        if metadata:
            logger.info(f"Found existing model metadata: {metadata.get('version')}")

        # Start heartbeat thread
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        logger.info("Jetson Client started successfully")
        return True

    def stop(self):
        """Stop the Jetson Client daemon."""
        logger.info("Stopping Jetson Client...")
        self.running = False

        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)

        logger.info("Jetson Client stopped")

    def run(self):
        """Run the client (blocking)."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda sig, frame: self.stop())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.stop())

        if not self.start():
            logger.error("Failed to start Jetson Client")
            return 1

        logger.info("Jetson Client is running. Press Ctrl+C to stop.")

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        self.stop()
        return 0


def main():
    """Main entry point."""
    # Parse command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/etc/jetson-client/config.json"

    # Create and run client
    client = JetsonClient(config_path=config_path)
    return client.run()


if __name__ == "__main__":
    sys.exit(main())
