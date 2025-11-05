"""
API Client for communicating with the backend server.
Handles device registration, heartbeats, OTA updates, and formula execution.
"""
import requests
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """
    Client for communicating with the ML Platform backend.
    """

    def __init__(self, base_url: str, device_id: str, api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the backend (e.g., "https://api.example.com")
            device_id: Unique device identifier
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.device_id = device_id
        self.api_key = api_key
        self.session = requests.Session()

        # Set headers
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})

        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'JetsonClient/{device_id}'
        })

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        timeout: int = 30,
        retry_count: int = 3
    ) -> Optional[Dict]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Optional request data
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure

        Returns:
            Response JSON or None on failure
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(retry_count):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url, timeout=timeout, params=data)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=data, timeout=timeout)
                elif method.upper() == 'PATCH':
                    response = self.session.patch(url, json=data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retry_count}): {url}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All retries failed for {url}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error in request: {e}")
                return None

        return None

    def register_device(
        self,
        device_name: str,
        device_type: str,
        hardware_info: Dict,
        firmware_version: Optional[str] = None,
        os_version: Optional[str] = None,
        mac_address: Optional[str] = None,
        location: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Register this device with the backend.

        Args:
            device_name: Human-readable device name
            device_type: Device type (e.g., "jetson_orin")
            hardware_info: Hardware information dictionary
            firmware_version: Optional firmware version
            os_version: Optional OS version
            mac_address: Optional MAC address
            location: Optional physical location

        Returns:
            Registration response or None on failure
        """
        data = {
            'device_id': self.device_id,
            'device_name': device_name,
            'device_type': device_type,
            'hardware_info': hardware_info,
            'firmware_version': firmware_version,
            'os_version': os_version,
            'mac_address': mac_address,
            'location': location
        }

        logger.info(f"Registering device: {device_name} ({self.device_id})")
        response = self._make_request('POST', '/api/v1/edge-devices/register', data)

        if response:
            logger.info("Device registered successfully")
        else:
            logger.error("Device registration failed")

        return response

    def send_heartbeat(
        self,
        metrics: Dict[str, Optional[float]],
        status: str = "online",
        active_model_version: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Send heartbeat to backend with system metrics.

        Args:
            metrics: System metrics dictionary
            status: Device status (online, offline, error, maintenance)
            active_model_version: Currently active model version
            ip_address: Device IP address

        Returns:
            Heartbeat response with pending updates info
        """
        data = {
            'status': status,
            'cpu_usage_percent': metrics.get('cpu_usage_percent'),
            'memory_usage_percent': metrics.get('memory_usage_percent'),
            'disk_usage_percent': metrics.get('disk_usage_percent'),
            'gpu_usage_percent': metrics.get('gpu_usage_percent'),
            'temperature_celsius': metrics.get('temperature_celsius'),
            'active_model_version': active_model_version,
            'ip_address': ip_address,
            'metadata': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }

        response = self._make_request(
            'POST',
            f'/api/v1/edge-devices/{self.device_id}/heartbeat',
            data,
            timeout=10  # Shorter timeout for heartbeat
        )

        return response

    def check_for_updates(
        self,
        current_model_version: Optional[str] = None,
        current_tier: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Check for available OTA updates.

        Args:
            current_model_version: Current model version
            current_tier: Current model tier

        Returns:
            Update information or None
        """
        data = {
            'current_model_version': current_model_version,
            'current_tier': current_tier
        }

        response = self._make_request(
            'POST',
            f'/api/v1/edge-devices/{self.device_id}/check-updates',
            data
        )

        return response

    def download_model(self, download_url: str, output_path: str) -> bool:
        """
        Download a model file from the backend.

        Args:
            download_url: Full URL to download from
            output_path: Local path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading model from: {download_url}")

            # Use streaming download for large files
            response = self.session.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Download progress: {progress:.1f}%")

            logger.info(f"Model downloaded successfully: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False

    def confirm_deployment(
        self,
        deployment_id: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Confirm deployment completion to backend.

        Args:
            deployment_id: Deployment ID from update info
            success: Whether deployment succeeded
            error_message: Error message if failed

        Returns:
            Confirmation response or None
        """
        data = {
            'success': success,
            'error_message': error_message
        }

        response = self._make_request(
            'POST',
            f'/api/v1/edge-devices/{self.device_id}/deployments/{deployment_id}/confirm',
            data
        )

        return response

    def submit_execution_result(
        self,
        formula_id: int,
        input_values: Dict,
        output_values: Dict,
        execution_time_ms: float,
        status: str = "completed"
    ) -> Optional[Dict]:
        """
        Submit formula execution result to backend.

        Args:
            formula_id: Formula ID that was executed
            input_values: Input data
            output_values: Output/predictions
            execution_time_ms: Execution time in milliseconds
            status: Execution status

        Returns:
            Submission response or None
        """
        data = {
            'formula_id': formula_id,
            'input_values': input_values,
            'output_values': output_values,
            'execution_time_ms': execution_time_ms,
            'status': status,
            'edge_device_id': self.device_id
        }

        response = self._make_request(
            'POST',
            '/api/v1/formulas/execute',
            data
        )

        return response

    def ping(self) -> bool:
        """
        Ping the backend to check connectivity.

        Returns:
            True if backend is reachable, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ping failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Initialize client
    client = BackendAPIClient(
        base_url="http://localhost:8000",
        device_id="jetson-orin-001"
    )

    # Test ping
    if client.ping():
        print("✓ Backend is reachable")
    else:
        print("✗ Backend is not reachable")

    # Test registration
    hardware_info = {
        'model': 'AGX Orin 32GB',
        'total_memory_gb': 32,
        'architecture': 'aarch64'
    }

    result = client.register_device(
        device_name="Jetson Orin Dev Board",
        device_type="jetson_orin",
        hardware_info=hardware_info,
        os_version="Ubuntu 20.04 LTS",
        location="Lab A"
    )

    if result:
        print(f"✓ Device registered: {result}")
    else:
        print("✗ Device registration failed")
