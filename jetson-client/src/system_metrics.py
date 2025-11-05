"""
System Metrics Collector for Jetson AGX Orin 32GB.
Collects CPU, GPU, memory, disk, and temperature metrics.
"""
import os
import psutil
import subprocess
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class JetsonMetrics:
    """
    Collects system metrics from Jetson AGX Orin.
    Uses tegrastats for GPU metrics and psutil for system metrics.
    """

    def __init__(self):
        self.is_jetson = self._check_if_jetson()
        if not self.is_jetson:
            logger.warning("Not running on Jetson device. Some metrics will be simulated.")

    def _check_if_jetson(self) -> bool:
        """Check if running on a Jetson device."""
        return os.path.exists('/etc/nv_tegra_release') or os.path.exists('/sys/devices/virtual/thermal/thermal_zone0')

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage (all cores average)."""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {e}")
            return 0.0

    def get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            mem = psutil.virtual_memory()
            return mem.percent
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0

    def get_disk_usage(self) -> float:
        """Get disk usage percentage for root partition."""
        try:
            disk = psutil.disk_usage('/')
            return disk.percent
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return 0.0

    def get_gpu_usage(self) -> Optional[float]:
        """
        Get GPU usage percentage using tegrastats.
        AGX Orin has integrated Ampere GPU.
        """
        if not self.is_jetson:
            return None

        try:
            # Try reading from jetson_stats (if installed)
            # jetson_stats is the recommended way for Orin
            try:
                from jtop import jtop
                with jtop() as jetson:
                    if jetson.ok():
                        return float(jetson.stats['GPU'])
            except ImportError:
                logger.debug("jtop not available, trying tegrastats")

            # Fallback to tegrastats parsing
            result = subprocess.run(
                ['tegrastats', '--interval', '100'],
                capture_output=True,
                text=True,
                timeout=2
            )

            # Parse tegrastats output
            # Format: "GR3D_FREQ 99%"
            if 'GR3D_FREQ' in result.stdout:
                for part in result.stdout.split():
                    if '%' in part and part.replace('%', '').replace('.', '').isdigit():
                        return float(part.replace('%', ''))

            return None

        except subprocess.TimeoutExpired:
            logger.debug("tegrastats timeout")
            return None
        except Exception as e:
            logger.error(f"Error getting GPU usage: {e}")
            return None

    def get_temperature(self) -> Optional[float]:
        """
        Get device temperature in Celsius.
        AGX Orin has multiple thermal zones - returns the highest.
        """
        if not self.is_jetson:
            return None

        try:
            # Try reading from jetson_stats first
            try:
                from jtop import jtop
                with jtop() as jetson:
                    if jetson.ok():
                        temps = jetson.stats.get('Temp', {})
                        if temps:
                            # Return the maximum temperature
                            max_temp = max(temps.values()) if temps else None
                            return float(max_temp) if max_temp else None
            except ImportError:
                logger.debug("jtop not available, trying thermal zones")

            # Fallback to reading thermal zones directly
            thermal_zones = []
            for i in range(10):  # Check first 10 thermal zones
                temp_file = f'/sys/devices/virtual/thermal/thermal_zone{i}/temp'
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        # Temperature is in millidegrees
                        temp_millidegrees = int(f.read().strip())
                        temp_celsius = temp_millidegrees / 1000.0
                        thermal_zones.append(temp_celsius)

            if thermal_zones:
                return max(thermal_zones)  # Return highest temperature

            return None

        except Exception as e:
            logger.error(f"Error getting temperature: {e}")
            return None

    def get_all_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get all system metrics at once.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'cpu_usage_percent': self.get_cpu_usage(),
            'memory_usage_percent': self.get_memory_usage(),
            'disk_usage_percent': self.get_disk_usage(),
            'gpu_usage_percent': self.get_gpu_usage(),
            'temperature_celsius': self.get_temperature()
        }

        logger.debug(f"Collected metrics: {metrics}")
        return metrics

    def get_hardware_info(self) -> Dict[str, str]:
        """
        Get hardware information about the Jetson device.

        Returns:
            Dictionary with hardware details
        """
        info = {
            'hostname': os.uname().nodename,
            'os': 'Linux',
            'architecture': os.uname().machine,
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'total_disk_gb': round(psutil.disk_usage('/').total / (1024**3), 2)
        }

        # Try to get Jetson-specific info
        if self.is_jetson:
            try:
                # Read L4T version
                if os.path.exists('/etc/nv_tegra_release'):
                    with open('/etc/nv_tegra_release', 'r') as f:
                        l4t_version = f.read().strip()
                        info['l4t_version'] = l4t_version

                # Try to get Jetpack version from jtop
                try:
                    from jtop import jtop
                    with jtop() as jetson:
                        if jetson.ok():
                            info['jetpack_version'] = jetson.board.get('jetpack', 'Unknown')
                            info['device_model'] = jetson.board.get('hardware', {}).get('Model', 'Unknown')
                except ImportError:
                    pass

            except Exception as e:
                logger.error(f"Error getting Jetson info: {e}")

        return info


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    metrics_collector = JetsonMetrics()

    print("=== Jetson AGX Orin System Metrics ===")
    print(f"Is Jetson device: {metrics_collector.is_jetson}")
    print()

    print("Hardware Info:")
    hw_info = metrics_collector.get_hardware_info()
    for key, value in hw_info.items():
        print(f"  {key}: {value}")
    print()

    print("Current Metrics:")
    metrics = metrics_collector.get_all_metrics()
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: N/A")
