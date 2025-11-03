"""
ADWIN Drift Detection and OTA (Over-The-Air) Updates for Reasoner AI Platform.

Implements:
- ADWIN algorithm for concept drift detection
- Automatic model retraining triggers
- OTA updates for edge nodes
- Formula versioning and rollback
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from river import drift
from loguru import logger
import asyncio
import httpx


class DriftDetector:
    """
    Drift detection using ADWIN (ADaptive WINdowing) algorithm.
    Monitors formula performance and triggers retraining when drift detected.
    """
    
    def __init__(self):
        # Store ADWIN instances per formula
        self.drift_detectors: Dict[str, drift.ADWIN] = {}
        
        # Drift detection thresholds
        self.drift_threshold = 0.002
        
        # Track drift events
        self.drift_history: List[Dict[str, Any]] = []
    
    def initialize_detector(self, formula_id: str, delta: float = 0.002):
        """
        Initialize ADWIN detector for a formula.
        
        Args:
            formula_id: Formula identifier
            delta: Confidence parameter (smaller = more sensitive)
        """
        self.drift_detectors[formula_id] = drift.ADWIN(delta=delta)
        logger.info(f"Initialized drift detector for formula: {formula_id}")
    
    def update(
        self,
        formula_id: str,
        success: bool,
        error_magnitude: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update drift detector with new execution result.
        
        Args:
            formula_id: Formula identifier
            success: Whether execution was successful
            error_magnitude: Relative error if available
            
        Returns:
            {
                "drift_detected": bool,
                "change_detected": bool,
                "warning": Optional[str],
                "recommendation": Optional[str]
            }
        """
        # Initialize if not exists
        if formula_id not in self.drift_detectors:
            self.initialize_detector(formula_id)
        
        detector = self.drift_detectors[formula_id]
        
        # Convert to metric (higher is better)
        # Use 1.0 for success, 0.0 for failure
        # Or use (1 - error) if error magnitude available
        if error_magnitude is not None:
            metric_value = max(0.0, 1.0 - error_magnitude)
        else:
            metric_value = 1.0 if success else 0.0
        
        # Update detector
        detector.update(metric_value)
        
        # Check for drift
        drift_detected = detector.drift_detected
        change_detected = detector.change_detected
        
        result = {
            "drift_detected": drift_detected,
            "change_detected": change_detected,
            "total_updates": detector.n_detections,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if drift_detected:
            # Log drift event
            drift_event = {
                "formula_id": formula_id,
                "detected_at": datetime.utcnow(),
                "n_detections": detector.n_detections,
                "metric_value": metric_value
            }
            self.drift_history.append(drift_event)
            
            result["warning"] = "Performance drift detected - formula may need retraining"
            result["recommendation"] = "Review recent executions and consider revalidation"
            
            logger.warning(f"Drift detected for formula {formula_id}")
        
        elif change_detected:
            result["warning"] = "Performance change detected - monitoring"
            logger.info(f"Performance change detected for formula {formula_id}")
        
        return result
    
    def get_drift_history(
        self,
        formula_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get drift detection history."""
        history = self.drift_history
        
        if formula_id:
            history = [h for h in history if h['formula_id'] == formula_id]
        
        if since:
            history = [h for h in history if h['detected_at'] >= since]
        
        return history
    
    def reset_detector(self, formula_id: str):
        """Reset drift detector (after retraining)."""
        if formula_id in self.drift_detectors:
            self.drift_detectors[formula_id] = drift.ADWIN(delta=self.drift_threshold)
            logger.info(f"Reset drift detector for formula: {formula_id}")


class OTAUpdateManager:
    """
    Over-The-Air update manager for edge nodes.
    Handles formula updates, version management, and rollback.
    """
    
    def __init__(self):
        self.edge_nodes: Dict[str, Dict[str, Any]] = {}
        self.update_queue: List[Dict[str, Any]] = []
        self.version_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_edge_node(
        self,
        node_id: str,
        node_url: str,
        node_capabilities: Dict[str, Any]
    ):
        """Register an edge node for OTA updates."""
        self.edge_nodes[node_id] = {
            "url": node_url,
            "capabilities": node_capabilities,
            "last_update": None,
            "current_formulas": {},
            "status": "online"
        }
        logger.info(f"Registered edge node: {node_id}")
    
    def queue_formula_update(
        self,
        formula_id: str,
        formula_data: Dict[str, Any],
        version: str,
        target_nodes: Optional[List[str]] = None,
        priority: str = "normal"
    ):
        """
        Queue formula update for edge nodes.
        
        Args:
            formula_id: Formula identifier
            formula_data: Complete formula definition
            version: Version string
            target_nodes: Specific nodes to update (None = all)
            priority: "critical", "high", "normal", "low"
        """
        update = {
            "update_id": f"{formula_id}_{version}_{datetime.utcnow().timestamp()}",
            "formula_id": formula_id,
            "formula_data": formula_data,
            "version": version,
            "target_nodes": target_nodes or list(self.edge_nodes.keys()),
            "priority": priority,
            "queued_at": datetime.utcnow(),
            "status": "queued"
        }
        
        self.update_queue.append(update)
        
        # Store version history
        if formula_id not in self.version_history:
            self.version_history[formula_id] = []
        
        self.version_history[formula_id].append({
            "version": version,
            "released_at": datetime.utcnow(),
            "formula_data": formula_data
        })
        
        logger.info(f"Queued OTA update: {update['update_id']}")
    
    async def process_update_queue(self):
        """Process queued updates and send to edge nodes."""
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        self.update_queue.sort(key=lambda x: priority_order[x["priority"]])
        
        processed = []
        
        for update in self.update_queue:
            if update["status"] != "queued":
                continue
            
            update["status"] = "processing"
            
            # Send to target nodes
            for node_id in update["target_nodes"]:
                if node_id not in self.edge_nodes:
                    continue
                
                node = self.edge_nodes[node_id]
                
                try:
                    success = await self._send_update_to_node(
                        node_url=node["url"],
                        formula_id=update["formula_id"],
                        formula_data=update["formula_data"],
                        version=update["version"]
                    )
                    
                    if success:
                        # Update node's current formulas
                        node["current_formulas"][update["formula_id"]] = update["version"]
                        node["last_update"] = datetime.utcnow()
                        logger.info(f"Successfully updated node {node_id} with formula {update['formula_id']}")
                    else:
                        logger.error(f"Failed to update node {node_id}")
                
                except Exception as e:
                    logger.error(f"Error updating node {node_id}: {e}")
            
            update["status"] = "completed"
            update["completed_at"] = datetime.utcnow()
            processed.append(update)
        
        # Remove processed updates
        self.update_queue = [u for u in self.update_queue if u["status"] != "completed"]
        
        return processed
    
    async def _send_update_to_node(
        self,
        node_url: str,
        formula_id: str,
        formula_data: Dict[str, Any],
        version: str
    ) -> bool:
        """Send update to a specific edge node."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{node_url}/update/formula",
                    json={
                        "formula_id": formula_id,
                        "formula_data": formula_data,
                        "version": version
                    },
                    timeout=30.0
                )
                
                return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Failed to send update to {node_url}: {e}")
            return False
    
    async def rollback_formula(
        self,
        formula_id: str,
        target_version: Optional[str] = None,
        target_nodes: Optional[List[str]] = None
    ) -> bool:
        """
        Rollback formula to previous version.
        
        Args:
            formula_id: Formula to rollback
            target_version: Specific version (None = previous version)
            target_nodes: Specific nodes (None = all nodes with this formula)
        """
        if formula_id not in self.version_history:
            logger.error(f"No version history for formula: {formula_id}")
            return False
        
        versions = self.version_history[formula_id]
        
        if not versions:
            logger.error(f"No versions available for rollback: {formula_id}")
            return False
        
        # Get target version
        if target_version:
            version_data = next((v for v in versions if v["version"] == target_version), None)
            if not version_data:
                logger.error(f"Version {target_version} not found for {formula_id}")
                return False
        else:
            # Get previous version (second to last)
            if len(versions) < 2:
                logger.error(f"No previous version available for {formula_id}")
                return False
            version_data = versions[-2]
        
        # Queue rollback update
        self.queue_formula_update(
            formula_id=formula_id,
            formula_data=version_data["formula_data"],
            version=version_data["version"],
            target_nodes=target_nodes,
            priority="critical"
        )
        
        logger.info(f"Queued rollback for {formula_id} to version {version_data['version']}")
        return True
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an edge node."""
        return self.edge_nodes.get(node_id)
    
    def get_all_nodes_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all edge nodes."""
        return self.edge_nodes


# Global instances
drift_detector = DriftDetector()
ota_manager = OTAUpdateManager()


# Example usage
if __name__ == "__main__":
    # Drift detection example
    detector = DriftDetector()
    
    # Simulate executions
    for i in range(100):
        # Simulate gradual performance degradation
        success = True if i < 50 else (i % 3 != 0)
        error = 0.02 if i < 50 else 0.10
        
        result = detector.update("formula_123", success, error)
        
        if result["drift_detected"]:
            print(f"Drift detected at execution {i}")
            print(f"Result: {result}")
    
    # OTA update example
    ota = OTAUpdateManager()
    
    # Register edge nodes
    ota.register_edge_node(
        node_id="edge_node_1",
        node_url="http://192.168.1.10:8080",
        node_capabilities={"cpu": "jetson_orin", "memory": "8GB"}
    )
    
    # Queue update
    ota.queue_formula_update(
        formula_id="concrete_strength",
        formula_data={"expression": "...", "params": {}},
        version="1.2.0",
        priority="high"
    )
    
    # Process updates (would be run in async loop)
    # await ota.process_update_queue()
