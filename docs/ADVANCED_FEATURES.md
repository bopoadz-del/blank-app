# Advanced Features: Credibility Tiers, MLflow, ADWIN & OTA

## Overview

The Reasoner AI Platform now includes **complete implementations** of advanced features:

1. **Credibility Tier System** - 5-tier formula trust hierarchy with automatic promotion/demotion
2. **MLflow Integration** - Complete experiment tracking and metrics
3. **ADWIN Drift Detection** - Automatic performance degradation detection
4. **OTA Updates** - Over-the-air formula updates for edge nodes

---

## 1. Credibility Tier System

### Five-Tier Hierarchy

```
TIER 1: Auto-Deploy (No Review)
  ├─ Confidence: ≥95%
  ├─ Executions: ≥100
  ├─ Success Rate: ≥98%
  ├─ Source: ISO/ASTM standards
  └─ Policy: Immediate deployment, no notification

TIER 2: Monitored Deploy
  ├─ Confidence: ≥90%
  ├─ Executions: ≥50
  ├─ Success Rate: ≥95%
  ├─ Source: ISO/National codes/Peer-reviewed
  └─ Policy: Auto-deploy with notification

TIER 3: Review Required
  ├─ Confidence: ≥80%
  ├─ Executions: ≥25
  ├─ Success Rate: ≥90%
  ├─ Source: Peer-reviewed/Consultant
  └─ Policy: Human review before deployment

TIER 4: Validation Required
  ├─ Confidence: ≥70%
  ├─ Executions: ≥10
  ├─ Success Rate: ≥85%
  ├─ Source: Internal/AI-discovered
  └─ Policy: Validation + review required

TIER 5: Sandbox Only
  ├─ Confidence: Any
  ├─ Executions: Any
  ├─ Success Rate: Any
  ├─ Source: Unverified/User-submitted
  └─ Policy: Testing only, no production
```

### Automatic Tier Assignment

```python
from app.services.credibility_tiers import credibility_manager

# New formula from ISO standard
tier = credibility_manager.determine_initial_tier(
    source="ISO_10243",
    validation_passed=True,
    peer_reviewed=False
)
# Returns: TIER_2 (monitored deploy)

# New AI-discovered formula
tier = credibility_manager.determine_initial_tier(
    source="AI_discovered",
    validation_passed=False
)
# Returns: TIER_5 (sandbox only)
```

### Automatic Promotion

System automatically evaluates promotion after each execution:

```python
# After 75 successful executions
promotion = credibility_manager.evaluate_tier_promotion(
    current_tier="tier_3_review_required",
    confidence_score=0.92,
    total_executions=75,
    successful_executions=72,
    validation_stages_passed=["syntactic", "dimensional", "physical", "empirical"],
    source="ISO_10243"
)

# Returns:
# {
#   "promote": True,
#   "new_tier": "tier_2_monitored_deploy",
#   "reason": "Promoted to tier_2: confidence=0.92, executions=75, success_rate=0.96",
#   "auto_deploy": True
# }
```

### Automatic Demotion

System demotes formulas when performance degrades:

```python
# After recent failures
demotion = credibility_manager.evaluate_tier_demotion(
    current_tier="tier_2_monitored_deploy",
    confidence_score=0.85,  # Dropped below 0.90
    total_executions=120,
    successful_executions=110,
    recent_failures=5,  # 5 failures in last 10 executions
    validation_failures=["physical", "empirical"]
)

# Returns:
# {
#   "demote": True,
#   "new_tier": "tier_3_review_required",
#   "reason": "Demoted: confidence dropped to 0.85, 5 recent failures",
#   "suspend": False,
#   "requires_review": True
# }
```

### Deployment Policies by Tier

```python
# Get policy for a tier
policy = credibility_manager.get_deployment_policy("tier_2_monitored_deploy")

# Returns:
# {
#   "auto_deploy": True,
#   "requires_review": False,
#   "requires_validation": False,
#   "notification_required": True,
#   "sandbox_only": False,
#   "max_concurrent_deployments": 100
# }
```

### Database Schema

```python
# Formula model now includes:
class Formula:
    credibility_tier = Column(String(50), default="tier_5_sandbox_only")
    tier_updated_at = Column(DateTime)
    tier_change_reason = Column(Text)
```

### Migration

```bash
# Add tier fields to existing database
docker-compose exec backend python -m app.core.migrate_add_tiers
```

---

## 2. MLflow Integration

### Complete Experiment Tracking

```python
from app.services.mlflow_tracking import mlflow_tracker

# Track formula execution
run_id = mlflow_tracker.track_formula_execution(
    formula_id="concrete_strength",
    formula_name="Concrete Compressive Strength",
    input_values={"S_ultimate": 50, "k": 0.005, "maturity": 2000},
    output_values=46.6,
    context={"climate": "hot_arid", "material": "concrete"},
    execution_time=0.023,
    success=True,
    confidence_score=0.92,
    validation_passed=True
)
```

### Track Confidence Updates

```python
mlflow_tracker.track_confidence_update(
    formula_id="concrete_strength",
    old_confidence=0.90,
    new_confidence=0.92,
    reason="Successful execution in hot_arid context",
    total_executions=75,
    success_rate=0.96
)
```

### Track Validation Results

```python
mlflow_tracker.track_validation(
    formula_id="concrete_strength",
    validation_stages=[
        {"stage": "syntactic", "passed": True, "confidence": 1.0},
        {"stage": "dimensional", "passed": True, "confidence": 0.95},
        {"stage": "physical", "passed": True, "confidence": 0.92},
        {"stage": "empirical", "passed": True, "confidence": 0.98},
        {"stage": "operational", "passed": True, "confidence": 0.90}
    ],
    overall_passed=True
)
```

### Track Tier Changes

```python
mlflow_tracker.track_tier_change(
    formula_id="concrete_strength",
    old_tier="tier_3_review_required",
    new_tier="tier_2_monitored_deploy",
    reason="Performance threshold exceeded",
    auto_deploy=True
)
```

### Get Historical Metrics

```python
metrics = mlflow_tracker.get_formula_metrics("concrete_strength", limit=100)

# Returns:
# {
#   "formula_id": "concrete_strength",
#   "total_runs": 75,
#   "confidence_history": [
#     {"timestamp": 1234567890, "confidence": 0.92},
#     ...
#   ],
#   "average_execution_time": 0.025,
#   "success_rate": 0.96,
#   "last_run_time": 1234567900
# }
```

### Access MLflow UI

```bash
# MLflow UI available at:
open http://localhost:5000

# View:
# - All experiments
# - Confidence trends
# - Execution history
# - Performance metrics
# - Tier changes
```

---

## 3. ADWIN Drift Detection

### Automatic Performance Monitoring

```python
from app.services.drift_detection import drift_detector

# System automatically monitors each execution
result = drift_detector.update(
    formula_id="concrete_strength",
    success=True,
    error_magnitude=0.02  # 2% error
)

# Returns:
# {
#   "drift_detected": False,  # or True if drift detected
#   "change_detected": False,  # Early warning
#   "total_updates": 75,
#   "timestamp": "2024-11-02T13:00:00Z",
#   "warning": None,  # or "Performance drift detected..."
#   "recommendation": None  # or "Review recent executions..."
# }
```

### Drift Detection Example

```python
# After 100 executions with gradual performance degradation
for i in range(100):
    success = i < 80  # Performance drops after 80 executions
    error = 0.02 if i < 80 else 0.15
    
    result = drift_detector.update("formula_123", success, error)
    
    if result["drift_detected"]:
        print(f"🚨 Drift detected at execution {i}")
        print(f"Recommendation: {result['recommendation']}")
        # Trigger: Revalidation, retraining, or tier demotion
```

### Get Drift History

```python
history = drift_detector.get_drift_history(
    formula_id="concrete_strength",
    since=datetime.utcnow() - timedelta(days=7)
)

# Returns list of drift events
```

### Reset After Retraining

```python
# After revalidating or updating formula
drift_detector.reset_detector("concrete_strength")
```

### Integration with Tier System

```python
# Drift detection triggers tier review
if result["drift_detected"]:
    # Automatically demote one tier
    demotion = credibility_manager.evaluate_tier_demotion(
        current_tier=formula.credibility_tier,
        ...
        recent_failures=5
    )
    
    if demotion["demote"]:
        formula.credibility_tier = demotion["new_tier"]
```

---

## 4. OTA (Over-The-Air) Updates

### Register Edge Nodes

```python
from app.services.drift_detection import ota_manager

# Register Jetson edge node
ota_manager.register_edge_node(
    node_id="jetson_orin_1",
    node_url="http://192.168.1.10:8080",
    node_capabilities={
        "hardware": "jetson_orin_nano",
        "memory": "8GB",
        "storage": "128GB",
        "location": "construction_site_a"
    }
)
```

### Queue Formula Updates

```python
# New formula version ready
ota_manager.queue_formula_update(
    formula_id="concrete_strength",
    formula_data={
        "expression": "S_ultimate * (1 - exp(-k * maturity))",
        "input_parameters": {...},
        "output_parameters": {...},
        "version": "1.2.0"
    },
    version="1.2.0",
    target_nodes=["jetson_orin_1", "jetson_orin_2"],
    priority="high"  # "critical", "high", "normal", "low"
)
```

### Process Updates

```python
# Run in background (async)
import asyncio

async def update_loop():
    while True:
        processed = await ota_manager.process_update_queue()
        print(f"Processed {len(processed)} updates")
        await asyncio.sleep(60)  # Check every minute

# Start update loop
asyncio.create_task(update_loop())
```

### Rollback Formula

```python
# Rollback to previous version if issues detected
success = await ota_manager.rollback_formula(
    formula_id="concrete_strength",
    target_version="1.1.0",  # or None for previous version
    target_nodes=["jetson_orin_1"]
)
```

### Check Node Status

```python
# Get status of specific node
status = ota_manager.get_node_status("jetson_orin_1")

# Returns:
# {
#   "url": "http://192.168.1.10:8080",
#   "capabilities": {...},
#   "last_update": "2024-11-02T13:00:00Z",
#   "current_formulas": {
#     "concrete_strength": "1.2.0",
#     "beam_deflection": "2.0.1"
#   },
#   "status": "online"
# }

# Get all nodes
all_nodes = ota_manager.get_all_nodes_status()
```

### Version Management

```python
# View version history
history = ota_manager.version_history["concrete_strength"]

# Returns:
# [
#   {"version": "1.0.0", "released_at": "2024-10-01T...", "formula_data": {...}},
#   {"version": "1.1.0", "released_at": "2024-10-15T...", "formula_data": {...}},
#   {"version": "1.2.0", "released_at": "2024-11-02T...", "formula_data": {...}}
# ]
```

---

## 5. Complete Integration Example

### Workflow: Formula Execution → Learning → Tier Update → OTA

```python
# 1. Execute formula
from app.services.reasoner import reasoner_engine
from app.services.tinker import tinker_ml
from app.services.credibility_tiers import credibility_manager
from app.services.mlflow_tracking import mlflow_tracker
from app.services.drift_detection import drift_detector, ota_manager

# Execute
result = await reasoner_engine.execute_formula(
    formula_expression=formula.formula_expression,
    input_values={"S_ultimate": 50, "k": 0.005, "maturity": 2000},
    context={"climate": "hot_arid"}
)

# 2. Track in MLflow
run_id = mlflow_tracker.track_formula_execution(
    formula_id=formula.formula_id,
    formula_name=formula.name,
    input_values=input_values,
    output_values=result["result"],
    context=context,
    execution_time=result["execution_time"],
    success=result["success"],
    confidence_score=formula.confidence_score
)

# 3. Update confidence (Tinker ML)
confidence_update = await tinker_ml.update_confidence_from_execution(
    db=db,
    formula_id=formula.id,
    execution_success=result["success"],
    context=context
)

# 4. Check for drift (ADWIN)
drift_result = drift_detector.update(
    formula_id=formula.formula_id,
    success=result["success"],
    error_magnitude=0.02
)

if drift_result["drift_detected"]:
    # Performance degrading - review tier
    logger.warning(f"Drift detected for {formula.formula_id}")

# 5. Evaluate tier promotion/demotion
if formula.total_executions % 10 == 0:  # Check every 10 executions
    promotion = credibility_manager.evaluate_tier_promotion(
        current_tier=formula.credibility_tier,
        confidence_score=formula.confidence_score,
        total_executions=formula.total_executions,
        successful_executions=formula.successful_executions,
        validation_stages_passed=formula.validation_stages_passed,
        source=formula.source
    )
    
    if promotion["promote"]:
        # Update tier
        old_tier = formula.credibility_tier
        formula.credibility_tier = promotion["new_tier"].value
        formula.tier_updated_at = datetime.utcnow()
        formula.tier_change_reason = promotion["reason"]
        
        # Track in MLflow
        mlflow_tracker.track_tier_change(
            formula_id=formula.formula_id,
            old_tier=old_tier,
            new_tier=formula.credibility_tier,
            reason=promotion["reason"],
            auto_deploy=promotion["auto_deploy"]
        )
        
        # 6. Queue OTA update if auto-deploy enabled
        if promotion["auto_deploy"]:
            ota_manager.queue_formula_update(
                formula_id=formula.formula_id,
                formula_data={
                    "expression": formula.formula_expression,
                    "input_parameters": formula.input_parameters,
                    "output_parameters": formula.output_parameters,
                    "version": formula.version
                },
                version=formula.version,
                priority="high"
            )
```

---

## 6. Configuration

### Environment Variables

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=reasoner-formulas

# Credibility tiers
AUTO_DEPLOY_CONFIDENCE_THRESHOLD=0.95
HUMAN_REVIEW_CONFIDENCE_THRESHOLD=0.70

# Drift detection
ADWIN_DELTA=0.002  # Sensitivity (smaller = more sensitive)
DRIFT_CHECK_INTERVAL=10  # Check every N executions

# OTA
OTA_UPDATE_INTERVAL=300  # Seconds between update checks
OTA_ROLLBACK_TIMEOUT=60  # Seconds to wait for rollback
```

---

## 7. Monitoring

### Dashboard Metrics

- **Tier Distribution**: Number of formulas in each tier
- **Promotion Rate**: Formulas promoted in last 7 days
- **Demotion Rate**: Formulas demoted in last 7 days
- **Drift Alerts**: Active drift warnings
- **OTA Status**: Pending updates, node health

### Alerts

```python
# Setup alerts for critical events
if tier_change["new_tier"] == "tier_5_sandbox_only":
    send_alert("Formula suspended due to performance issues")

if drift_result["drift_detected"]:
    send_alert(f"Drift detected: {formula.name}")

if ota_update_failed:
    send_alert(f"OTA update failed for node: {node_id}")
```

---

## Conclusion

The Reasoner AI Platform now has **complete, production-ready implementations** of:

✅ **5-Tier Credibility System** - Automatic trust management  
✅ **MLflow Integration** - Complete experiment tracking  
✅ **ADWIN Drift Detection** - Performance monitoring  
✅ **OTA Updates** - Edge node formula updates  

**All systems are fully integrated and operational.**
