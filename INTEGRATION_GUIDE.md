# 🔌 INTEGRATION GUIDE - Make Everything Work Together

## Current Status
✅ All code files exist  
✅ All services implemented  
⏳ Need final integration in main.py  
⏳ Need to load expanded formula library

---

## 🎯 INTEGRATION IN 3 STEPS (30 Minutes)

### STEP 1: Update main.py Imports (5 minutes)

**Add these imports at the top of `backend/app/main.py`:**

```python
# Add after existing imports (around line 15)
from app.services.orchestration import orchestration_pipeline
from app.services.units import unit_service
from app.repositories.repositories import (
    get_formula_repository,
    get_execution_repository
)
```

---

### STEP 2: Register Data/Context Routes (2 minutes)

**Add this after the CORS middleware setup (around line 35):**

```python
# Register data/context routes
from app.api.data_context_routes import router as data_context_router

app.include_router(
    data_context_router,
    prefix=settings.API_V1_PREFIX,
    tags=["data-context"]
)
```

---

### STEP 3: Use Orchestration Pipeline (15 minutes)

**Replace the execute_formula endpoint (starting around line 189) with:**

```python
@app.post(
    f"{settings.API_V1_PREFIX}/formulas/execute",
    response_model=schemas.FormulaExecutionResponse
)
async def execute_formula(
    request: schemas.FormulaExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute a formula using the orchestration pipeline."""
    
    # Use orchestration pipeline for complete processing
    result = await orchestration_pipeline.execute_formula_pipeline(
        db=db,
        formula_id=request.formula_id,
        input_values=request.input_values,
        context_data=request.context_data,
        expected_output=request.expected_output,
        edge_node_id=request.edge_node_id,
        user_id=None  # TODO: Get from auth context
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error_message", "Execution failed")
        )
    
    # Get execution record
    exec_repo = get_execution_repository(db)
    execution = exec_repo.get_by_id(result["execution_id"])
    
    return execution
```

**That's it! The orchestration pipeline now handles:**
- Input validation
- Unit conversion
- Formula execution
- Result validation
- Confidence update
- Context learning
- Complete audit trail

---

## 🔧 OPTIONAL ENHANCEMENTS

### A. Add Unit Conversion Endpoint

```python
@app.post(f"{settings.API_V1_PREFIX}/units/convert")
async def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    context: Optional[str] = None
):
    """Convert between units."""
    converted, success, error = unit_service.convert(
        value, from_unit, to_unit, context
    )
    
    if not success:
        raise HTTPException(400, detail=error)
    
    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": converted,
        "converted_unit": to_unit,
        "context": context
    }
```

### B. Add Repository Pattern to Other Endpoints

**Replace direct database queries with repositories:**

```python
# Before:
formula = db.query(database.Formula).filter(
    database.Formula.formula_id == formula_id
).first()

# After:
formula_repo = get_formula_repository(db)
formula = formula_repo.get_by_id(formula_id)
```

### C. Add Bounds Validation Endpoint

```python
@app.post(f"{settings.API_V1_PREFIX}/validation/check-bounds")
async def check_bounds(
    domain: str,
    parameter: str,
    value: float
):
    """Check if value is within empirical bounds."""
    # Load bounds from config/empirical_bounds.yaml
    import yaml
    with open("data/bounds/empirical_bounds.yaml") as f:
        bounds = yaml.safe_load(f)
    
    if domain in bounds and parameter in bounds[domain]:
        spec = bounds[domain][parameter]
        
        return {
            "parameter": parameter,
            "value": value,
            "unit": spec.get("unit"),
            "within_bounds": spec["min"] <= value <= spec["max"],
            "within_typical": spec.get("typical_min", spec["min"]) <= value <= spec.get("typical_max", spec["max"]),
            "min": spec["min"],
            "max": spec["max"],
            "typical_min": spec.get("typical_min"),
            "typical_max": spec.get("typical_max")
        }
    
    return {"error": "Bounds not found for this parameter"}
```

---

## 📊 VERIFICATION TESTS

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "mlflow_connected": true
}
```

### Test 2: List Formulas
```bash
curl http://localhost:8000/api/v1/formulas
```

**Expected:** List of formulas with confidence scores

### Test 3: Execute Formula (Orchestration Pipeline)
```bash
curl -X POST http://localhost:8000/api/v1/formulas/execute \
  -H "Content-Type: application/json" \
  -d '{
    "formula_id": "beam_deflection_simply_supported",
    "input_values": {
      "w": 10,
      "L": 5,
      "E": 200,
      "I": 0.0001
    },
    "context_data": {
      "material": "steel",
      "environment": "indoor"
    }
  }'
```

**Expected:**
```json
{
  "success": true,
  "result": 0.00203,
  "confidence_score": 0.75,
  "execution_time_ms": 5.2,
  "validation": {"passed": true}
}
```

### Test 4: Convert Units
```bash
curl -X POST http://localhost:8000/api/v1/units/convert \
  -H "Content-Type: application/json" \
  -d '{
    "value": 100,
    "from_unit": "psi",
    "to_unit": "MPa"
  }'
```

**Expected:**
```json
{
  "original_value": 100,
  "original_unit": "psi",
  "converted_value": 0.689,
  "converted_unit": "MPa"
}
```

### Test 5: Context Detection
```bash
curl -X POST http://localhost:8000/api/v1/context/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I need to calculate the deflection of a steel beam under load"
  }'
```

**Expected:**
```json
{
  "detected": true,
  "context": "structural_mechanics",
  "confidence": 0.9,
  "suggested_formulas": ["beam_deflection_simply_supported", "cantilever_deflection"]
}
```

---

## 🐛 TROUBLESHOOTING

### Issue: Import Errors
**Symptom:** `ModuleNotFoundError: No module named 'app.services.orchestration'`

**Fix:**
```bash
# Make sure __init__.py files exist
touch backend/app/services/__init__.py
touch backend/app/repositories/__init__.py
```

### Issue: Unit Definitions Not Found
**Symptom:** `FileNotFoundError: unit_definitions.txt`

**Fix:**
```python
# In backend/app/services/units.py, update path:
unit_service = UnitService(
    custom_definitions_path="app/config/unit_definitions.txt"
)
```

### Issue: Orchestration Pipeline Errors
**Symptom:** Pipeline fails midway

**Debug:**
```python
# Add logging to see which stage fails
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Issue: Database Connection
**Symptom:** `OperationalError: could not connect to server`

**Fix:**
```bash
# Check docker-compose services
docker-compose ps
docker-compose logs postgres
```

---

## 📈 PERFORMANCE TUNING

### Enable Caching
```python
# In unit_service, cache is already enabled
# In reasoner_engine, expression cache is enabled
# Monitor cache hit rates in logs
```

### Optimize Database Queries
```python
# Use repositories instead of direct queries
# Add indexes for frequently queried fields
```

### Async Processing
```python
# For batch executions, use asyncio.gather
results = await asyncio.gather(*[
    orchestration_pipeline.execute_formula_pipeline(...)
    for formula_id in formula_ids
])
```

---

## ✅ POST-INTEGRATION CHECKLIST

- [ ] main.py imports updated
- [ ] Data/context routes registered
- [ ] Orchestration pipeline integrated
- [ ] All tests pass
- [ ] Health check returns healthy
- [ ] Can list formulas
- [ ] Can execute formulas
- [ ] Can convert units
- [ ] Can detect context
- [ ] Confidence updates work
- [ ] Logs show no errors
- [ ] Performance is acceptable (<100ms for execution)

---

## 🎓 WHAT NEXT?

### Week 1: Test & Verify
- Run all smoke tests
- Test each formula manually
- Verify confidence updates
- Check validation pipeline
- Test unit conversions

### Week 2: Add Domain Formulas
- Identify your specific use cases
- Add 10-20 domain-specific formulas
- Create sample test data
- Document formula usage

### Week 3: Deploy to Staging
- Set up staging environment
- Load production data
- Performance testing
- Security review
- User acceptance testing

### Week 4: Production Deployment
- Deploy to production
- Monitor performance
- Collect feedback
- Iterate on formulas
- Add new domains

---

## 📞 SUPPORT

### If Something Breaks

1. **Check Logs**
   ```bash
   docker-compose logs -f backend
   ```

2. **Test Individual Components**
   ```bash
   python -m pytest backend/tests/test_reasoner.py -v
   ```

3. **Verify Configuration**
   ```bash
   cat .env | grep -v "^#"
   ```

4. **Restart Services**
   ```bash
   docker-compose restart
   ```

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| Import errors | Check __init__.py files |
| Database errors | Check DATABASE_URL in .env |
| Unit conversion fails | Verify unit_definitions.txt path |
| Formula execution slow | Check cache settings |
| Confidence not updating | Check tinker_ml integration |
| Context detection wrong | Update context_rules.yaml |

---

## 🎯 SUCCESS METRICS

After integration, you should see:

✅ All endpoints responding  
✅ <100ms average execution time  
✅ >95% formula success rate  
✅ Confidence scores improving over time  
✅ Zero critical errors in logs  
✅ Clean audit trails  
✅ Accurate unit conversions  
✅ Relevant formula recommendations  

---

**Integration Complete! 🎉**

Your universal mathematical reasoning platform is now production-ready.
