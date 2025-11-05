"""
Load testing configuration and helper utilities.
"""
from dataclasses import dataclass
from typing import Dict, List
import json


@dataclass
class LoadTestConfig:
    """Configuration for load tests."""

    host: str = "http://localhost:8000"
    users: int = 100
    spawn_rate: int = 10
    run_time: str = "5m"


@dataclass
class TestScenario:
    """Test scenario configuration."""

    name: str
    users: int
    spawn_rate: int
    run_time: str
    tags: List[str]


# Predefined test scenarios
SCENARIOS = {
    "smoke": TestScenario(
        name="Smoke Test",
        users=10,
        spawn_rate=2,
        run_time="2m",
        tags=["read", "health"]
    ),
    "baseline": TestScenario(
        name="Baseline Load Test",
        users=50,
        spawn_rate=5,
        run_time="10m",
        tags=["read", "write"]
    ),
    "stress": TestScenario(
        name="Stress Test",
        users=200,
        spawn_rate=20,
        run_time="15m",
        tags=["stress"]
    ),
    "spike": TestScenario(
        name="Spike Test",
        users=500,
        spawn_rate=100,
        run_time="5m",
        tags=["stress"]
    ),
    "endurance": TestScenario(
        name="Endurance Test",
        users=100,
        spawn_rate=10,
        run_time="60m",
        tags=["read", "write"]
    ),
}


# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_p50": 200,  # ms
    "response_time_p95": 500,  # ms
    "response_time_p99": 1000,  # ms
    "error_rate": 0.01,  # 1%
    "requests_per_second": 100,
}


def get_scenario(name: str) -> TestScenario:
    """Get a test scenario by name."""
    return SCENARIOS.get(name, SCENARIOS["baseline"])


def generate_locust_command(scenario: str = "baseline") -> str:
    """Generate locust command for a given scenario."""
    config = get_scenario(scenario)
    tags = " ".join([f"--tags {tag}" for tag in config.tags])

    return f"""locust -f tests/load/locustfile.py \\
    --host {LoadTestConfig.host} \\
    --users {config.users} \\
    --spawn-rate {config.spawn_rate} \\
    --run-time {config.run_time} \\
    {tags} \\
    --html reports/load_test_{scenario}.html \\
    --csv reports/load_test_{scenario}
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        print(f"Locust command for '{scenario}' scenario:")
        print(generate_locust_command(scenario))
    else:
        print("Available scenarios:")
        for name, scenario in SCENARIOS.items():
            print(f"\n{name}:")
            print(f"  Users: {scenario.users}")
            print(f"  Spawn Rate: {scenario.spawn_rate}")
            print(f"  Run Time: {scenario.run_time}")
            print(f"  Tags: {', '.join(scenario.tags)}")
