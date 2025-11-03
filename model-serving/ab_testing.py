"""
A/B Testing Infrastructure for Model Serving

Utilities for comparing and testing multiple model versions:
- Traffic splitting
- Performance monitoring
- Statistical significance testing
- Champion/Challenger testing
- Multi-armed bandit
- Gradual rollout

Author: ML Framework Team
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import random
import time
from dataclasses import dataclass, field


# ============================================================================
# TRAFFIC ROUTER
# ============================================================================

@dataclass
class ModelVariant:
    """Model variant configuration."""
    name: str
    model: Any
    traffic_weight: float = 0.5
    inference_fn: Optional[Any] = None

    # Metrics
    total_requests: int = 0
    total_latency: float = 0.0
    errors: int = 0
    predictions: deque = field(default_factory=lambda: deque(maxlen=10000))


class TrafficRouter:
    """
    Route traffic between multiple model variants for A/B testing.
    """

    def __init__(
        self,
        variants: Dict[str, ModelVariant],
        routing_strategy: str = 'weighted'
    ):
        """
        Initialize traffic router.

        Parameters:
        -----------
        variants : dict
            Dictionary of {variant_name: ModelVariant}.
        routing_strategy : str
            Routing strategy: 'weighted', 'random', 'round_robin', 'epsilon_greedy'.
        """
        self.variants = variants
        self.routing_strategy = routing_strategy

        # Normalize weights
        total_weight = sum(v.traffic_weight for v in variants.values())
        for variant in variants.values():
            variant.traffic_weight /= total_weight

        # Round robin counter
        self.round_robin_counter = 0
        self.variant_names = list(variants.keys())

        # Epsilon-greedy parameters
        self.epsilon = 0.1
        self.best_variant = self.variant_names[0]

    def route(self, request_id: Optional[str] = None) -> str:
        """
        Select variant for request.

        Parameters:
        -----------
        request_id : str, optional
            Request ID for consistent routing.

        Returns:
        --------
        variant_name : str
            Selected variant name.
        """
        if self.routing_strategy == 'weighted':
            return self._weighted_route()
        elif self.routing_strategy == 'random':
            return random.choice(self.variant_names)
        elif self.routing_strategy == 'round_robin':
            return self._round_robin_route()
        elif self.routing_strategy == 'epsilon_greedy':
            return self._epsilon_greedy_route()
        elif self.routing_strategy == 'consistent_hash':
            return self._consistent_hash_route(request_id)
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def _weighted_route(self) -> str:
        """Weighted random selection."""
        rand = random.random()
        cumulative = 0.0

        for name, variant in self.variants.items():
            cumulative += variant.traffic_weight
            if rand <= cumulative:
                return name

        return self.variant_names[-1]

    def _round_robin_route(self) -> str:
        """Round-robin selection."""
        variant_name = self.variant_names[self.round_robin_counter % len(self.variant_names)]
        self.round_robin_counter += 1
        return variant_name

    def _epsilon_greedy_route(self) -> str:
        """Epsilon-greedy selection (multi-armed bandit)."""
        if random.random() < self.epsilon:
            # Explore: random variant
            return random.choice(self.variant_names)
        else:
            # Exploit: best variant
            return self.best_variant

    def _consistent_hash_route(self, request_id: str) -> str:
        """Consistent hashing for user stickiness."""
        if request_id is None:
            return self._weighted_route()

        hash_value = hash(request_id) % 100
        cumulative = 0.0

        for name, variant in self.variants.items():
            cumulative += variant.traffic_weight * 100
            if hash_value < cumulative:
                return name

        return self.variant_names[-1]

    def predict(
        self,
        inputs: Any,
        request_id: Optional[str] = None
    ) -> Tuple[Any, str]:
        """
        Route request and get prediction.

        Parameters:
        -----------
        inputs : Any
            Input data.
        request_id : str, optional
            Request ID.

        Returns:
        --------
        prediction : Any
            Model prediction.
        variant_name : str
            Variant that served the request.
        """
        # Select variant
        variant_name = self.route(request_id)
        variant = self.variants[variant_name]

        # Track request
        variant.total_requests += 1

        try:
            # Run inference
            start_time = time.time()

            if variant.inference_fn is not None:
                prediction = variant.inference_fn(inputs)
            else:
                prediction = variant.model.predict(inputs)

            latency = time.time() - start_time

            # Track metrics
            variant.total_latency += latency
            variant.predictions.append(prediction)

            return prediction, variant_name

        except Exception as e:
            variant.errors += 1
            raise e

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update traffic weights.

        Parameters:
        -----------
        new_weights : dict
            New weights for each variant.
        """
        total_weight = sum(new_weights.values())

        for name, weight in new_weights.items():
            if name in self.variants:
                self.variants[name].traffic_weight = weight / total_weight

    def update_best_variant(self):
        """Update best variant for epsilon-greedy (based on average latency)."""
        best_name = None
        best_latency = float('inf')

        for name, variant in self.variants.items():
            if variant.total_requests > 0:
                avg_latency = variant.total_latency / variant.total_requests
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    best_name = name

        if best_name:
            self.best_variant = best_name


# ============================================================================
# PERFORMANCE COMPARATOR
# ============================================================================

class ModelComparator:
    """
    Compare performance of model variants.
    """

    @staticmethod
    def compare_metrics(variants: Dict[str, ModelVariant]) -> Dict[str, Dict]:
        """
        Compare metrics across variants.

        Parameters:
        -----------
        variants : dict
            Model variants.

        Returns:
        --------
        comparison : dict
            Comparison metrics.
        """
        comparison = {}

        for name, variant in variants.items():
            if variant.total_requests > 0:
                avg_latency = variant.total_latency / variant.total_requests
                error_rate = variant.errors / variant.total_requests
            else:
                avg_latency = 0.0
                error_rate = 0.0

            comparison[name] = {
                'total_requests': variant.total_requests,
                'avg_latency_ms': avg_latency * 1000,
                'error_rate': error_rate,
                'traffic_weight': variant.traffic_weight
            }

        return comparison

    @staticmethod
    def statistical_significance_test(
        variant_a_predictions: List[float],
        variant_b_predictions: List[float],
        metric: str = 'mean'
    ) -> Dict[str, Any]:
        """
        Test for statistical significance between two variants.

        Parameters:
        -----------
        variant_a_predictions : list
            Predictions from variant A.
        variant_b_predictions : list
            Predictions from variant B.
        metric : str
            Metric to compare ('mean', 'median').

        Returns:
        --------
        results : dict
            Statistical test results.
        """
        a = np.array(variant_a_predictions)
        b = np.array(variant_b_predictions)

        # Basic statistics
        if metric == 'mean':
            stat_a = np.mean(a)
            stat_b = np.mean(b)
        else:  # median
            stat_a = np.median(a)
            stat_b = np.median(b)

        # Simple t-test approximation
        n_a, n_b = len(a), len(b)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

        # Pooled standard error
        se = np.sqrt(var_a/n_a + var_b/n_b)

        # T-statistic
        if se > 0:
            t_stat = (stat_a - stat_b) / se
        else:
            t_stat = 0.0

        # Approximate p-value (simplified)
        # For production, use scipy.stats.ttest_ind
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / np.sqrt(2))))

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((var_a + var_b) / 2)
        if pooled_std > 0:
            cohens_d = (stat_a - stat_b) / pooled_std
        else:
            cohens_d = 0.0

        return {
            'variant_a_stat': float(stat_a),
            'variant_b_stat': float(stat_b),
            'difference': float(stat_a - stat_b),
            'relative_diff_percent': float((stat_a - stat_b) / stat_b * 100) if stat_b != 0 else 0.0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'cohens_d': float(cohens_d),
            'sample_size_a': n_a,
            'sample_size_b': n_b
        }


# ============================================================================
# CHAMPION/CHALLENGER
# ============================================================================

class ChampionChallenger:
    """
    Champion/Challenger testing pattern.
    """

    def __init__(
        self,
        champion_model: Any,
        challenger_model: Any,
        challenger_traffic: float = 0.1,
        promotion_threshold: float = 0.05  # 5% improvement
    ):
        """
        Initialize Champion/Challenger testing.

        Parameters:
        -----------
        champion_model : Model
            Current production model (champion).
        challenger_model : Model
            New model to test (challenger).
        challenger_traffic : float
            Fraction of traffic to challenger (0.0 to 1.0).
        promotion_threshold : float
            Minimum improvement for promotion.
        """
        self.variants = {
            'champion': ModelVariant('champion', champion_model, 1 - challenger_traffic),
            'challenger': ModelVariant('challenger', challenger_model, challenger_traffic)
        }

        self.router = TrafficRouter(self.variants, routing_strategy='weighted')
        self.promotion_threshold = promotion_threshold

    def predict(self, inputs: Any) -> Tuple[Any, str]:
        """Route and predict."""
        return self.router.predict(inputs)

    def evaluate_promotion(self) -> Dict[str, Any]:
        """
        Evaluate whether to promote challenger to champion.

        Returns:
        --------
        evaluation : dict
            Promotion evaluation results.
        """
        champion = self.variants['champion']
        challenger = self.variants['challenger']

        # Ensure enough samples
        if champion.total_requests < 100 or challenger.total_requests < 100:
            return {
                'recommendation': 'wait',
                'reason': 'Not enough samples',
                'champion_requests': champion.total_requests,
                'challenger_requests': challenger.total_requests
            }

        # Compare latency
        champion_latency = champion.total_latency / champion.total_requests
        challenger_latency = challenger.total_latency / challenger.total_requests

        latency_improvement = (champion_latency - challenger_latency) / champion_latency

        # Compare error rates
        champion_error_rate = champion.errors / champion.total_requests
        challenger_error_rate = challenger.errors / challenger.total_requests

        evaluation = {
            'champion_latency_ms': champion_latency * 1000,
            'challenger_latency_ms': challenger_latency * 1000,
            'latency_improvement_percent': latency_improvement * 100,
            'champion_error_rate': champion_error_rate,
            'challenger_error_rate': challenger_error_rate,
            'champion_requests': champion.total_requests,
            'challenger_requests': challenger.total_requests
        }

        # Promotion decision
        if (latency_improvement >= self.promotion_threshold and
            challenger_error_rate <= champion_error_rate):
            evaluation['recommendation'] = 'promote'
            evaluation['reason'] = 'Challenger shows significant improvement'
        elif challenger_error_rate > champion_error_rate * 1.1:  # 10% worse
            evaluation['recommendation'] = 'rollback'
            evaluation['reason'] = 'Challenger has higher error rate'
        else:
            evaluation['recommendation'] = 'continue'
            evaluation['reason'] = 'Continue testing, no clear winner yet'

        return evaluation

    def promote_challenger(self):
        """Promote challenger to champion."""
        # Swap models
        self.variants['champion'].model = self.variants['challenger'].model

        # Reset metrics
        for variant in self.variants.values():
            variant.total_requests = 0
            variant.total_latency = 0.0
            variant.errors = 0
            variant.predictions.clear()

        print("Challenger promoted to champion")


# ============================================================================
# GRADUAL ROLLOUT
# ============================================================================

class GradualRollout:
    """
    Gradual rollout manager for safe model deployment.
    """

    def __init__(
        self,
        old_model: Any,
        new_model: Any,
        stages: List[float] = [0.05, 0.10, 0.25, 0.50, 1.0],
        stage_duration_minutes: int = 60,
        rollback_error_threshold: float = 0.05
    ):
        """
        Initialize gradual rollout.

        Parameters:
        -----------
        old_model : Model
            Current model.
        new_model : Model
            New model to roll out.
        stages : list
            Traffic percentages for each stage.
        stage_duration_minutes : int
            Duration of each stage in minutes.
        rollback_error_threshold : float
            Error rate threshold for automatic rollback.
        """
        self.variants = {
            'old': ModelVariant('old', old_model, 1.0),
            'new': ModelVariant('new', new_model, 0.0)
        }

        self.router = TrafficRouter(self.variants, routing_strategy='weighted')
        self.stages = stages
        self.stage_duration_minutes = stage_duration_minutes
        self.rollback_error_threshold = rollback_error_threshold

        self.current_stage = 0
        self.stage_start_time = time.time()

    def predict(self, inputs: Any) -> Tuple[Any, str]:
        """Route and predict."""
        return self.router.predict(inputs)

    def advance_stage(self) -> Dict[str, Any]:
        """
        Advance to next rollout stage.

        Returns:
        --------
        status : dict
            Rollout status.
        """
        # Check if enough time has passed
        time_elapsed_minutes = (time.time() - self.stage_start_time) / 60

        if time_elapsed_minutes < self.stage_duration_minutes:
            return {
                'status': 'in_progress',
                'current_stage': self.current_stage,
                'traffic_to_new': self.stages[self.current_stage] * 100,
                'time_remaining_minutes': self.stage_duration_minutes - time_elapsed_minutes
            }

        # Check error rate
        new_variant = self.variants['new']
        if new_variant.total_requests > 0:
            error_rate = new_variant.errors / new_variant.total_requests

            if error_rate > self.rollback_error_threshold:
                return {
                    'status': 'rollback',
                    'reason': f'Error rate {error_rate:.2%} exceeds threshold {self.rollback_error_threshold:.2%}',
                    'current_stage': self.current_stage
                }

        # Advance to next stage
        self.current_stage += 1

        if self.current_stage >= len(self.stages):
            return {
                'status': 'complete',
                'message': 'Rollout complete, 100% traffic to new model'
            }

        # Update traffic weights
        new_traffic = self.stages[self.current_stage]
        self.router.update_weights({
            'old': 1 - new_traffic,
            'new': new_traffic
        })

        self.stage_start_time = time.time()

        return {
            'status': 'advanced',
            'current_stage': self.current_stage,
            'traffic_to_new': new_traffic * 100,
            'message': f'Advanced to stage {self.current_stage+1}/{len(self.stages)}'
        }


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("A/B TESTING EXAMPLES")
    print("=" * 70)

    print("\n1. Traffic Router")
    print("-" * 70)
    print("""
# Create model variants
variants = {
    'model_v1': ModelVariant('v1', model_v1, traffic_weight=0.7),
    'model_v2': ModelVariant('v2', model_v2, traffic_weight=0.3)
}

# Create router
router = TrafficRouter(variants, routing_strategy='weighted')

# Route requests
for i in range(1000):
    prediction, variant = router.predict(input_data)
    print(f"Request {i} routed to {variant}")

# Compare performance
comparison = ModelComparator.compare_metrics(variants)
print(comparison)
""")

    print("\n2. Champion/Challenger")
    print("-" * 70)
    print("""
# Set up Champion/Challenger
cc = ChampionChallenger(
    champion_model=champion,
    challenger_model=challenger,
    challenger_traffic=0.1  # 10% to challenger
)

# Serve traffic
for request in requests:
    prediction, variant = cc.predict(request)

# Evaluate promotion
evaluation = cc.evaluate_promotion()
print(f"Recommendation: {evaluation['recommendation']}")
print(f"Reason: {evaluation['reason']}")

if evaluation['recommendation'] == 'promote':
    cc.promote_challenger()
""")

    print("\n3. Gradual Rollout")
    print("-" * 70)
    print("""
# Set up gradual rollout
rollout = GradualRollout(
    old_model=current_model,
    new_model=new_model,
    stages=[0.05, 0.10, 0.25, 0.50, 1.0],
    stage_duration_minutes=60
)

# Monitor and advance stages
while True:
    # Serve traffic
    for request in batch_requests:
        prediction, variant = rollout.predict(request)

    # Check if ready to advance
    status = rollout.advance_stage()
    print(f"Status: {status['status']}")

    if status['status'] == 'complete':
        print("Rollout complete!")
        break
    elif status['status'] == 'rollback':
        print(f"Rolling back: {status['reason']}")
        break

    time.sleep(60)  # Check every minute
""")

    print("\n" + "=" * 70)
    print("A/B testing examples completed!")
    print("=" * 70)
