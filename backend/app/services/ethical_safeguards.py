"""
Ethical Safeguards Service.
Implements red lines, context-aware overrides, and ethical decision tracking.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
import uuid
import logging

from ..models.ethical_layer import (
    EthicalOverride,
    EthicalConfiguration,
    EthicalAuditLog,
    KnowledgeSource,
    CredibilityTier
)
from ..models.database import Formula, FormulaTier
from ..models.auth import User

logger = logging.getLogger(__name__)


class EthicalSafeguards:
    """
    Enforces ethical safeguards, red lines, and context-aware overrides.
    """

    # Red lines that can never be overridden
    RED_LINES = {
        "human_safety_tier_1_only": {
            "description": "Human safety calculations must use Tier 1 sources only",
            "domains": ["structural", "safety", "life_safety", "emergency"],
            "required_tier": 1
        },
        "environmental_compliance_gov_only": {
            "description": "Environmental compliance requires government sources",
            "domains": ["environmental", "epa", "emissions"],
            "source_types": ["government_regulation", "iso_standard"]
        },
        "financial_audited_only": {
            "description": "Financial reporting uses audited formulas only",
            "domains": ["financial", "accounting", "reporting"],
            "required_verified": True
        },
        "no_automated_hr_decisions": {
            "description": "No automation of hiring/firing decisions",
            "domains": ["hr", "hiring", "employment"],
            "requires_human": True
        },
        "medical_expert_review": {
            "description": "Medical or health impacts require expert review",
            "domains": ["medical", "health", "safety"],
            "requires_expert_committee": True
        }
    }

    def __init__(self, db: Session, deployment_name: str = "default"):
        """
        Initialize Ethical Safeguards.

        Args:
            db: Database session
            deployment_name: Deployment identifier (e.g., "diriyah", "mssdppg")
        """
        self.db = db
        self.deployment_name = deployment_name
        self.config = self._load_config()

    def _load_config(self) -> EthicalConfiguration:
        """Load or create ethical configuration for this deployment."""
        config = self.db.query(EthicalConfiguration).filter(
            EthicalConfiguration.deployment_name == self.deployment_name
        ).first()

        if not config:
            # Create default configuration
            config = EthicalConfiguration(
                deployment_name=self.deployment_name,
                default_tier=2,  # Conservative start
                auto_promotion_enabled=True,
                safety_margin=1.2,  # 20% safety factor
                domain_min_tiers={
                    "structural": 1,
                    "scheduling": 2,
                    "cost": 2,
                    "logistics": 3
                },
                red_lines=list(self.RED_LINES.keys()),
                audit_frequency="weekly"
            )
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)

        return config

    def check_red_lines(
        self,
        formula: Formula,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Check if formula execution would violate any red lines.

        Args:
            formula: Formula to check
            context: Execution context with domain, environment, etc.

        Returns:
            Tuple of (passes_red_lines, violated_red_lines)
        """
        violations = []
        domain = context.get('domain', '').lower()

        for red_line_id, red_line in self.RED_LINES.items():
            # Check if this red line applies to this domain
            if not any(d in domain for d in red_line.get('domains', [])):
                continue

            # Check required tier
            if 'required_tier' in red_line:
                if formula.tier.value > red_line['required_tier']:
                    violations.append(
                        f"{red_line['description']}: Formula is Tier {formula.tier.value}, requires Tier {red_line['required_tier']}"
                    )

            # Check source types
            if 'source_types' in red_line:
                source_id = formula.metadata.get('source_id') if formula.metadata else None
                if source_id:
                    source = self.db.query(KnowledgeSource).filter(
                        KnowledgeSource.source_id == source_id
                    ).first()

                    if source and source.source_type.value not in red_line['source_types']:
                        violations.append(
                            f"{red_line['description']}: Source type {source.source_type.value} not allowed"
                        )

            # Check verification requirement
            if red_line.get('required_verified', False):
                if not formula.metadata or not formula.metadata.get('verified', False):
                    violations.append(
                        f"{red_line['description']}: Formula must be verified"
                    )

            # Check human requirement
            if red_line.get('requires_human', False):
                violations.append(
                    f"{red_line['description']}: Human decision required"
                )

            # Check expert committee requirement
            if red_line.get('requires_expert_committee', False):
                if not context.get('expert_committee_approved', False):
                    violations.append(
                        f"{red_line['description']}: Expert committee approval required"
                    )

        passes = len(violations) == 0
        return passes, violations

    def apply_context_overrides(
        self,
        formula: Formula,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply context-aware overrides.

        Args:
            formula: Formula being executed
            context: Execution context

        Returns:
            Modified context with adjustments applied
        """
        modifications = {
            'original_credibility': None,
            'adjusted_credibility': None,
            'safety_margin': self.config.safety_margin,
            'overrides_applied': []
        }

        # Get formula source credibility if available
        source_id = formula.metadata.get('source_id') if formula.metadata else None
        if source_id:
            source = self.db.query(KnowledgeSource).filter(
                KnowledgeSource.source_id == source_id
            ).first()

            if source:
                original_credibility = source.credibility_score
                adjusted_credibility = original_credibility

                # Find applicable overrides
                overrides = self.db.query(EthicalOverride).filter(
                    EthicalOverride.is_active == True
                ).order_by(EthicalOverride.priority.desc()).all()

                for override in overrides:
                    # Check if override applies
                    if not self._override_applies(override, context):
                        continue

                    # Apply credibility adjustment
                    adjusted_credibility += override.credibility_adjustment
                    adjusted_credibility = max(0.0, min(1.0, adjusted_credibility))

                    # Apply safety margin
                    if override.safety_margin_multiplier != 1.0:
                        modifications['safety_margin'] *= override.safety_margin_multiplier

                    modifications['overrides_applied'].append({
                        'override_id': override.override_id,
                        'name': override.override_name,
                        'credibility_adjustment': override.credibility_adjustment,
                        'safety_margin': override.safety_margin_multiplier
                    })

                modifications['original_credibility'] = original_credibility
                modifications['adjusted_credibility'] = adjusted_credibility

        return modifications

    def _override_applies(self, override: EthicalOverride, context: Dict) -> bool:
        """Check if an override applies to the given context."""
        # Check deployment scope
        if override.applicable_deployments:
            if self.deployment_name not in override.applicable_deployments and "all" not in override.applicable_deployments:
                return False

        # Check domain scope
        domain = context.get('domain', '').lower()
        if override.applicable_domains:
            if not any(d in domain for d in override.applicable_domains):
                return False

        # Check trigger conditions
        trigger_conditions = override.trigger_conditions or {}

        for condition_key, condition_value in trigger_conditions.items():
            context_value = context.get(condition_key)

            # Simple equality check (can be extended for complex conditions)
            if context_value != condition_value:
                return False

        return True

    def check_domain_min_tier(self, domain: str, tier: int) -> Tuple[bool, Optional[str]]:
        """
        Check if tier meets domain minimum requirement.

        Args:
            domain: Domain name
            tier: Proposed tier

        Returns:
            Tuple of (passes, violation_message)
        """
        domain_min_tiers = self.config.domain_min_tiers or {}

        min_tier = domain_min_tiers.get(domain.lower())
        if min_tier and tier > min_tier:
            return False, f"Domain '{domain}' requires minimum Tier {min_tier}, formula is Tier {tier}"

        return True, None

    def log_ethical_decision(
        self,
        decision_type: str,
        formula: Formula,
        context: Dict,
        modifications: Dict,
        user: Optional[User] = None
    ) -> EthicalAuditLog:
        """
        Log an ethical decision with full transparency.

        Args:
            decision_type: Type of decision (formula_approval, tier_assignment, etc.)
            formula: Formula involved
            context: Execution context
            modifications: Applied modifications
            user: User who approved (if applicable)

        Returns:
            Created audit log entry
        """
        # Build decision explanation
        explanation_parts = [f"Decision type: {decision_type}"]

        if modifications.get('adjusted_credibility'):
            explanation_parts.append(
                f"Credibility adjusted from {modifications['original_credibility']:.3f} to {modifications['adjusted_credibility']:.3f}"
            )

        if modifications.get('safety_margin') != 1.0:
            explanation_parts.append(
                f"Safety margin applied: {modifications['safety_margin']:.2f}x"
            )

        if modifications.get('overrides_applied'):
            override_names = [o['name'] for o in modifications['overrides_applied']]
            explanation_parts.append(f"Overrides: {', '.join(override_names)}")

        explanation = "; ".join(explanation_parts)

        # Get source information
        sources_cited = []
        source_credibility_scores = {}

        if formula.metadata and 'source_id' in formula.metadata:
            source_id = formula.metadata['source_id']
            source = self.db.query(KnowledgeSource).filter(
                KnowledgeSource.source_id == source_id
            ).first()

            if source:
                sources_cited.append({
                    'source_id': source.source_id,
                    'source_name': source.source_name,
                    'source_type': source.source_type.value
                })
                source_credibility_scores[source.source_id] = source.credibility_score

        # Create audit log
        audit_log = EthicalAuditLog(
            audit_id=f"ethical_{uuid.uuid4().hex[:12]}",
            decision_type=decision_type,
            formula_id=formula.id,
            credibility_tier_assigned=formula.tier.value if hasattr(formula.tier, 'value') else formula.tier,
            source_credibility_scores=source_credibility_scores,
            overrides_applied=[o['override_id'] for o in modifications.get('overrides_applied', [])],
            red_lines_checked=list(self.RED_LINES.keys()),
            required_human_approval=context.get('requires_human_approval', False),
            approved_by_user_id=user.id if user else None,
            approval_timestamp=datetime.utcnow() if user else None,
            deployment_name=self.deployment_name,
            domain=context.get('domain'),
            context_factors=context,
            decision_explanation=explanation,
            sources_cited=sources_cited
        )

        self.db.add(audit_log)
        self.db.commit()
        self.db.refresh(audit_log)

        logger.info(f"Ethical decision logged: {audit_log.audit_id} - {decision_type}")

        return audit_log

    def can_auto_execute(
        self,
        formula: Formula,
        context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if formula can be auto-executed without human approval.

        Args:
            formula: Formula to check
            context: Execution context

        Returns:
            Tuple of (can_auto_execute, reason)
        """
        # Check red lines
        passes_red_lines, violations = self.check_red_lines(formula, context)
        if not passes_red_lines:
            return False, f"Red line violations: {'; '.join(violations)}"

        # Check tier requirements
        domain = context.get('domain', '').lower()
        passes_tier, tier_message = self.check_domain_min_tier(domain, formula.tier.value if hasattr(formula.tier, 'value') else formula.tier)
        if not passes_tier:
            return False, tier_message

        # Apply overrides
        modifications = self.apply_context_overrides(formula, context)

        # Determine if human approval needed based on tier
        if formula.tier == FormulaTier.TIER_1_CERTIFIED:
            # Tier 1: Fully automated
            return True, "Tier 1 - Fully automated"
        elif formula.tier == FormulaTier.TIER_2_VALIDATED:
            # Tier 2: Supervised execution - manager approval
            return False, "Tier 2 - Manager approval required"
        elif formula.tier == FormulaTier.TIER_3_TESTING:
            # Tier 3: Expert committee required
            return False, "Tier 3 - Expert committee approval required (2+ experts)"
        else:  # Tier 4
            # Tier 4: Research mode - sandbox only
            return False, "Tier 4 - Sandbox only, no production deployment"


# Prebuilt context-aware overrides
STANDARD_OVERRIDES = [
    {
        "override_id": "extreme_weather",
        "override_category": "environmental",
        "override_name": "Extreme Weather Conditions",
        "trigger_conditions": {"weather": "extreme"},
        "credibility_adjustment": -0.15,
        "safety_margin_multiplier": 1.2,
        "applicable_domains": ["structural", "outdoor", "construction"],
        "description": "Increase safety margins by 20% and reduce credibility during extreme weather"
    },
    {
        "override_id": "seismic_zone",
        "override_category": "environmental",
        "override_name": "Seismic Zone",
        "trigger_conditions": {"seismic_zone": True},
        "credibility_adjustment": 0.0,
        "required_min_tier": 2,
        "applicable_domains": ["structural"],
        "description": "Require Tier 2+ for structural calculations in seismic zones"
    },
    {
        "override_id": "critical_infrastructure",
        "override_category": "environmental",
        "override_name": "Critical Infrastructure",
        "trigger_conditions": {"infrastructure_type": "critical"},
        "credibility_adjustment": 0.0,
        "required_min_tier": 1,
        "applicable_domains": ["structural", "safety", "life_safety"],
        "description": "Enforce Tier 1 sources only for critical infrastructure"
    },
    {
        "override_id": "rush_conditions",
        "override_category": "temporal",
        "override_name": "Rush/Accelerated Schedule",
        "trigger_conditions": {"schedule": "rush"},
        "credibility_adjustment": -0.10,
        "applicable_domains": ["all"],
        "description": "Alert on accelerated decisions, reduce all source credibility by 10%"
    },
    {
        "override_id": "night_operations",
        "override_category": "temporal",
        "override_name": "Night Operations",
        "trigger_conditions": {"time_of_day": "night"},
        "credibility_adjustment": -0.05,
        "applicable_domains": ["inspection", "qa", "qc"],
        "description": "Reduce credibility by 5% for visual inspections at night"
    }
]
