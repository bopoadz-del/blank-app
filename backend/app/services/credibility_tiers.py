"""
Credibility Tier System for The Reasoner AI Platform.

Implements automatic tier assignment and promotion based on:
- Source credibility (ISO > Consultant > AI)
- Performance history
- Validation results
- Domain expert review
"""
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger


class CredibilityTier(str, Enum):
    """
    Credibility tiers determining formula trust level and autonomy.
    
    TIER_1: Highest trust - Auto-deploy without review
    TIER_2: High trust - Auto-deploy with notification
    TIER_3: Medium trust - Require human review
    TIER_4: Low trust - Require validation & review
    TIER_5: Unverified - Sandbox only, cannot deploy
    """
    TIER_1 = "tier_1_auto_deploy"
    TIER_2 = "tier_2_monitored_deploy"
    TIER_3 = "tier_3_review_required"
    TIER_4 = "tier_4_validation_required"
    TIER_5 = "tier_5_sandbox_only"


class SourceCredibility(str, Enum):
    """Source credibility levels."""
    ISO_STANDARD = "iso_standard"        # ISO, ASTM, ACI, AISC
    NATIONAL_CODE = "national_code"      # Building codes
    PEER_REVIEWED = "peer_reviewed"      # Academic papers
    CONSULTANT_VERIFIED = "consultant"   # Engineering consultant
    INTERNAL_VALIDATED = "internal"      # Company validated
    AI_DISCOVERED = "ai_discovered"      # ML-generated
    USER_SUBMITTED = "user_submitted"    # Unverified user input


class CredibilityManager:
    """
    Manages formula credibility tiers and automatic tier promotion/demotion.
    """
    
    def __init__(self):
        # Tier thresholds
        self.tier_thresholds = {
            CredibilityTier.TIER_1: {
                "min_confidence": 0.95,
                "min_executions": 100,
                "min_success_rate": 0.98,
                "allowed_sources": [
                    SourceCredibility.ISO_STANDARD,
                    SourceCredibility.NATIONAL_CODE
                ]
            },
            CredibilityTier.TIER_2: {
                "min_confidence": 0.90,
                "min_executions": 50,
                "min_success_rate": 0.95,
                "allowed_sources": [
                    SourceCredibility.ISO_STANDARD,
                    SourceCredibility.NATIONAL_CODE,
                    SourceCredibility.PEER_REVIEWED,
                    SourceCredibility.CONSULTANT_VERIFIED
                ]
            },
            CredibilityTier.TIER_3: {
                "min_confidence": 0.80,
                "min_executions": 25,
                "min_success_rate": 0.90,
                "allowed_sources": [
                    SourceCredibility.PEER_REVIEWED,
                    SourceCredibility.CONSULTANT_VERIFIED,
                    SourceCredibility.INTERNAL_VALIDATED
                ]
            },
            CredibilityTier.TIER_4: {
                "min_confidence": 0.70,
                "min_executions": 10,
                "min_success_rate": 0.85,
                "allowed_sources": [
                    SourceCredibility.INTERNAL_VALIDATED,
                    SourceCredibility.AI_DISCOVERED
                ]
            },
            CredibilityTier.TIER_5: {
                "min_confidence": 0.0,
                "min_executions": 0,
                "min_success_rate": 0.0,
                "allowed_sources": [
                    SourceCredibility.AI_DISCOVERED,
                    SourceCredibility.USER_SUBMITTED
                ]
            }
        }
    
    def determine_initial_tier(
        self,
        source: str,
        validation_passed: bool = False,
        peer_reviewed: bool = False
    ) -> CredibilityTier:
        """
        Determine initial tier when formula is first added.
        
        Args:
            source: Source of the formula
            validation_passed: Whether initial validation passed
            peer_reviewed: Whether peer reviewed
            
        Returns:
            Initial credibility tier
        """
        source_cred = self._classify_source(source)
        
        # ISO/National codes start at TIER_2 if validated
        if source_cred in [SourceCredibility.ISO_STANDARD, SourceCredibility.NATIONAL_CODE]:
            if validation_passed:
                return CredibilityTier.TIER_2
            else:
                return CredibilityTier.TIER_3
        
        # Peer-reviewed starts at TIER_3 if validated
        elif source_cred == SourceCredibility.PEER_REVIEWED:
            if validation_passed:
                return CredibilityTier.TIER_3
            else:
                return CredibilityTier.TIER_4
        
        # Consultant/Internal starts at TIER_4
        elif source_cred in [SourceCredibility.CONSULTANT_VERIFIED, SourceCredibility.INTERNAL_VALIDATED]:
            return CredibilityTier.TIER_4
        
        # AI/User starts at TIER_5 (sandbox)
        else:
            return CredibilityTier.TIER_5
    
    def evaluate_tier_promotion(
        self,
        current_tier: CredibilityTier,
        confidence_score: float,
        total_executions: int,
        successful_executions: int,
        validation_stages_passed: List[str],
        source: str,
        peer_review_status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if formula should be promoted to higher tier.
        
        Returns:
            {
                "promote": bool,
                "new_tier": CredibilityTier,
                "reason": str,
                "auto_deploy": bool
            }
        """
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        source_cred = self._classify_source(source)
        
        # Check each tier from highest to lowest
        for tier in [CredibilityTier.TIER_1, CredibilityTier.TIER_2, 
                     CredibilityTier.TIER_3, CredibilityTier.TIER_4]:
            
            # Skip if already at or above this tier
            if self._tier_level(current_tier) <= self._tier_level(tier):
                continue
            
            thresholds = self.tier_thresholds[tier]
            
            # Check all criteria
            meets_confidence = confidence_score >= thresholds["min_confidence"]
            meets_executions = total_executions >= thresholds["min_executions"]
            meets_success_rate = success_rate >= thresholds["min_success_rate"]
            meets_source = source_cred in thresholds["allowed_sources"]
            meets_validation = len(validation_stages_passed) >= 4  # At least 4/5 stages
            
            if meets_confidence and meets_executions and meets_success_rate and meets_source and meets_validation:
                reason = f"Promoted to {tier.value}: confidence={confidence_score:.2f}, " \
                         f"executions={total_executions}, success_rate={success_rate:.2f}"
                
                return {
                    "promote": True,
                    "new_tier": tier,
                    "reason": reason,
                    "auto_deploy": tier in [CredibilityTier.TIER_1, CredibilityTier.TIER_2],
                    "requires_review": tier == CredibilityTier.TIER_3
                }
        
        # No promotion
        return {
            "promote": False,
            "new_tier": current_tier,
            "reason": "Criteria not met for promotion",
            "auto_deploy": False
        }
    
    def evaluate_tier_demotion(
        self,
        current_tier: CredibilityTier,
        confidence_score: float,
        total_executions: int,
        successful_executions: int,
        recent_failures: int,
        validation_failures: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if formula should be demoted to lower tier.
        
        Args:
            recent_failures: Number of failures in last N executions
            validation_failures: Recent validation stage failures
            
        Returns:
            {
                "demote": bool,
                "new_tier": CredibilityTier,
                "reason": str,
                "suspend": bool
            }
        """
        success_rate = successful_executions / total_executions if total_executions > 0 else 1.0
        
        # Critical failure conditions
        if recent_failures >= 5:
            return {
                "demote": True,
                "new_tier": CredibilityTier.TIER_5,
                "reason": f"Suspended due to {recent_failures} recent failures",
                "suspend": True,
                "requires_investigation": True
            }
        
        # Check if current tier requirements still met
        current_thresholds = self.tier_thresholds[current_tier]
        
        fails_confidence = confidence_score < current_thresholds["min_confidence"] - 0.05
        fails_success_rate = success_rate < current_thresholds["min_success_rate"] - 0.03
        has_validation_failures = len(validation_failures) >= 2
        
        if fails_confidence or fails_success_rate or has_validation_failures:
            # Demote one tier
            new_tier = self._demote_one_tier(current_tier)
            
            reasons = []
            if fails_confidence:
                reasons.append(f"confidence dropped to {confidence_score:.2f}")
            if fails_success_rate:
                reasons.append(f"success rate dropped to {success_rate:.2f}")
            if has_validation_failures:
                reasons.append(f"{len(validation_failures)} validation failures")
            
            return {
                "demote": True,
                "new_tier": new_tier,
                "reason": f"Demoted: {', '.join(reasons)}",
                "suspend": False,
                "requires_review": True
            }
        
        # No demotion
        return {
            "demote": False,
            "new_tier": current_tier,
            "reason": "Performance within tier requirements",
            "suspend": False
        }
    
    def get_deployment_policy(self, tier: CredibilityTier) -> Dict[str, Any]:
        """
        Get deployment policy for a given tier.
        
        Returns:
            {
                "auto_deploy": bool,
                "requires_review": bool,
                "requires_validation": bool,
                "notification_required": bool,
                "sandbox_only": bool
            }
        """
        policies = {
            CredibilityTier.TIER_1: {
                "auto_deploy": True,
                "requires_review": False,
                "requires_validation": False,
                "notification_required": False,
                "sandbox_only": False,
                "max_concurrent_deployments": None
            },
            CredibilityTier.TIER_2: {
                "auto_deploy": True,
                "requires_review": False,
                "requires_validation": False,
                "notification_required": True,
                "sandbox_only": False,
                "max_concurrent_deployments": 100
            },
            CredibilityTier.TIER_3: {
                "auto_deploy": False,
                "requires_review": True,
                "requires_validation": False,
                "notification_required": True,
                "sandbox_only": False,
                "max_concurrent_deployments": 10
            },
            CredibilityTier.TIER_4: {
                "auto_deploy": False,
                "requires_review": True,
                "requires_validation": True,
                "notification_required": True,
                "sandbox_only": False,
                "max_concurrent_deployments": 5
            },
            CredibilityTier.TIER_5: {
                "auto_deploy": False,
                "requires_review": True,
                "requires_validation": True,
                "notification_required": True,
                "sandbox_only": True,
                "max_concurrent_deployments": 1
            }
        }
        
        return policies[tier]
    
    def _classify_source(self, source: str) -> SourceCredibility:
        """Classify source credibility."""
        source_lower = source.lower()
        
        # ISO/ASTM/ACI/AISC standards
        if any(std in source_lower for std in ['iso', 'astm', 'aci', 'aisc', 'en199']):
            return SourceCredibility.ISO_STANDARD
        
        # National codes
        elif any(code in source_lower for code in ['building code', 'national code', 'saso', 'uae code']):
            return SourceCredibility.NATIONAL_CODE
        
        # Peer-reviewed
        elif any(journal in source_lower for journal in ['journal', 'peer-reviewed', 'conference', 'ieee']):
            return SourceCredibility.PEER_REVIEWED
        
        # Consultant
        elif 'consultant' in source_lower:
            return SourceCredibility.CONSULTANT_VERIFIED
        
        # Internal
        elif 'internal' in source_lower or 'validated' in source_lower:
            return SourceCredibility.INTERNAL_VALIDATED
        
        # AI
        elif 'ai' in source_lower or 'ml' in source_lower or 'discovered' in source_lower:
            return SourceCredibility.AI_DISCOVERED
        
        # User submitted
        else:
            return SourceCredibility.USER_SUBMITTED
    
    def _tier_level(self, tier: CredibilityTier) -> int:
        """Get numeric level for tier (1=highest, 5=lowest)."""
        levels = {
            CredibilityTier.TIER_1: 1,
            CredibilityTier.TIER_2: 2,
            CredibilityTier.TIER_3: 3,
            CredibilityTier.TIER_4: 4,
            CredibilityTier.TIER_5: 5
        }
        return levels[tier]
    
    def _demote_one_tier(self, current_tier: CredibilityTier) -> CredibilityTier:
        """Demote formula by one tier."""
        demotion_map = {
            CredibilityTier.TIER_1: CredibilityTier.TIER_2,
            CredibilityTier.TIER_2: CredibilityTier.TIER_3,
            CredibilityTier.TIER_3: CredibilityTier.TIER_4,
            CredibilityTier.TIER_4: CredibilityTier.TIER_5,
            CredibilityTier.TIER_5: CredibilityTier.TIER_5  # Can't go lower
        }
        return demotion_map[current_tier]


# Global instance
credibility_manager = CredibilityManager()


# Example usage
if __name__ == "__main__":
    manager = CredibilityManager()
    
    # Example 1: New ISO standard formula
    tier = manager.determine_initial_tier(
        source="ISO_10243",
        validation_passed=True
    )
    print(f"ISO formula initial tier: {tier}")
    
    # Example 2: Check for promotion
    promotion = manager.evaluate_tier_promotion(
        current_tier=CredibilityTier.TIER_3,
        confidence_score=0.92,
        total_executions=75,
        successful_executions=72,
        validation_stages_passed=["syntactic", "dimensional", "physical", "empirical"],
        source="ISO_10243"
    )
    print(f"Promotion evaluation: {promotion}")
    
    # Example 3: Get deployment policy
    policy = manager.get_deployment_policy(CredibilityTier.TIER_2)
    print(f"TIER_2 policy: {policy}")
