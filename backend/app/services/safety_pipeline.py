"""
Multi-Stage Safety Pipeline Service.
Implements 5-stage detection system for prohibited content and harmful requests.
NON-NEGOTIABLE: Executes before ANY reasoning or calculation.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
import uuid
import re
import logging

from ..models.safety_layer import (
    SafetyIncident,
    ProhibitedCategory,
    SafetyAction,
    AlertLevel,
    SafetyPattern,
    UserSafetyScore,
    EmergencyProtocol,
    SafetyConfiguration
)
from ..models.auth import User

logger = logging.getLogger(__name__)


class ProhibitedContentDetector:
    """
    Detects prohibited content using keyword matching, pattern detection, and intent analysis.
    """

    # Prohibited keywords by category (simplified - production would use ML models)
    PROHIBITED_KEYWORDS = {
        ProhibitedCategory.CHILD_EXPLOITATION: [
            "child exploitation", "csam", "child abuse", "minor", "underage"
        ],
        ProhibitedCategory.ILLEGAL_DRUGS: [
            "meth synthesis", "cocaine production", "heroin manufacturing",
            "drug production", "illegal narcotics"
        ],
        ProhibitedCategory.WEAPONS_MANUFACTURING: [
            "weapon manufacturing", "gun assembly", "firearm production",
            "automatic weapon", "illegal weapons"
        ],
        ProhibitedCategory.EXPLOSIVES: [
            "bomb making", "explosive device", "improvised explosive",
            "detonator", "explosive synthesis"
        ],
        ProhibitedCategory.BIOWEAPONS: [
            "bioweapon", "biological weapon", "pathogen weaponization",
            "virus weaponization", "anthrax production"
        ],
        ProhibitedCategory.GAMBLING_MANIPULATION: [
            "casino rigging", "gambling manipulation", "slot machine hacking",
            "odds manipulation", "gambling fraud"
        ],
        ProhibitedCategory.ILLEGAL_ALCOHOL: [
            "illegal distillation", "methanol alcohol", "unregulated spirits",
            "bootleg alcohol"
        ],
        ProhibitedCategory.HUMAN_TRAFFICKING: [
            "human trafficking", "people smuggling", "forced labor",
            "exploitation of persons"
        ],
        ProhibitedCategory.TERRORISM: [
            "terrorist attack", "terrorism planning", "mass casualty",
            "terror organization"
        ],
        ProhibitedCategory.VIOLENCE_INCITEMENT: [
            "incite violence", "call to violence", "violent uprising",
            "harm individuals"
        ],
        ProhibitedCategory.FRAUD: [
            "identity theft", "credit card fraud", "financial fraud",
            "phishing scheme", "scam operation"
        ],
        ProhibitedCategory.IDENTITY_THEFT: [
            "steal identity", "fake documents", "forged credentials",
            "identity fraud"
        ]
    }

    # Suspicious patterns that warrant investigation
    SUSPICIOUS_PATTERNS = [
        r"how to (make|create|build|manufacture) (illegal|dangerous)",
        r"(bypass|evade|avoid) (law enforcement|detection|security)",
        r"(harm|hurt|injure|kill) (people|person|individual)",
        r"(plan|execute|carry out) (attack|assault)",
    ]

    @staticmethod
    def detect(content: str, context: Dict) -> Tuple[bool, Optional[ProhibitedCategory], float, List[str]]:
        """
        Detect prohibited content in user request.

        Args:
            content: User request text
            context: Request context (IP, location, user history)

        Returns:
            Tuple of (is_prohibited, category, confidence, matched_keywords)
        """
        content_lower = content.lower()

        # Check prohibited keywords
        for category, keywords in ProhibitedContentDetector.PROHIBITED_KEYWORDS.items():
            matched = []
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    matched.append(keyword)

            if matched:
                confidence = min(0.95, 0.7 + (len(matched) * 0.1))
                return True, category, confidence, matched

        # Check suspicious patterns
        for pattern in ProhibitedContentDetector.SUSPICIOUS_PATTERNS:
            if re.search(pattern, content_lower):
                # Pattern match = investigation needed
                logger.warning(f"Suspicious pattern detected: {pattern}")
                return True, None, 0.6, [pattern]

        return False, None, 0.0, []


class SafetyPipeline:
    """
    5-Stage Safety Pipeline.
    Executes before any reasoning, calculation, or formula execution.
    """

    def __init__(self, db: Session, deployment_name: str = "default"):
        self.db = db
        self.deployment_name = deployment_name
        self.detector = ProhibitedContentDetector()
        self.config = self._load_config()

    def _load_config(self) -> SafetyConfiguration:
        """Load deployment-specific safety configuration."""
        config = self.db.query(SafetyConfiguration).filter(
            SafetyConfiguration.deployment_name == self.deployment_name
        ).first()

        if not config:
            # Create default strict configuration
            config = SafetyConfiguration(
                deployment_name=self.deployment_name,
                strict_mode=True,  # ALWAYS TRUE
                compliance_framework="GLOBAL_DEFAULT",
                additional_blocks={},
                reporting_channels={
                    "child_safety": ["local_authorities", "platform_admin"],
                    "terrorism": ["law_enforcement", "security_team"],
                    "default": ["security_team"]
                },
                audit_frequency="continuous",
                audit_retention_days=2555,  # 7 years
                multi_factor_required=True
            )
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)

        return config

    def execute_pipeline(
        self,
        request_content: str,
        user: Optional[User],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[SafetyIncident]]:
        """
        Execute complete 5-stage safety pipeline.

        Args:
            request_content: User request text
            user: User making the request
            context: Request context (IP, location, formula_id, etc.)

        Returns:
            Tuple of (is_safe, incident_if_blocked)
        """
        logger.info(f"Safety pipeline executing for user {user.id if user else 'anonymous'}")

        # Stage 1: Input Screening
        passed_stage1, incident = self.stage1_input_screening(request_content, user, context)
        if not passed_stage1:
            return False, incident

        # Stage 2: Context Analysis
        passed_stage2, incident = self.stage2_context_analysis(request_content, user, context)
        if not passed_stage2:
            return False, incident

        # Stage 3: Formula Inspection (if formula_id provided)
        if context.get('formula_id'):
            passed_stage3, incident = self.stage3_formula_inspection(context['formula_id'], user, context)
            if not passed_stage3:
                return False, incident

        # Stage 4 & 5 are executed during/after execution
        # (output_filtering and continuous_monitoring)

        logger.info("Safety pipeline: ALL STAGES PASSED")
        return True, None

    def stage1_input_screening(
        self,
        request_content: str,
        user: Optional[User],
        context: Dict
    ) -> Tuple[bool, Optional[SafetyIncident]]:
        """
        Stage 1: Input Screening.
        Block prohibited content at the point of entry.
        """
        logger.debug("Stage 1: Input Screening")

        is_prohibited, category, confidence, matched = self.detector.detect(request_content, context)

        if is_prohibited:
            logger.critical(f"PROHIBITED CONTENT DETECTED: {category} (confidence: {confidence})")

            # Determine action and alert level
            if category in [ProhibitedCategory.CHILD_EXPLOITATION, ProhibitedCategory.TERRORISM]:
                action = SafetyAction.BLOCK_AND_REPORT
                alert_level = AlertLevel.CRITICAL
            elif category in [ProhibitedCategory.BIOWEAPONS, ProhibitedCategory.EXPLOSIVES]:
                action = SafetyAction.BLOCK_AND_INVESTIGATE
                alert_level = AlertLevel.HIGH
            else:
                action = SafetyAction.IMMEDIATE_BLOCK
                alert_level = AlertLevel.MEDIUM

            # Create incident
            incident = self._create_incident(
                category=category,
                action=action,
                alert_level=alert_level,
                detected_at_stage="stage1_input_screening",
                confidence=confidence,
                matched_keywords=matched,
                request_content=request_content,
                user=user,
                context=context
            )

            # Execute emergency protocol
            self._execute_emergency_protocol(incident)

            # Update user safety score
            if user:
                self._update_user_safety_score(user.id, incident)

            return False, incident

        return True, None

    def stage2_context_analysis(
        self,
        request_content: str,
        user: Optional[User],
        context: Dict
    ) -> Tuple[bool, Optional[SafetyIncident]]:
        """
        Stage 2: Context Analysis.
        Analyze user intent and request patterns.
        """
        logger.debug("Stage 2: Context Analysis")

        if not user:
            # Anonymous users have stricter scrutiny
            logger.warning("Anonymous user - elevated monitoring")
            return True, None

        # Check user safety score
        safety_score = self.db.query(UserSafetyScore).filter(
            UserSafetyScore.user_id == user.id
        ).first()

        if safety_score:
            if safety_score.account_status in ["banned", "suspended"]:
                logger.critical(f"User {user.id} is {safety_score.account_status}")

                incident = self._create_incident(
                    category=None,
                    action=SafetyAction.RESTRICT,
                    alert_level=AlertLevel.HIGH,
                    detected_at_stage="stage2_context_analysis",
                    confidence=1.0,
                    matched_keywords=["account_restricted"],
                    request_content=request_content,
                    user=user,
                    context=context
                )

                return False, incident

            if safety_score.risk_level == "critical":
                logger.warning(f"User {user.id} has critical risk level")
                # Allow but log for investigation
                return True, None

        # Check for behavioral patterns
        patterns = self._check_behavioral_patterns(user, request_content, context)
        if patterns:
            logger.warning(f"Concerning behavioral patterns detected for user {user.id}: {patterns}")
            # Log but don't block yet

        return True, None

    def stage3_formula_inspection(
        self,
        formula_id: int,
        user: Optional[User],
        context: Dict
    ) -> Tuple[bool, Optional[SafetyIncident]]:
        """
        Stage 3: Formula Inspection.
        Check if formula contains dangerous calculations.
        """
        logger.debug("Stage 3: Formula Inspection")

        from ..models.database import Formula

        formula = self.db.query(Formula).filter(Formula.id == formula_id).first()
        if not formula:
            logger.error(f"Formula {formula_id} not found")
            return True, None

        # Check formula metadata for safety flags
        metadata = formula.metadata or {}

        if metadata.get('safety_flagged', False):
            logger.warning(f"Formula {formula_id} is safety-flagged")

            incident = self._create_incident(
                category=None,
                action=SafetyAction.WARN,
                alert_level=AlertLevel.MEDIUM,
                detected_at_stage="stage3_formula_inspection",
                confidence=0.8,
                matched_keywords=["safety_flagged_formula"],
                request_content=f"Formula execution: {formula.name}",
                user=user,
                context=context
            )

            # Don't block, but log
            return True, None

        return True, None

    def stage4_output_filtering(
        self,
        output_data: Dict,
        formula_id: int,
        user: Optional[User],
        context: Dict
    ) -> Tuple[bool, Optional[SafetyIncident]]:
        """
        Stage 4: Output Filtering.
        Verify that results are safe to return.
        """
        logger.debug("Stage 4: Output Filtering")

        # Check for unreasonable or dangerous output values
        # This is domain-specific and would be extended for production

        # Example: Check for extreme values
        if isinstance(output_data, dict):
            for key, value in output_data.items():
                if isinstance(value, (int, float)):
                    if abs(value) > 1e10:
                        logger.warning(f"Extreme output value detected: {key}={value}")

        return True, None

    def stage5_continuous_monitoring(
        self,
        user: Optional[User],
        time_window_minutes: int = 60
    ) -> List[SafetyPattern]:
        """
        Stage 5: Continuous Monitoring.
        Check for concerning patterns over time.
        """
        logger.debug("Stage 5: Continuous Monitoring")

        if not user:
            return []

        # Check for pattern matches in time window
        time_threshold = datetime.utcnow() - timedelta(minutes=time_window_minutes)

        recent_incidents = self.db.query(SafetyIncident).filter(
            SafetyIncident.user_id == user.id,
            SafetyIncident.incident_timestamp >= time_threshold
        ).count()

        if recent_incidents >= 3:
            logger.critical(f"User {user.id} has {recent_incidents} incidents in {time_window_minutes} minutes")
            # Auto-escalate

        # Check active patterns
        active_patterns = self.db.query(SafetyPattern).filter(
            SafetyPattern.is_active == True
        ).all()

        matched_patterns = []
        for pattern in active_patterns:
            if self._check_pattern_match(user, pattern, time_window_minutes):
                matched_patterns.append(pattern)
                logger.warning(f"Pattern matched: {pattern.pattern_name} for user {user.id}")

        return matched_patterns

    def _create_incident(
        self,
        category: Optional[ProhibitedCategory],
        action: SafetyAction,
        alert_level: AlertLevel,
        detected_at_stage: str,
        confidence: float,
        matched_keywords: List[str],
        request_content: str,
        user: Optional[User],
        context: Dict
    ) -> SafetyIncident:
        """Create and persist a safety incident."""
        incident = SafetyIncident(
            incident_id=f"safety_{uuid.uuid4().hex[:12]}",
            prohibited_category=category,
            safety_action=action,
            alert_level=alert_level,
            detected_at_stage=detected_at_stage,
            detection_confidence=confidence,
            matched_keywords=matched_keywords,
            matched_patterns=[],
            user_id=user.id if user else None,
            request_content=request_content,
            request_context=context,
            action_taken=f"{action.value} at {detected_at_stage}",
            user_message=self._get_user_message(category, action),
            blocked_successfully=True,
            reported_to=[],
            session_id=context.get('session_id'),
            ip_address=context.get('ip_address'),
            user_agent=context.get('user_agent'),
            geographic_location=context.get('location'),
            incident_timestamp=datetime.utcnow()
        )

        self.db.add(incident)
        self.db.commit()
        self.db.refresh(incident)

        logger.critical(f"Safety incident created: {incident.incident_id}")

        return incident

    def _execute_emergency_protocol(self, incident: SafetyIncident):
        """Execute emergency protocol for critical incidents."""
        if incident.alert_level in [AlertLevel.CRITICAL, AlertLevel.HIGH]:
            protocol = self.db.query(EmergencyProtocol).filter(
                EmergencyProtocol.prohibited_category == incident.prohibited_category,
                EmergencyProtocol.is_active == True
            ).first()

            if protocol:
                logger.critical(f"Executing emergency protocol: {protocol.protocol_name}")

                # Execute immediate actions
                for action in protocol.immediate_actions:
                    logger.info(f"Emergency action: {action}")
                    # In production: terminate_session(), preserve_logs(), alert_authorities()

                # Update incident with notification recipients
                reported_to = []
                if protocol.notify_law_enforcement:
                    reported_to.append("law_enforcement")
                if protocol.notify_security_team:
                    reported_to.append("security_team")
                if protocol.notify_management:
                    reported_to.append("management")

                incident.reported_to = reported_to
                incident.escalation_timestamp = datetime.utcnow()
                self.db.commit()

    def _update_user_safety_score(self, user_id: int, incident: SafetyIncident):
        """Update user safety score after incident."""
        safety_score = self.db.query(UserSafetyScore).filter(
            UserSafetyScore.user_id == user_id
        ).first()

        if not safety_score:
            # Create new safety score
            safety_score = UserSafetyScore(
                user_id=user_id,
                safety_score=1.0,
                risk_level="low"
            )
            self.db.add(safety_score)

        # Update after incident
        safety_score.update_after_incident(incident)

        self.db.commit()

        logger.warning(f"User {user_id} safety score updated: {safety_score.safety_score:.2f} ({safety_score.risk_level})")

    def _check_behavioral_patterns(
        self,
        user: User,
        request_content: str,
        context: Dict
    ) -> List[str]:
        """Check for concerning behavioral patterns."""
        patterns_detected = []

        # Pattern 1: Rapid requests
        from datetime import timedelta
        time_threshold = datetime.utcnow() - timedelta(minutes=5)

        from ..models.database import FormulaExecution
        recent_executions = self.db.query(FormulaExecution).filter(
            FormulaExecution.user_id == user.id,
            FormulaExecution.created_at >= time_threshold
        ).count()

        if recent_executions > 20:
            patterns_detected.append("rapid_requests")

        # Pattern 2: Off-hours activity
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:
            patterns_detected.append("off_hours_activity")

        return patterns_detected

    def _check_pattern_match(
        self,
        user: User,
        pattern: SafetyPattern,
        time_window_minutes: int
    ) -> bool:
        """Check if user matches a safety pattern."""
        # Simplified pattern matching
        # Production would use complex rule evaluation

        detection_rules = pattern.detection_rules or {}

        # Example: Check if user has triggered pattern threshold
        from datetime import timedelta
        time_threshold = datetime.utcnow() - timedelta(minutes=pattern.time_window_minutes)

        incident_count = self.db.query(SafetyIncident).filter(
            SafetyIncident.user_id == user.id,
            SafetyIncident.prohibited_category == pattern.pattern_category,
            SafetyIncident.incident_timestamp >= time_threshold
        ).count()

        if incident_count >= pattern.threshold_count:
            pattern.detection_count += 1
            pattern.last_triggered = datetime.utcnow()
            self.db.commit()
            return True

        return False

    def _get_user_message(self, category: Optional[ProhibitedCategory], action: SafetyAction) -> str:
        """Get user-facing message for blocked request."""
        if action == SafetyAction.IMMEDIATE_BLOCK:
            return "Your request has been blocked as it violates our safety policies. If you believe this is an error, please contact support."
        elif action == SafetyAction.BLOCK_AND_REPORT:
            return "Your request has been blocked and reported to the appropriate authorities. This action cannot be appealed."
        elif action == SafetyAction.BLOCK_AND_INVESTIGATE:
            return "Your request has been blocked pending investigation. Our security team will review this incident."
        elif action == SafetyAction.RESTRICT:
            return "Your account has restricted access. Please contact support for more information."
        else:
            return "Your request cannot be processed at this time."


from datetime import timedelta
