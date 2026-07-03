"""
Adaptive Resilience Engine — GART v3.0.

Moore's Law vs Murphy's Law: Exponential growth capability meets
Murphy's Law failure handling. Inspired by Andrew Lo's research on
financial system resilience.

Core principle: Better technology IS the solution (Version 2.0 philosophy).
Applies dynamic regulation inspired by CME SPAN-style portfolio risk management.

Components:
    - AdaptiveResilienceEngine: Main resilience controller
    - EntropyRegulator: Dynamic entropy regulation (0.2-0.7 optimal)
    - GracefulDegradation: Component-level degradation strategies
    - FeedbackLoop: Self-healing feedback loops
    - SystemState: Complete system state snapshot

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class FailureSeverity(Enum):
    """Severity levels for system failures."""

    LOW = "low"           # Minor issue, no user impact
    MEDIUM = "medium"     # Degraded performance
    HIGH = "high"         # Significant feature loss
    CRITICAL = "critical"  # System-threatening


class DegradedMode(Enum):
    """Available degraded operation modes."""

    FULL = "full"           # Normal operation
    REDUCED = "reduced"     # Reduced feature set
    ESSENTIAL = "essential" # Core functions only
    MINIMAL = "minimal"     # Survival mode
    OFFLINE = "offline"     # System offline


@dataclass
class FailureEvent:
    """A system failure event.

    Attributes:
        component: Which component failed.
        severity: Failure severity level.
        error_message: Human-readable error description.
        timestamp: When the failure occurred.
        context: Additional failure context.
    """

    component: str
    severity: FailureSeverity
    error_message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """A recovery action to take after a failure.

    Attributes:
        action_type: Type of recovery action.
        target_component: Component to recover.
        parameters: Recovery parameters.
        estimated_time_ms: Estimated recovery time.
    """

    action_type: str
    target_component: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_time_ms: float = 1000.0


@dataclass
class SystemState:
    """Complete snapshot of system state.

    Attributes:
        active_components: Currently active components.
        entropy_level: Current system entropy.
        load_factor: Current system load (0.0-1.0).
        error_count: Recent error count.
        degradation_level: Current degradation level.
        metrics: Arbitrary system metrics.
    """

    active_components: List[str] = field(default_factory=list)
    entropy_level: float = 0.5
    load_factor: float = 0.5
    error_count: int = 0
    degradation_level: DegradedMode = DegradedMode.FULL
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdaptationDecision:
    """Decision produced by the adaptation engine.

    Attributes:
        should_adapt: Whether adaptation is needed.
        adaptation_type: Type of adaptation.
        target_entropy: Target entropy level.
        degraded_components: Components to degrade.
        recovery_actions: Recovery actions to take.
        reason: Human-readable decision rationale.
    """

    should_adapt: bool
    adaptation_type: str
    target_entropy: float = 0.5
    degraded_components: List[str] = field(default_factory=list)
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    reason: str = ""


@dataclass
class AdaptationEvent:
    """Record of an adaptation event.

    Attributes:
        timestamp: When the adaptation occurred.
        decision: The adaptation decision.
        result: Result of the adaptation.
    """

    timestamp: float = field(default_factory=time.time)
    decision: Optional[AdaptationDecision] = None
    result: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EntropyRegulator
# ---------------------------------------------------------------------------


class EntropyRegulator:
    """Dynamic entropy regulator for the resilience engine.

    Maintains system entropy within the optimal range of 0.2-0.7.
    If entropy is too low, injects novel stimuli. If too high,
    stabilizes via anchor reinforcement.

    Attributes:
        target_entropy: Desired entropy level (default 0.45).
        current_entropy: Current measured entropy.
        min_optimal: Minimum optimal entropy.
        max_optimal: Maximum optimal entropy.
    """

    MIN_OPTIMAL: float = 0.2
    MAX_OPTIMAL: float = 0.7

    def __init__(self, target: float = 0.45) -> None:
        self.target_entropy = target
        self.current_entropy = target
        self._regulation_history: List[Tuple[float, float]] = []

    def measure(self, system_state: SystemState) -> float:
        """Measure current entropy from system state.

        Args:
            system_state: Current system state.

        Returns:
            Measured entropy value.
        """
        self.current_entropy = system_state.entropy_level
        return self.current_entropy

    def adjust(self, delta: float) -> float:
        """Adjust entropy by given delta.

        Args:
            delta: Amount to adjust (positive = increase).

        Returns:
            New entropy value.
        """
        self.current_entropy = max(0.0, min(1.0, self.current_entropy + delta))
        self._regulation_history.append((time.time(), self.current_entropy))
        return self.current_entropy

    def regulate(self, current: float) -> float:
        """Regulate entropy toward optimal range.

        Args:
            current: Current entropy value.

        Returns:
            Recommended entropy value.
        """
        if current < self.MIN_OPTIMAL:
            # Too low — inject entropy
            delta = (self.target_entropy - current) * 0.5
            logger.info(
                "Entropy too low (%.3f < %.2f), injecting %.4f",
                current, self.MIN_OPTIMAL, delta,
            )
            return min(self.target_entropy, current + delta)
        elif current > self.MAX_OPTIMAL:
            # Too high — stabilize
            delta = (current - self.target_entropy) * 0.3
            logger.info(
                "Entropy too high (%.3f > %.2f), stabilizing by %.4f",
                current, self.MAX_OPTIMAL, delta,
            )
            return max(self.target_entropy, current - delta)
        return current

    def get_status(self) -> Dict[str, Any]:
        """Get regulator status.

        Returns:
            Status dictionary.
        """
        return {
            "target": self.target_entropy,
            "current": self.current_entropy,
            "optimal_range": (self.MIN_OPTIMAL, self.MAX_OPTIMAL),
            "regulation_count": len(self._regulation_history),
            "needs_regulation": not (self.MIN_OPTIMAL <= self.current_entropy <= self.MAX_OPTIMAL),
        }


# ---------------------------------------------------------------------------
# GracefulDegradation
# ---------------------------------------------------------------------------


class GracefulDegradation:
    """Handles graceful degradation of system components.

    Reduces functionality while maintaining core operation,
    switches to backup implementations, and notifies the orchestrator.

    Degradation levels (ordered):
        FULL -> REDUCED -> ESSENTIAL -> MINIMAL -> OFFLINE
    """

    def __init__(self) -> None:
        self.degradation_levels: List[DegradedMode] = [
            DegradedMode.FULL,
            DegradedMode.REDUCED,
            DegradedMode.ESSENTIAL,
            DegradedMode.MINIMAL,
            DegradedMode.OFFLINE,
        ]
        self.current_level_idx: int = 0
        self._component_modes: Dict[str, DegradedMode] = {}

    @property
    def current_level(self) -> DegradedMode:
        """Current degradation level."""
        return self.degradation_levels[self.current_level_idx]

    def step_down(self) -> None:
        """Degrade one level down."""
        if self.current_level_idx < len(self.degradation_levels) - 1:
            self.current_level_idx += 1
            logger.warning(
                "System degraded: %s -> %s",
                self.degradation_levels[self.current_level_idx - 1].value,
                self.current_level.value,
            )

    def step_up(self) -> None:
        """Restore one level up."""
        if self.current_level_idx > 0:
            self.current_level_idx -= 1
            logger.info(
                "System restored: %s -> %s",
                self.degradation_levels[self.current_level_idx + 1].value,
                self.current_level.value,
            )

    def degrade(
        self,
        component: str,
        severity: FailureSeverity,
    ) -> DegradedMode:
        """Degrade a specific component based on failure severity.

        Args:
            component: Component name.
            severity: Failure severity.

        Returns:
            New degraded mode for the component.
        """
        severity_steps: Dict[FailureSeverity, int] = {
            FailureSeverity.LOW: 0,
            FailureSeverity.MEDIUM: 1,
            FailureSeverity.HIGH: 2,
            FailureSeverity.CRITICAL: 3,
        }

        steps = severity_steps.get(severity, 1)
        current_idx = self.degradation_levels.index(
            self._component_modes.get(component, DegradedMode.FULL)
        )
        new_idx = min(len(self.degradation_levels) - 1, current_idx + steps)
        new_mode = self.degradation_levels[new_idx]
        self._component_modes[component] = new_mode

        logger.info(
            "Component '%s' degraded to %s (severity: %s)",
            component, new_mode.value, severity.value,
        )
        return new_mode

    def get_component_mode(self, component: str) -> DegradedMode:
        """Get current degraded mode for a component.

        Args:
            component: Component name.

        Returns:
            Current DegradedMode.
        """
        return self._component_modes.get(component, DegradedMode.FULL)

    def get_current_capabilities(self) -> List[str]:
        """Get list of currently available capabilities.

        Returns:
            List of capability strings.
        """
        capabilities: Dict[DegradedMode, List[str]] = {
            DegradedMode.FULL: [
                "full_generation", "dual_swarm", "cross_attention",
                "master_mixing", "tournament", "memory_consolidation",
            ],
            DegradedMode.REDUCED: [
                "generation", "single_swarm", "basic_mixing",
                "tournament", "memory_read",
            ],
            DegradedMode.ESSENTIAL: [
                "basic_generation", "basic_mixing",
            ],
            DegradedMode.MINIMAL: [
                "text_generation_only",
            ],
            DegradedMode.OFFLINE: [],
        }
        return capabilities.get(self.current_level, [])

    def reset(self) -> None:
        """Reset degradation to full operation."""
        self.current_level_idx = 0
        self._component_modes.clear()


# ---------------------------------------------------------------------------
# FeedbackLoop
# ---------------------------------------------------------------------------


class FeedbackLoop:
    """Self-healing feedback loop for system resilience.

    Monitors a component and triggers recovery actions when
    failures are detected. Uses a simple PID-style controller
    for loop regulation.

    Attributes:
        loop_id: Unique loop identifier.
        component: Monitored component name.
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
    """

    def __init__(
        self,
        loop_id: str,
        component: str,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.01,
    ) -> None:
        self.loop_id = loop_id
        self.component = component
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._cycle_count: int = 0

    def cycle(self, setpoint: float, measured: float) -> float:
        """Execute one feedback loop cycle.

        Args:
            setpoint: Desired value.
            measured: Current measured value.

        Returns:
            Control output.
        """
        error = setpoint - measured
        self._integral += error
        derivative = error - self._last_error

        output = (
            self.kp * error +
            self.ki * self._integral +
            self.kd * derivative
        )

        self._last_error = error
        self._cycle_count += 1

        return output

    def tune_gains(self, kp: float, ki: float, kd: float) -> None:
        """Tune PID gains.

        Args:
            kp: New proportional gain.
            ki: New integral gain.
            kd: New derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        logger.info(
            "Feedback loop %s gains tuned: Kp=%.3f, Ki=%.3f, Kd=%.3f",
            self.loop_id, kp, ki, kd,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get loop statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "loop_id": self.loop_id,
            "component": self.component,
            "cycles": self._cycle_count,
            "last_error": self._last_error,
            "integral": self._integral,
        }


# ---------------------------------------------------------------------------
# AdaptiveResilienceEngine
# ---------------------------------------------------------------------------


class AdaptiveResilienceEngine:
    """Adaptive Resilience Engine: Moore's Law vs Murphy's Law.

    Inspired by Andrew Lo's research on financial system resilience.
    Exponential growth capability + Murphy's Law failure handling.

    Core Philosophy:
        - Moore's Law: System capability grows exponentially
        - Murphy's Law: Whatever can go wrong, will go wrong (faster with computers)
        - Resolution: Better technology IS the solution (Version 2.0 philosophy)
        - Regulation: CME SPAN-style portfolio risk management applied to AI systems

    Attributes:
        moore_growth_rate: Exponential capability multiplier.
        murphy_failure_prob: Base failure probability.
    """

    def __init__(self) -> None:
        self.moore_growth_rate = 1.0
        self.murphy_failure_prob = 0.05
        self.entropy_regulator = EntropyRegulator(target=0.45)
        self.feedback_loops: List[FeedbackLoop] = []
        self.degradation_controller = GracefulDegradation()
        self.adaptation_history: List[AdaptationEvent] = []
        self._failure_count: int = 0
        self._success_count: int = 0
        self._start_time: float = time.time()

    def adapt(self, system_state: SystemState) -> AdaptationDecision:
        """Main adaptation decision function.

        Applies Moore's Law growth, Murphy's Law failure prediction,
        and SPAN-style risk regulation to determine adaptation actions.

        Args:
            system_state: Current system state snapshot.

        Returns:
            AdaptationDecision with recommended actions.
        """
        decision = AdaptationDecision(should_adapt=False, adaptation_type="none")

        # Moore's Law: increase capability exponentially
        elapsed_hours = (time.time() - self._start_time) / 3600.0
        self.moore_growth_rate = 1.0 + elapsed_hours * 0.01

        # Murphy's Law: predict and handle failures
        predicted_failures = self.predict_failure_risk(system_state)

        # Check entropy regulation
        entropy_status = self.entropy_regulator.get_status()
        needs_entropy_reg = entropy_status["needs_regulation"]

        if needs_entropy_reg:
            new_entropy = self.entropy_regulator.regulate(
                system_state.entropy_level
            )
            decision.should_adapt = True
            decision.adaptation_type = "entropy_regulation"
            decision.target_entropy = new_entropy
            decision.reason = (
                f"Entropy {system_state.entropy_level:.3f} outside optimal range "
                f"({EntropyRegulator.MIN_OPTIMAL}-{EntropyRegulator.MAX_OPTIMAL})"
            )

        elif predicted_failures > 0.3:
            # High failure risk — proactive degradation
            decision.should_adapt = True
            decision.adaptation_type = "proactive_degradation"
            decision.reason = f"Predicted failure risk: {predicted_failures:.3f}"

            # Identify components to degrade
            if system_state.load_factor > 0.8:
                decision.degraded_components.append("non_essential_features")
                self.degradation_controller.step_down()

        elif system_state.error_count > 10:
            # Too many errors — enter recovery mode
            decision.should_adapt = True
            decision.adaptation_type = "recovery_mode"
            decision.reason = f"Error count {system_state.error_count} exceeds threshold"
            decision.recovery_actions.append(RecoveryAction(
                action_type="reset_feedbacks",
                target_component="all",
            ))

        # Moore's Law: if things are going well, grow capability
        if self._success_count > self._failure_count * 2:
            decision.should_adapt = True
            if decision.adaptation_type == "none":
                decision.adaptation_type = "capability_growth"
            decision.reason += f" | Moore growth rate: {self.moore_growth_rate:.4f}"

        # Record adaptation event
        self.adaptation_history.append(AdaptationEvent(
            decision=decision,
            result={"predicted_failures": predicted_failures},
        ))

        return decision

    def regulate_entropy(self, current: float, target: float) -> float:
        """Regulate system entropy to target level.

        Optimal range: 0.2-0.7
        - If < 0.2: inject entropy via novel stimuli
        - If > 0.7: stabilize via anchor reinforcement

        Args:
            current: Current entropy value.
            target: Target entropy value.

        Returns:
            Regulated entropy value.
        """
        self.entropy_regulator.target_entropy = target
        return self.entropy_regulator.regulate(current)

    def handle_failure(self, failure: FailureEvent) -> RecoveryAction:
        """Handle a failure event with self-healing.

        Args:
            failure: The failure event.

        Returns:
            RecoveryAction to take.
        """
        self._failure_count += 1

        logger.error(
            "Failure in '%s' [%s]: %s",
            failure.component, failure.severity.value, failure.error_message,
        )

        # Degrade affected component
        degraded_mode = self.degradation_controller.degrade(
            failure.component, failure.severity,
        )

        # Create recovery action
        action = RecoveryAction(
            action_type="self_heal",
            target_component=failure.component,
            parameters={
                "severity": failure.severity.value,
                "degraded_mode": degraded_mode.value,
                "failure_context": failure.context,
            },
        )

        # For critical failures, add circuit breaker
        if failure.severity == FailureSeverity.CRITICAL:
            action.parameters["circuit_breaker"] = True
            action.parameters["cooldown_seconds"] = 60.0
            self.degradation_controller.step_down()

        # Run feedback loops
        for loop in self.feedback_loops:
            if loop.component == failure.component:
                loop.cycle(setpoint=0.0, measured=1.0)

        return action

    def predict_failure_risk(self, system_state: SystemState) -> float:
        """Predict probability of upcoming failure.

        Murphy's Law: The probability increases with system complexity
        and load. Modeled as a function of load, error rate, and entropy.

        Args:
            system_state: Current system state.

        Returns:
            Predicted failure probability (0.0-1.0).
        """
        # Base Murphy probability
        base_prob = self.murphy_failure_prob

        # Load factor increases failure probability
        load_component = system_state.load_factor ** 2

        # Error rate increases probability
        total_ops = self._success_count + self._failure_count
        error_rate = self._failure_count / max(total_ops, 1)
        error_component = error_rate * 2.0

        # Entropy too high or low increases probability
        entropy_component = 0.0
        if system_state.entropy_level > 0.8 or system_state.entropy_level < 0.1:
            entropy_component = 0.2

        risk = min(1.0, base_prob + load_component * 0.3 + error_component * 0.3 + entropy_component)
        return risk

    def compute_moore_murphy_balance(self) -> float:
        """Compute the balance between Moore growth and Murphy risk.

        Returns:
            Balance score (>0 means Moore is winning, <0 means Murphy).
        """
        moore_score = math.log2(self.moore_growth_rate) if self.moore_growth_rate > 1 else -0.1
        murphy_score = -self.murphy_failure_prob * 10
        return moore_score + murphy_score

    def add_feedback_loop(self, component: str) -> FeedbackLoop:
        """Add a new feedback loop for a component.

        Args:
            component: Component to monitor.

        Returns:
            The created FeedbackLoop.
        """
        loop = FeedbackLoop(
            loop_id=f"loop_{len(self.feedback_loops)}",
            component=component,
        )
        self.feedback_loops.append(loop)
        return loop

    def report_status(self) -> Dict[str, Any]:
        """Generate comprehensive status report.

        Returns:
            Status dictionary.
        """
        return {
            "moore_growth_rate": self.moore_growth_rate,
            "murphy_failure_prob": self.murphy_failure_prob,
            "moore_murphy_balance": self.compute_moore_murphy_balance(),
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "uptime_hours": (time.time() - self._start_time) / 3600.0,
            "degradation_level": self.degradation_controller.current_level.value,
            "capabilities": self.degradation_controller.get_current_capabilities(),
            "entropy_regulator": self.entropy_regulator.get_status(),
            "feedback_loops": len(self.feedback_loops),
            "adaptation_count": len(self.adaptation_history),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Adaptive Resilience Engine loaded successfully.")

    engine = AdaptiveResilienceEngine()

    # Simulate system state
    state = SystemState(
        active_components=["lllm", "daw", "orchestrator"],
        entropy_level=0.85,  # Too high!
        load_factor=0.6,
        error_count=2,
    )

    print(f"\n--- Adaptation Decision ---")
    decision = engine.adapt(state)
    print(f"Should adapt: {decision.should_adapt}")
    print(f"Type: {decision.adaptation_type}")
    print(f"Reason: {decision.reason}")
    print(f"Target entropy: {decision.target_entropy:.3f}")

    # Test failure handling
    print(f"\n--- Failure Handling ---")
    failure = FailureEvent(
        component="lllm_encoder",
        severity=FailureSeverity.HIGH,
        error_message="Encoder output NaN detected",
    )
    recovery = engine.handle_failure(failure)
    print(f"Recovery: {recovery.action_type} on {recovery.target_component}")
    print(f"Status: {engine.report_status()}")
