"""
Adaptive Engine — Resilience & Recovery System for GART v3.0.

Provides circuit breaker patterns, exponential backoff, health monitoring,
and automatic failover for the dual-swarm system.

Components:
    - CircuitBreaker: Fail-fast pattern for unstable operations
    - ExponentialBackoff: Retry with exponential delay
    - HealthMonitor: Component health tracking
    - FailoverController: Automatic failover logic
    - AdaptiveEngine: Main orchestrator

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResilienceError(Exception):
    """Base exception for resilience operations."""


class CircuitOpenError(ResilienceError):
    """Raised when circuit breaker is open."""


class HealthCheckError(ResilienceError):
    """Raised when a health check fails."""


class FailoverError(ResilienceError):
    """Raised when failover cannot complete."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverStrategy(Enum):
    """Failover strategies."""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker.

    Attributes:
        failure_threshold: Failures before opening.
        recovery_timeout: Seconds before half-open.
        half_open_max_calls: Max calls in half-open.
        success_threshold: Successes to close.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff.

    Attributes:
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        multiplier: Delay multiplier per attempt.
        jitter: Random jitter factor.
        max_retries: Maximum retry attempts.
    """

    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.1
    max_retries: int = 5


@dataclass
class HealthSnapshot:
    """A point-in-time health snapshot.

    Attributes:
        component: Component name.
        status: Health status.
        latency_ms: Response latency.
        error_rate: Error rate (0.0-1.0).
        timestamp: Snapshot time.
        metadata: Additional data.
    """

    component: str
    status: HealthStatus
    latency_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Circuit breaker for protecting operations.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Failing fast, requests rejected immediately.
        HALF_OPEN: Testing if service recovered.

    Attributes:
        name: Circuit breaker name.
        config: Circuit configuration.
        state: Current circuit state.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitConfig] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._total_calls = 0
        self._total_failures = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    def can_execute(self) -> bool:
        """Check if execution is allowed.

        Returns:
            True if execution should proceed.
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit '%s' entering half-open", self.name)
                    return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return True

    def record_success(self) -> None:
        """Record a successful execution."""
        self._total_calls += 1

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info("Circuit '%s' closed", self.name)
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed execution."""
        self._total_calls += 1
        self._total_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit '%s' re-opened", self.name)
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit '%s' opened after %d failures",
                self.name, self._failure_count,
            )

    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation through the circuit breaker.

        Args:
            operation: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Operation result.

        Raises:
            CircuitOpenError: If circuit is open.
        """
        if not self.can_execute():
            raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = await operation(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_rate": (
                self._total_failures / max(self._total_calls, 1)
            ),
        }


# ---------------------------------------------------------------------------
# ExponentialBackoff
# ---------------------------------------------------------------------------


class ExponentialBackoff:
    """Exponential backoff retry mechanism.

    Retries failed operations with exponentially increasing delays.

    Attributes:
        config: Backoff configuration.
        attempt: Current attempt number.
    """

    def __init__(self, config: Optional[BackoffConfig] = None) -> None:
        self.config = config or BackoffConfig()
        self.attempt = 0

    def next_delay(self) -> float:
        """Calculate next retry delay.

        Returns:
            Delay in seconds.
        """
        delay = self.config.initial_delay * (
            self.config.multiplier ** self.attempt
        )
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter = delay * self.config.jitter * random.uniform(-1, 1)
        delay += jitter

        return max(0, delay)

    async def execute(
        self,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute with retry.

        Args:
            operation: Async function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Operation result.

        Raises:
            Exception: If all retries exhausted.
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            self.attempt = attempt
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.next_delay()
                    logger.warning(
                        "Attempt %d failed, retrying in %.2fs: %s",
                        attempt + 1, delay, e,
                    )
                    await asyncio.sleep(delay)

        raise last_error

    def reset(self) -> None:
        """Reset attempt counter."""
        self.attempt = 0


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Monitor health of system components.

    Tracks health status, latency, and error rates for
    registered components.
    """

    def __init__(
        self,
        unhealthy_threshold: float = 0.5,
        degraded_threshold: float = 0.2,
    ) -> None:
        self.unhealthy_threshold = unhealthy_threshold
        self.degraded_threshold = degraded_threshold
        self._snapshots: Dict[str, List[HealthSnapshot]] = {}
        self._window_size = 100

    def record(
        self,
        component: str,
        latency_ms: float,
        error: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a health observation.

        Args:
            component: Component name.
            latency_ms: Latency in milliseconds.
            error: Whether an error occurred.
            metadata: Additional metadata.
        """
        status = HealthStatus.HEALTHY
        if error:
            status = HealthStatus.UNHEALTHY
        elif latency_ms > 1000:
            status = HealthStatus.DEGRADED

        snapshot = HealthSnapshot(
            component=component,
            status=status,
            latency_ms=latency_ms,
            error_rate=1.0 if error else 0.0,
            metadata=metadata or {},
        )

        if component not in self._snapshots:
            self._snapshots[component] = []

        self._snapshots[component].append(snapshot)

        # Keep window size limited
        if len(self._snapshots[component]) > self._window_size:
            self._snapshots[component] = self._snapshots[component][-self._window_size:]

    def get_status(self, component: str) -> HealthStatus:
        """Get current health status of a component.

        Args:
            component: Component name.

        Returns:
            HealthStatus.
        """
        snapshots = self._snapshots.get(component, [])
        if not snapshots:
            return HealthStatus.UNKNOWN

        recent = snapshots[-20:]
        error_rate = sum(1 for s in recent if s.status == HealthStatus.UNHEALTHY) / len(recent)

        if error_rate >= self.unhealthy_threshold:
            return HealthStatus.UNHEALTHY
        elif error_rate >= self.degraded_threshold:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def get_snapshot(self, component: str) -> Optional[HealthSnapshot]:
        """Get latest snapshot for a component.

        Args:
            component: Component name.

        Returns:
            Latest snapshot or None.
        """
        snapshots = self._snapshots.get(component, [])
        return snapshots[-1] if snapshots else None

    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """Get health status of all components.

        Returns:
            Dict of component -> status.
        """
        return {
            component: self.get_status(component)
            for component in self._snapshots.keys()
        }

    def is_healthy(self, component: str) -> bool:
        """Check if a component is healthy.

        Args:
            component: Component name.

        Returns:
            True if healthy.
        """
        return self.get_status(component) == HealthStatus.HEALTHY

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "components": list(self._snapshots.keys()),
            "statuses": {c: s.value for c, s in self.get_all_statuses().items()},
            "total_snapshots": sum(len(s) for s in self._snapshots.values()),
        }


# ---------------------------------------------------------------------------
# FailoverController
# ---------------------------------------------------------------------------


class FailoverController:
    """Control automatic failover between components.

    Routes requests to healthy instances and handles
    failover when primary fails.
    """

    def __init__(self, health_monitor: HealthMonitor) -> None:
        self.health_monitor = health_monitor
        self._primaries: Dict[str, str] = {}
        self._secondaries: Dict[str, List[str]] = {}
        self._active: Dict[str, str] = {}

    def register_service(
        self,
        service_name: str,
        primary: str,
        secondaries: List[str],
    ) -> None:
        """Register a service with failover configuration.

        Args:
            service_name: Service identifier.
            primary: Primary instance.
            secondaries: Backup instances.
        """
        self._primaries[service_name] = primary
        self._secondaries[service_name] = secondaries
        self._active[service_name] = primary

    def get_active(self, service_name: str) -> Optional[str]:
        """Get the active instance for a service.

        Args:
            service_name: Service to query.

        Returns:
            Active instance identifier.
        """
        # Check if current active is healthy
        active = self._active.get(service_name)
        if active and self.health_monitor.is_healthy(active):
            return active

        # Try to failover
        return self._failover(service_name)

    def _failover(self, service_name: str) -> Optional[str]:
        """Perform failover for a service.

        Args:
            service_name: Service to failover.

        Returns:
            New active instance or None.
        """
        candidates = self._secondaries.get(service_name, [])

        for candidate in candidates:
            if self.health_monitor.is_healthy(candidate):
                old_active = self._active.get(service_name)
                self._active[service_name] = candidate
                logger.warning(
                    "Failover: %s -> %s (was %s)",
                    service_name, candidate, old_active,
                )
                return candidate

        # All candidates unhealthy, try primary as last resort
        primary = self._primaries.get(service_name)
        if primary:
            self._active[service_name] = primary
            return primary

        return None

    async def execute_with_failover(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute operation with failover.

        Args:
            service_name: Service to use.
            operation: Async operation.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Operation result.
        """
        instance = self.get_active(service_name)
        if not instance:
            raise FailoverError(f"No healthy instance for {service_name}")

        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            # Try failover
            new_instance = self._failover(service_name)
            if new_instance and new_instance != instance:
                return await operation(*args, **kwargs)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get failover statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "services": list(self._primaries.keys()),
            "active": dict(self._active),
            "primaries": dict(self._primaries),
        }


# ---------------------------------------------------------------------------
# AdaptiveEngine — Main orchestrator
# ---------------------------------------------------------------------------


class AdaptiveEngine:
    """Adaptive Resilience Engine for GART v3.0.

    Orchestrates circuit breakers, exponential backoff, health
    monitoring, and automatic failover.

    Attributes:
        health_monitor: Component health monitor.
        failover: Failover controller.
        circuits: Active circuit breakers.
        backoffs: Active backoff configurations.
    """

    def __init__(self) -> None:
        self.health_monitor = HealthMonitor()
        self.failover = FailoverController(self.health_monitor)
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._backoffs: Dict[str, ExponentialBackoff] = {}

    # --- Circuit breaker management ---

    def get_circuit(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name.

        Returns:
            CircuitBreaker instance.
        """
        if name not in self._circuits:
            self._circuits[name] = CircuitBreaker(name)
        return self._circuits[name]

    def remove_circuit(self, name: str) -> bool:
        """Remove a circuit breaker.

        Args:
            name: Circuit breaker name.

        Returns:
            True if removed.
        """
        return self._circuits.pop(name, None) is not None

    # --- Backoff management ---

    def get_backoff(self, name: str) -> ExponentialBackoff:
        """Get or create a backoff configuration.

        Args:
            name: Backoff configuration name.

        Returns:
            ExponentialBackoff instance.
        """
        if name not in self._backoffs:
            self._backoffs[name] = ExponentialBackoff()
        return self._backoffs[name]

    # --- Protected execution ---

    async def execute(
        self,
        operation: Callable,
        circuit_name: str = "default",
        backoff_name: str = "default",
        *args,
        **kwargs,
    ) -> Any:
        """Execute an operation with full protection.

        Applies circuit breaker, then exponential backoff.

        Args:
            operation: Async function to execute.
            circuit_name: Circuit breaker name.
            backoff_name: Backoff configuration name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Operation result.
        """
        circuit = self.get_circuit(circuit_name)
        backoff = self.get_backoff(backoff_name)

        async def _protected_operation(*a, **kw):
            return await backoff.execute(operation, *a, **kw)

        return await circuit.execute(_protected_operation, *args, **kwargs)

    # --- Health monitoring ---

    def record_health(
        self,
        component: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a health observation.

        Args:
            component: Component name.
            latency_ms: Latency in milliseconds.
            error: Whether an error occurred.
        """
        self.health_monitor.record(component, latency_ms, error)

    # --- Statistics ---

    def get_stats(self) -> Dict[str, Any]:
        """Get full engine statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "circuits": {
                name: cb.get_stats()
                for name, cb in self._circuits.items()
            },
            "health": self.health_monitor.get_stats(),
            "failover": self.failover.get_stats(),
            "backoff_configs": list(self._backoffs.keys()),
        }
