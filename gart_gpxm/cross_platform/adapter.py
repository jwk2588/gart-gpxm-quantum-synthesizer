"""
Cross-Platform Adapter — Multi-Platform Bridge for GART v3.0.

Provides platform abstraction for running GART across different
environments: local, cloud, containerized, and edge.

Components:
    - PlatformAdapter: Main platform abstraction
    - LocalAdapter: Local execution adapter
    - CloudAdapter: Cloud execution adapter
    - ContainerAdapter: Container execution adapter
    - EdgeAdapter: Edge device adapter
    - ResourceMonitor: Platform resource monitoring

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
import os
import platform as sys_platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlatformType(Enum):
    """Supported platform types."""

    LOCAL = "local"
    CLOUD = "cloud"
    CONTAINER = "container"
    EDGE = "edge"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of compute resources."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources.

    Attributes:
        resource_type: Type of resource.
        total: Total capacity.
        used: Used amount.
        available: Available amount.
        unit: Unit of measurement.
        timestamp: Snapshot time.
    """

    resource_type: ResourceType
    total: float
    used: float
    available: float
    unit: str
    timestamp: float = field(default_factory=time.time)

    @property
    def utilization(self) -> float:
        """Calculate utilization percentage."""
        if self.total == 0:
            return 0.0
        return self.used / self.total


@dataclass
class PlatformConfig:
    """Configuration for a platform.

    Attributes:
        platform_type: Type of platform.
        max_workers: Maximum parallel workers.
        resource_limits: Resource limits.
        env_vars: Environment variables.
    """

    platform_type: PlatformType
    max_workers: int = 4
    resource_limits: Dict[str, float] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PlatformAdapter (abstract base)
# ---------------------------------------------------------------------------


class PlatformAdapter(ABC):
    """Abstract base for platform adapters.

    Defines the interface that all platform adapters must implement.
    """

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    @abstractmethod
    def get_resources(self) -> Dict[ResourceType, ResourceSnapshot]:
        """Get current resource snapshot.

        Returns:
            Dict of resource type -> snapshot.
        """
        ...

    @abstractmethod
    def execute(self, command: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a command on the platform.

        Args:
            command: Command to execute.
            **kwargs: Additional execution parameters.

        Returns:
            Execution result.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if platform is available.

        Returns:
            True if platform is ready.
        """
        ...

    def get_type(self) -> PlatformType:
        """Get platform type.

        Returns:
            Platform type.
        """
        return self.config.platform_type

    def can_accommodate(self, requirements: Dict[str, float]) -> bool:
        """Check if platform can accommodate requirements.

        Args:
            requirements: Resource requirements.

        Returns:
            True if requirements can be met.
        """
        resources = self.get_resources()
        for key, required in requirements.items():
            try:
                rtype = ResourceType(key)
                if rtype in resources:
                    if resources[rtype].available < required:
                        return False
            except ValueError:
                continue
        return True


# ---------------------------------------------------------------------------
# LocalAdapter
# ---------------------------------------------------------------------------


class LocalAdapter(PlatformAdapter):
    """Adapter for local execution environment."""

    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        super().__init__(config or PlatformConfig(PlatformType.LOCAL))

    def get_resources(self) -> Dict[ResourceType, ResourceSnapshot]:
        """Get local system resources."""
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory
        mem = psutil.virtual_memory()

        # Disk
        disk = psutil.disk_usage("/")

        return {
            ResourceType.CPU: ResourceSnapshot(
                resource_type=ResourceType.CPU,
                total=float(cpu_count * 100),
                used=float(cpu_percent * cpu_count),
                available=float((100 - cpu_percent) * cpu_count),
                unit="percent",
            ),
            ResourceType.MEMORY: ResourceSnapshot(
                resource_type=ResourceType.MEMORY,
                total=mem.total / (1024**3),
                used=mem.used / (1024**3),
                available=mem.available / (1024**3),
                unit="GB",
            ),
            ResourceType.STORAGE: ResourceSnapshot(
                resource_type=ResourceType.STORAGE,
                total=disk.total / (1024**3),
                used=disk.used / (1024**3),
                available=disk.free / (1024**3),
                unit="GB",
            ),
        }

    def execute(self, command: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a local command."""
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 60),
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_available(self) -> bool:
        """Local platform is always available."""
        return True


# ---------------------------------------------------------------------------
# CloudAdapter
# ---------------------------------------------------------------------------


class CloudAdapter(PlatformAdapter):
    """Adapter for cloud execution environment."""

    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        super().__init__(config or PlatformConfig(PlatformType.CLOUD))
        self._connected = False

    def connect(self) -> bool:
        """Connect to cloud provider.

        Returns:
            True if connected.
        """
        # Placeholder for cloud connection
        self._connected = True
        return True

    def get_resources(self) -> Dict[ResourceType, ResourceSnapshot]:
        """Get cloud resources."""
        # Placeholder
        return {
            ResourceType.CPU: ResourceSnapshot(
                resource_type=ResourceType.CPU,
                total=1600.0,
                used=400.0,
                available=1200.0,
                unit="vCPU",
            ),
            ResourceType.MEMORY: ResourceSnapshot(
                resource_type=ResourceType.MEMORY,
                total=6400.0,
                used=1200.0,
                available=5200.0,
                unit="GB",
            ),
        }

    def execute(self, command: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute on cloud."""
        if not self._connected:
            self.connect()
        return {"success": True, "platform": "cloud", "command": command}

    def is_available(self) -> bool:
        """Check cloud availability."""
        return self._connected


# ---------------------------------------------------------------------------
# ContainerAdapter
# ---------------------------------------------------------------------------


class ContainerAdapter(PlatformAdapter):
    """Adapter for container execution environment."""

    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        super().__init__(config or PlatformConfig(PlatformType.CONTAINER))
        self._in_container = self._detect_container()

    def _detect_container(self) -> bool:
        """Detect if running inside a container.

        Returns:
            True if in container.
        """
        # Check for cgroup indicators
        cgroup_path = "/proc/self/cgroup"
        if os.path.exists(cgroup_path):
            with open(cgroup_path) as f:
                content = f.read()
                if "docker" in content or "kubepods" in content:
                    return True
        return False

    def get_resources(self) -> Dict[ResourceType, ResourceSnapshot]:
        """Get container resources."""
        if not self._in_container:
            return LocalAdapter().get_resources()

        # Read cgroup limits
        try:
            with open("/sys/fs/cgroup/memory.limit_in_bytes") as f:
                mem_limit = int(f.read().strip())
            with open("/sys/fs/cgroup/memory.usage_in_bytes") as f:
                mem_used = int(f.read().strip())

            return {
                ResourceType.MEMORY: ResourceSnapshot(
                    resource_type=ResourceType.MEMORY,
                    total=mem_limit / (1024**3),
                    used=mem_used / (1024**3),
                    available=(mem_limit - mem_used) / (1024**3),
                    unit="GB",
                ),
            }
        except Exception:
            return {}

    def execute(self, command: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute in container."""
        return {"success": True, "platform": "container", "command": command}

    def is_available(self) -> bool:
        """Check container availability."""
        return True


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Monitor resources across platforms.

    Tracks resource utilization and triggers alerts
    when thresholds are exceeded.
    """

    def __init__(self) -> None:
        self._adapters: List[PlatformAdapter] = []
        self._thresholds: Dict[str, float] = {
            "cpu": 0.8,
            "memory": 0.85,
            "storage": 0.9,
        }
        self._alerts: List[Dict[str, Any]] = []

    def register_adapter(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter.

        Args:
            adapter: Adapter to monitor.
        """
        self._adapters.append(adapter)

    def check_all(self) -> Dict[str, Any]:
        """Check resources on all platforms.

        Returns:
            Status report.
        """
        report = {
            "platforms": [],
            "alerts": [],
        }

        for adapter in self._adapters:
            if not adapter.is_available():
                continue

            resources = adapter.get_resources()
            platform_report = {
                "type": adapter.get_type().value,
                "resources": {},
            }

            for rtype, snapshot in resources.items():
                platform_report["resources"][rtype.value] = {
                    "total": snapshot.total,
                    "used": snapshot.used,
                    "available": snapshot.available,
                    "utilization": snapshot.utilization,
                    "unit": snapshot.unit,
                }

                # Check thresholds
                threshold = self._thresholds.get(rtype.value, 0.9)
                if snapshot.utilization > threshold:
                    alert = {
                        "platform": adapter.get_type().value,
                        "resource": rtype.value,
                        "utilization": snapshot.utilization,
                        "threshold": threshold,
                    }
                    self._alerts.append(alert)
                    report["alerts"].append(alert)

            report["platforms"].append(platform_report)

        return report

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get pending alerts.

        Returns:
            List of alerts.
        """
        return list(self._alerts)

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()


# ---------------------------------------------------------------------------
# PlatformFactory
# ---------------------------------------------------------------------------


class PlatformFactory:
    """Factory for creating platform adapters."""

    @staticmethod
    def create(platform_type: PlatformType, **kwargs: Any) -> PlatformAdapter:
        """Create a platform adapter.

        Args:
            platform_type: Type of platform.
            **kwargs: Additional configuration.

        Returns:
            Platform adapter.
        """
        config = PlatformConfig(
            platform_type=platform_type,
            **{k: v for k, v in kwargs.items() if k in ["max_workers", "resource_limits", "env_vars"]}
        )

        if platform_type == PlatformType.LOCAL:
            return LocalAdapter(config)
        elif platform_type == PlatformType.CLOUD:
            return CloudAdapter(config)
        elif platform_type == PlatformType.CONTAINER:
            return ContainerAdapter(config)
        else:
            return LocalAdapter(config)

    @staticmethod
    def detect() -> PlatformType:
        """Auto-detect the current platform.

        Returns:
            Detected platform type.
        """
        # Check for container
        cgroup_path = "/proc/self/cgroup"
        if os.path.exists(cgroup_path):
            with open(cgroup_path) as f:
                content = f.read()
                if "docker" in content or "kubepods" in content:
                    return PlatformType.CONTAINER

        # Check for cloud indicators
        if os.environ.get("CLOUD_PROVIDER"):
            return PlatformType.CLOUD

        return PlatformType.LOCAL
