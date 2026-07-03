"""
Cross-Platform Content Adapter — GART v3.0.

Adapts generated content for distribution across multiple platforms:
    - LinkedIn: Professional, insightful, long-form
    - Twitter/X: Punchy, conversational, thread-friendly
    - GitHub README: Technical, documented, code-focused
    - WeChat: Concise, mobile-optimized
    - Zhihu: Educational, analytical, reference-rich

Uses the Strategy pattern for platform-specific transformers.

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class Platform(Enum):
    """Supported content distribution platforms."""

    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    GITHUB = "github"
    WECHAT = "wechat"
    ZHIHU = "zhihu"


# ---------------------------------------------------------------------------
# ContentTransformer (Strategy pattern)
# ---------------------------------------------------------------------------


class ContentTransformer(ABC):
    """Abstract base class for platform content transformers.

    Implements the Strategy pattern — each platform defines its own
    transformation rules for adapting content to its audience.
    """

    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    @abstractmethod
    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for this platform.

        Args:
            content: Raw content to transform.
            metadata: Optional content metadata.

        Returns:
            Platform-formatted content.
        """
        ...

    @abstractmethod
    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate content meets platform requirements.

        Args:
            content: Content to validate.

        Returns:
            (is_valid, list_of_issues).
        """
        ...

    @abstractmethod
    def get_constraints(self) -> Dict[str, Any]:
        """Get platform content constraints.

        Returns:
            Dictionary with constraint parameters.
        """
        ...


# ---------------------------------------------------------------------------
# LinkedInTransformer
# ---------------------------------------------------------------------------


class LinkedInTransformer(ContentTransformer):
    """LinkedIn content transformer.

    Style: Professional, insightful, long-form
    Constraints: 3000 chars max, professional tone
    """

    MAX_LENGTH: int = 3000

    def __init__(self) -> None:
        super().__init__(Platform.LINKEDIN)

    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for LinkedIn.

        Adds professional framing, hashtags, and call-to-action.

        Args:
            content: Raw content.
            metadata: Optional metadata.

        Returns:
            LinkedIn-formatted content.
        """
        lines = [
            content,
            "",
            "---",
            "",
            "What are your thoughts on this approach? Share your insights below. 👇",
            "",
            "#AIMusic #GeneticPersona #MusicTech #Innovation",
        ]
        result = "\n".join(lines)

        # Truncate if needed
        if len(result) > self.MAX_LENGTH:
            result = result[:self.MAX_LENGTH - 3] + "..."

        return result

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate LinkedIn content.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        issues: List[str] = []
        if len(content) > self.MAX_LENGTH:
            issues.append(f"Content exceeds {self.MAX_LENGTH} characters")
        return (len(issues) == 0, issues)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_length": self.MAX_LENGTH,
            "tone": "professional",
            "hashtags": True,
            "cta": True,
        }


# ---------------------------------------------------------------------------
# TwitterTransformer
# ---------------------------------------------------------------------------


class TwitterTransformer(ContentTransformer):
    """Twitter/X content transformer.

    Style: Punchy, conversational, thread-friendly
    Constraints: 280 chars per tweet, thread support
    """

    MAX_TWEET_LENGTH: int = 280

    def __init__(self) -> None:
        super().__init__(Platform.TWITTER)

    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for Twitter.

        Breaks content into tweet-sized chunks for threading.

        Args:
            content: Raw content.
            metadata: Optional metadata.

        Returns:
            Twitter-formatted content (thread if needed).
        """
        sentences = content.replace("\n", " ").split(". ")
        tweets: List[str] = []
        current_tweet = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not sentence.endswith("."):
                sentence += "."

            test_tweet = current_tweet + " " + sentence if current_tweet else sentence
            if len(test_tweet) <= self.MAX_TWEET_LENGTH:
                current_tweet = test_tweet
            else:
                if current_tweet:
                    tweets.append(current_tweet.strip())
                current_tweet = sentence

        if current_tweet:
            tweets.append(current_tweet.strip())

        # Format as thread
        if len(tweets) <= 1:
            return tweets[0] if tweets else content[:self.MAX_TWEET_LENGTH]

        lines: List[str] = []
        for i, tweet in enumerate(tweets):
            lines.append(f"({i+1}/{len(tweets)}) {tweet}")
        return "\n\n".join(lines)

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate Twitter content.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        issues: List[str] = []
        tweets = content.split("\n\n")
        for i, tweet in enumerate(tweets):
            # Remove thread marker
            clean = tweet.split(" ", 1)[1] if tweet.startswith("(") and ")" in tweet[:10] else tweet
            if len(clean) > self.MAX_TWEET_LENGTH:
                issues.append(f"Tweet {i+1} exceeds {self.MAX_TWEET_LENGTH} chars")
        return (len(issues) == 0, issues)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "max_tweet_length": self.MAX_TWEET_LENGTH,
            "tone": "conversational",
            "threads": True,
        }


# ---------------------------------------------------------------------------
# GitHubReadmeTransformer
# ---------------------------------------------------------------------------


class GitHubReadmeTransformer(ContentTransformer):
    """GitHub README content transformer.

    Style: Technical, documented, code-focused
    Constraints: Markdown format, code blocks, badges
    """

    def __init__(self) -> None:
        super().__init__(Platform.GITHUB)

    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for GitHub README.

        Adds markdown formatting, code blocks, and documentation structure.

        Args:
            content: Raw content.
            metadata: Optional metadata with version, badges, etc.

        Returns:
            GitHub-formatted markdown.
        """
        meta = metadata or {}
        version = meta.get("version", "3.0.0")
        title = meta.get("title", "GART + GPXM")

        lines = [
            f"# {title}",
            "",
            f"![Version](https://img.shields.io/badge/version-{version}-blue)",
            "![Python](https://img.shields.io/badge/python-3.10%2B-green)",
            "![License](https://img.shields.io/badge/license-MIT-yellow)",
            "",
            "## Overview",
            "",
            content,
            "",
            "## Quick Start",
            "",
            "```python",
            "from gart_gpxm import DualSwarmOrchestrator",
            "",
            "orchestrator = DualSwarmOrchestrator()",
            "result = await orchestrator.coordinate_battle(artist_a, artist_b)",
            "```",
            "",
            "## Architecture",
            "",
            "This project implements a dual-swarm AI music production system",
            "combining genetic personas with expansive memory and quantum",
            "linguistic synthesis.",
            "",
            "## Documentation",
            "",
            "See the `docs/` directory for full documentation.",
        ]

        return "\n".join(lines)

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate GitHub README content.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        issues: List[str] = []
        if "# " not in content:
            issues.append("Missing H1 heading")
        if "```" not in content:
            issues.append("Missing code block examples")
        return (len(issues) == 0, issues)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "format": "markdown",
            "tone": "technical",
            "code_examples": True,
        }


# ---------------------------------------------------------------------------
# WeChatTransformer
# ---------------------------------------------------------------------------


class WeChatTransformer(ContentTransformer):
    """WeChat content transformer.

    Style: Concise, mobile-optimized, emoji-friendly
    Constraints: Mobile-friendly length
    """

    def __init__(self) -> None:
        super().__init__(Platform.WECHAT)

    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for WeChat.

        Shortens and simplifies for mobile reading.

        Args:
            content: Raw content.
            metadata: Optional metadata.

        Returns:
            WeChat-formatted content.
        """
        # Simplify to key points
        lines = content.split("\n")
        key_lines = [l for l in lines if l.strip() and not l.startswith("#")]

        # Take first few meaningful lines
        summary = key_lines[:10] if len(key_lines) > 10 else key_lines

        result = "\n".join(summary)

        # Add WeChat-style ending
        result += "\n\n---\n了解更多，请关注我们的公众号"

        return result

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate WeChat content.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        issues: List[str] = []
        if len(content) > 2000:
            issues.append("Content may be too long for mobile")
        return (len(issues) == 0, issues)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "tone": "concise",
            "mobile_optimized": True,
            "max_length": 2000,
        }


# ---------------------------------------------------------------------------
# ZhihuTransformer
# ---------------------------------------------------------------------------


class ZhihuTransformer(ContentTransformer):
    """Zhihu content transformer.

    Style: Educational, analytical, reference-rich
    Constraints: Long-form, academic tone
    """

    def __init__(self) -> None:
        super().__init__(Platform.ZHIHU)

    def transform(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Transform content for Zhihu.

        Adds analytical framing and reference structure.

        Args:
            content: Raw content.
            metadata: Optional metadata.

        Returns:
            Zhihu-formatted content.
        """
        lines = [
            "## 引言",
            "",
            content,
            "",
            "## 技术分析",
            "",
            "本文涉及的技术框架包括：",
            "- 遗传人格建模 (Genetic Persona Modeling)",
            "- 扩展记忆系统 (Expansive Memory Bank)",
            "- 量子语言合成器 (Quantum Linguistic Synthesizer)",
            "- 对偶群体协调 (Dual-Swarm Orchestration)",
            "",
            "## 参考文献",
            "",
            "1. GART v3.0 Architecture Specification",
            "2. GPXM Pipeline Design Document",
            "3. Entropic Scripting Framework Guide",
            "",
            "---",
            "",
            "欢迎在评论区讨论技术细节。",
        ]

        return "\n".join(lines)

    def validate(self, content: str) -> Tuple[bool, List[str]]:
        """Validate Zhihu content.

        Args:
            content: Content to check.

        Returns:
            Validation result.
        """
        issues: List[str] = []
        if "##" not in content:
            issues.append("Missing section headers")
        return (len(issues) == 0, issues)

    def get_constraints(self) -> Dict[str, Any]:
        return {
            "tone": "educational",
            "format": "markdown",
            "references": True,
        }


# ---------------------------------------------------------------------------
# CrossPlatformAdapter
# ---------------------------------------------------------------------------


class CrossPlatformAdapter:
    """Cross-Platform Content Adapter.

    Adapts generated content for distribution across multiple platforms.
    Uses the Strategy pattern for platform-specific transformers.

    Attributes:
        transformers: Dict mapping Platform to ContentTransformer.
    """

    def __init__(self) -> None:
        self.transformers: Dict[Platform, ContentTransformer] = {
            Platform.LINKEDIN: LinkedInTransformer(),
            Platform.TWITTER: TwitterTransformer(),
            Platform.GITHUB: GitHubReadmeTransformer(),
            Platform.WECHAT: WeChatTransformer(),
            Platform.ZHIHU: ZhihuTransformer(),
        }

    def distribute(
        self,
        content: str,
        platforms: List[Platform],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[Platform, str]:
        """Transform content for multiple platforms.

        Args:
            content: Raw content to distribute.
            platforms: List of target platforms.
            metadata: Optional content metadata.

        Returns:
            Dict mapping Platform to formatted content.

        Raises:
            ValueError: If an unsupported platform is requested.
        """
        results: Dict[Platform, str] = {}

        for platform in platforms:
            transformer = self.transformers.get(platform)
            if transformer is None:
                raise ValueError(f"Unsupported platform: {platform}")

            transformed = transformer.transform(content, metadata)
            results[platform] = transformed
            logger.info("Distributed to %s (%d chars)", platform.value, len(transformed))

        return results

    def adapt_tone(self, content: str, platform: Platform) -> str:
        """Adapt content tone for a specific platform.

        Args:
            content: Content to adapt.
            platform: Target platform.

        Returns:
            Tone-adapted content.
        """
        transformer = self.transformers.get(platform)
        if transformer is None:
            return content

        constraints = transformer.get_constraints()
        tone = constraints.get("tone", "neutral")

        tone_prefixes: Dict[str, str] = {
            "professional": "",
            "conversational": "",
            "technical": "",
            "concise": "",
            "educational": "",
        }

        # Transform
        return transformer.transform(content)

    def validate_all(
        self,
        content: str,
        platforms: List[Platform],
    ) -> Dict[Platform, Tuple[bool, List[str]]]:
        """Validate content for all specified platforms.

        Args:
            content: Content to validate.
            platforms: Platforms to validate against.

        Returns:
            Dict mapping Platform to validation results.
        """
        results: Dict[Platform, Tuple[bool, List[str]]] = {}

        for platform in platforms:
            transformer = self.transformers.get(platform)
            if transformer:
                results[platform] = transformer.validate(content)

        return results

    def add_transformer(
        self,
        platform: Platform,
        transformer: ContentTransformer,
    ) -> None:
        """Register a custom platform transformer.

        Args:
            platform: Platform identifier.
            transformer: Transformer instance.
        """
        self.transformers[platform] = transformer
        logger.info("Registered transformer for %s", platform.value)

    def get_platform_constraints(self, platform: Platform) -> Dict[str, Any]:
        """Get constraints for a platform.

        Args:
            platform: Platform to query.

        Returns:
            Constraints dictionary.
        """
        transformer = self.transformers.get(platform)
        if transformer:
            return transformer.get_constraints()
        return {}

    def get_summary(self, distribution: Dict[Platform, str]) -> str:
        """Generate summary of distribution results.

        Args:
            distribution: Results from distribute().

        Returns:
            Summary string.
        """
        lines = ["Cross-Platform Distribution Summary:", ""]
        for platform, content in distribution.items():
            lines.append(f"  {platform.value}: {len(content)} characters")
        lines.append(f"\n  Total platforms: {len(distribution)}")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Cross-Platform Adapter loaded successfully.")

    adapter = CrossPlatformAdapter()

    sample_content = (
        "GART v3.0 + GPXM is a dual-swarm AI music production system "
        "that uses genetic personas with expansive memory to create "
        "authentic-sounding artist collaborations. The system combines "
        "PyTorch neural architectures with entropic scripting for "
        "style-accurate music generation."
    )

    platforms = [Platform.TWITTER, Platform.LINKEDIN, Platform.GITHUB]
    results = adapter.distribute(sample_content, platforms)

    for platform, content in results.items():
        print(f"\n--- {platform.value.upper()} ---")
        print(content[:200] + "...")

    print(f"\n{adapter.get_summary(results)}")
