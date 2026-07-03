"""GART v3.0 + GPXM Quantum Linguistic Synthesizer.

A dual-swarm AI music production system combining the Generative
Adversarial Rap Tournament (GART) with Genetic Persona & Expansive
Memory (GPXM) frameworks.

Top-level package exports.
"""

from __future__ import annotations

# Core modules
from gart_gpxm.core.lllm_architecture import (
    LLLM_Architecture,
    LLLM_Config,
    LLLM_TrainingPipeline,
    PositionalEncoding,
    V1Encoder,
    V2Encoder,
    CrossAttentionFusion,
    OutputDecoder,
    TrainingConfig,
    TrainingHistory,
    generate_square_subsequent_mask,
)

from gart_gpxm.core.daw_utilities import (
    AudioSignal,
    EQProfile,
    ReverbConfig,
    DAWConfig,
    DAW_SubAgent_Utilities,
    VerseStem,
    Pattern,
    Note,
    CadenceShiftedText,
    generate_ybnba_kick_pattern,
    generate_hive_synth_riff,
)

from gart_gpxm.core.dual_swarm_orchestrator import (
    MatchupConfig,
    BattleResult,
    TournamentResult,
    SwarmCoordinationResult,
    LinguisticEngine,
    VerseConstructor,
    StemGenerator,
    LinguisticDAW,
    HiveMindAggregator,
    StyleSynthesizer,
    EntropyInjector,
    FrankensteinHiveMind,
    SidechainCompressor,
    MasterMixerEvaluator,
    DualSwarmOrchestrator,
)

# GPXM modules
from gart_gpxm.gpxm.genetic_persona import (
    VoiceDifferentiationParameters,
    EntropicScript,
    MemoryEntry,
    SessionMemory,
    ProjectMemory,
    CoreIdentityMemory,
    Episode,
    EventMemory,
    Fact,
    KnowledgeGraph,
    KnowledgeMemory,
    ConsolidationResult,
    ExpansiveMemoryBank,
    GeneticPersona,
)

from gart_gpxm.gpxm.entropic_scripting import (
    VocabularyMatrix,
    FlowArchitecture,
    ThematicEngine,
    ProsodicFeatures,
    CulturalSemantics,
    Guardrails,
    EntropicScript,
    EntropyLevel,
    EntropyController,
)

from gart_gpxm.gpxm.roster_manager import (
    SlotInfo,
    RosterError,
    RosterFullError,
    DuplicateArtistError,
    RosterManager,
    cosine_similarity,
)

# Inversion module
from gart_gpxm.inversion.diagram_to_code_agent import (
    Token,
    ASTNode,
    ClassNode,
    MethodNode,
    Edge,
    MermaidAST,
    MermaidLexer,
    MermaidASTParser,
    DependencyMapper,
    PythonCodeGenerator,
    DiagramToCodeInversionAgent,
)

# Resilience module
from gart_gpxm.resilience.adaptive_engine import (
    FailureSeverity,
    DegradedMode,
    FailureEvent,
    RecoveryAction,
    SystemState,
    AdaptationDecision,
    AdaptationEvent,
    EntropyRegulator,
    GracefulDegradation,
    FeedbackLoop,
    AdaptiveResilienceEngine,
)

# Tournament module
from gart_gpxm.tournament.tournament_engine import (
    BattleResult as TBattleResult,
    LeaderboardEntry,
    TournamentResult as TTournamentResult,
    ScoringEngine,
    EloRatingSystem,
    MatchupSimulator,
    TournamentBracket,
    TournamentEngine,
)

# Stash Box module
from gart_gpxm.stash_box.reconstruction_engine import (
    StyleBrief,
    BeatPattern,
    LyricalSection,
    TrackConcept,
    BlendedStyle,
    StashBoxReconstructionEngine,
)

# Cross-platform module
from gart_gpxm.cross_platform.adapter import (
    Platform,
    ContentTransformer,
    LinkedInTransformer,
    TwitterTransformer,
    GitHubReadmeTransformer,
    WeChatTransformer,
    ZhihuTransformer,
    CrossPlatformAdapter,
)

__version__ = "3.0.0"
__author__ = "GART Architecture Team"

__all__ = [
    # LLLM
    "LLLM_Architecture",
    "LLLM_Config",
    "LLLM_TrainingPipeline",
    "PositionalEncoding",
    "V1Encoder",
    "V2Encoder",
    "CrossAttentionFusion",
    "OutputDecoder",
    "TrainingConfig",
    "TrainingHistory",
    # DAW
    "AudioSignal",
    "EQProfile",
    "ReverbConfig",
    "DAWConfig",
    "DAW_SubAgent_Utilities",
    "VerseStem",
    "Pattern",
    "Note",
    "CadenceShiftedText",
    # Dual Swarm
    "MatchupConfig",
    "BattleResult",
    "TournamentResult",
    "SwarmCoordinationResult",
    "LinguisticEngine",
    "VerseConstructor",
    "StemGenerator",
    "LinguisticDAW",
    "HiveMindAggregator",
    "StyleSynthesizer",
    "EntropyInjector",
    "FrankensteinHiveMind",
    "SidechainCompressor",
    "MasterMixerEvaluator",
    "DualSwarmOrchestrator",
    # GPXM
    "VoiceDifferentiationParameters",
    "MemoryEntry",
    "SessionMemory",
    "ProjectMemory",
    "CoreIdentityMemory",
    "Episode",
    "EventMemory",
    "Fact",
    "KnowledgeGraph",
    "KnowledgeMemory",
    "ConsolidationResult",
    "ExpansiveMemoryBank",
    "GeneticPersona",
    # Entropic Scripting
    "VocabularyMatrix",
    "FlowArchitecture",
    "ThematicEngine",
    "ProsodicFeatures",
    "CulturalSemantics",
    "Guardrails",
    "EntropyLevel",
    "EntropyController",
    # Roster
    "SlotInfo",
    "RosterError",
    "RosterFullError",
    "DuplicateArtistError",
    "RosterManager",
    "cosine_similarity",
    # Inversion
    "Token",
    "ASTNode",
    "ClassNode",
    "MethodNode",
    "Edge",
    "MermaidAST",
    "MermaidLexer",
    "MermaidASTParser",
    "DependencyMapper",
    "PythonCodeGenerator",
    "DiagramToCodeInversionAgent",
    # Resilience
    "FailureSeverity",
    "DegradedMode",
    "FailureEvent",
    "RecoveryAction",
    "SystemState",
    "AdaptationDecision",
    "AdaptationEvent",
    "EntropyRegulator",
    "GracefulDegradation",
    "FeedbackLoop",
    "AdaptiveResilienceEngine",
    # Tournament
    "LeaderboardEntry",
    "ScoringEngine",
    "EloRatingSystem",
    "MatchupSimulator",
    "TournamentBracket",
    "TournamentEngine",
    # Stash Box
    "StyleBrief",
    "BeatPattern",
    "LyricalSection",
    "TrackConcept",
    "BlendedStyle",
    "StashBoxReconstructionEngine",
    # Cross-platform
    "Platform",
    "ContentTransformer",
    "LinkedInTransformer",
    "TwitterTransformer",
    "GitHubReadmeTransformer",
    "WeChatTransformer",
    "ZhihuTransformer",
    "CrossPlatformAdapter",
]
