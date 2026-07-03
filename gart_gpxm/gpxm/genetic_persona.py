"""
Genetic Persona — Artist DNA + Skill-Inheritance Engine for GART v3.0.

Combines parent/child skill crossover with mutation rate control and
virtual-twin diff checking to produce artist DNA profiles.

Components:
    - ArtistDNA: Immutable dataclass holding skill inheritance data
    - SkillGene: Individual skill with dominance/recessiveness
    - GeneticPersona: Main builder class with crossover + mutation
    - GenealogicalTree: Family tree tracking across generations
    - TwinDiffChecker: Virtual-twin collision detection

Author: GART Architecture Team
Version: 3.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GeneticError(Exception):
    """Base exception for genetic persona operations."""


class MutationError(GeneticError):
    """Raised when mutation produces invalid genetic data."""


class CrossoverError(GeneticError):
    """Raised when crossover between incompatible DNAs fails."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GeneDominance(Enum):
    """Gene dominance levels for skill inheritance."""

    DOMINANT = "dominant"      # Always expressed when present
    RECESSIVE = "recessive"    # Only expressed without dominant allele
    CODOMINANT = "codominant"  # Both alleles expressed
    VARIABLE = "variable"      # Expression varies by context


class GeneType(Enum):
    """Types of genetic skill genes."""

    FLOW = "flow"
    DELIVERY = "delivery"
    LYRICAL_DENSITY = "lyrical_density"
    WORDPLAY = "wordplay"
    STORYTELLING = "storytelling"
    CADENCE = "cadence"
    RHYME_COMPLEXITY = "rhyme_complexity"
    VOCAL_PRESENCE = "vocal_presence"
    EMOTIONAL_RANGE = "emotional_range"
    CULTURAL_DEPTH = "cultural_depth"
    BREATH_CONTROL = "breath_control"
    AD_LIB_STYLE = "ad_lib_style"
    PUNCHLINE_DENSITY = "punchline_density"
    BEAT_SELECTION = "beat_selection"
    COLLAB_CHEMISTRY = "collab_chemistry"


class MutationType(Enum):
    """Types of mutations that can occur in artist DNA."""

    POINT = "point"          # Single skill value change
    INSERTION = "insertion"  # New skill gene added
    DELETION = "deletion"    # Skill gene removed
    INVERSION = "inversion"  # Skill order reversed
    AMPLIFICATION = "amplification"  # Skill value boosted
    EPISTATIC = "epistatic"  # Gene interaction change


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillGene:
    """An individual skill gene with genetic properties.

    Attributes:
        gene_type: Type of skill gene.
        base_value: Base skill level (0.0-1.0).
        dominance: Dominance level for inheritance.
        mutation_rate: Probability of mutation per generation.
        allele_variants: List of possible allele values.
        expression_threshold: Minimum value for phenotypic expression.
    """

    gene_type: GeneType
    base_value: float = 0.5
    dominance: GeneDominance = GeneDominance.DOMINANT
    mutation_rate: float = 0.05
    allele_variants: Tuple[float, ...] = field(default_factory=tuple)
    expression_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.base_value <= 1.0):
            raise ValueError(f"base_value must be 0.0-1.0, got {self.base_value}")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be 0.0-1.0, got {self.mutation_rate}")
        if not (0.0 <= self.expression_threshold <= 1.0):
            raise ValueError(f"expression_threshold must be 0.0-1.0")

    def mutate(self, mutation_type: MutationType, rng: random.Random) -> SkillGene:
        """Apply a mutation to this gene.

        Args:
            mutation_type: Type of mutation to apply.
            rng: Random number generator.

        Returns:
            New SkillGene with mutation applied.

        Raises:
            MutationError: If mutation produces invalid values.
        """
        try:
            if mutation_type == MutationType.POINT:
                delta = rng.uniform(-0.1, 0.1)
                new_value = max(0.0, min(1.0, self.base_value + delta))
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=new_value,
                    dominance=self.dominance,
                    mutation_rate=min(1.0, self.mutation_rate * 1.1),
                    allele_variants=self.allele_variants,
                    expression_threshold=self.expression_threshold,
                )
            elif mutation_type == MutationType.AMPLIFICATION:
                new_value = min(1.0, self.base_value * 1.2)
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=new_value,
                    dominance=self.dominance,
                    mutation_rate=self.mutation_rate,
                    allele_variants=self.allele_variants,
                    expression_threshold=self.expression_threshold,
                )
            elif mutation_type == MutationType.INSERTION:
                new_alleles = list(self.allele_variants)
                new_alleles.append(rng.uniform(0.0, 1.0))
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=self.base_value,
                    dominance=self.dominance,
                    mutation_rate=self.mutation_rate,
                    allele_variants=tuple(new_alleles),
                    expression_threshold=self.expression_threshold,
                )
            elif mutation_type == MutationType.DELETION:
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=max(0.0, self.base_value - 0.15),
                    dominance=self.dominance,
                    mutation_rate=self.mutation_rate,
                    allele_variants=self.allele_variants,
                    expression_threshold=self.expression_threshold,
                )
            elif mutation_type == MutationType.INVERSION:
                new_value = 1.0 - self.base_value
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=new_value,
                    dominance=self.dominance,
                    mutation_rate=self.mutation_rate,
                    allele_variants=self.allele_variants,
                    expression_threshold=self.expression_threshold,
                )
            elif mutation_type == MutationType.EPISTATIC:
                # Epistatic: change dominance pattern
                new_dominance = rng.choice(list(GeneDominance))
                return SkillGene(
                    gene_type=self.gene_type,
                    base_value=self.base_value,
                    dominance=new_dominance,
                    mutation_rate=self.mutation_rate,
                    allele_variants=self.allele_variants,
                    expression_threshold=self.expression_threshold,
                )
            else:
                raise MutationError(f"Unknown mutation type: {mutation_type}")
        except Exception as e:
            raise MutationError(f"Mutation failed for {self.gene_type}: {e}")

    def express(self) -> float:
        """Calculate phenotypic expression level of this gene.

        Returns:
            Expression value (0.0-1.0), 0.0 if below threshold.
        """
        if self.base_value < self.expression_threshold:
            return 0.0
        return self.base_value


@dataclass(frozen=True)
class ArtistDNA:
    """Immutable artist DNA profile containing all skill genes.

    Attributes:
        persona_id: Unique identifier for this DNA profile.
        generation: Generation number (0 = original).
        parent_ids: Tuple of parent persona IDs.
        genes: Dict of GeneType -> SkillGene mapping.
        timestamp: Creation timestamp.
        mutation_history: List of applied mutations.
        signature: SHA-256 fingerprint of the DNA.
    """

    persona_id: str
    generation: int = 0
    parent_ids: Tuple[str, ...] = field(default_factory=tuple)
    genes: Dict[GeneType, SkillGene] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    signature: str = ""

    def __post_init__(self) -> None:
        # Compute signature if not provided
        if not self.signature:
            sig = self._compute_signature()
            object.__setattr__(self, "signature", sig)

    def _compute_signature(self) -> str:
        """Compute SHA-256 signature of this DNA profile.

        Returns:
            Hex digest string.
        """
        data = {
            "persona_id": self.persona_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "genes": {k.value: v.base_value for k, v in self.genes.items()},
            "timestamp": self.timestamp,
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def get_expressed_traits(self) -> Dict[GeneType, float]:
        """Get all expressed (above-threshold) traits.

        Returns:
            Dict of GeneType to expression value.
        """
        return {
            gene_type: gene.express()
            for gene_type, gene in self.genes.items()
            if gene.express() > 0
        }

    def get_trait_value(self, gene_type: GeneType) -> float:
        """Get the expression value of a specific trait.

        Args:
            gene_type: The gene type to query.

        Returns:
            Expression value (0.0 if gene not present).
        """
        gene = self.genes.get(gene_type)
        return gene.express() if gene else 0.0

    def get_overall_fitness(self) -> float:
        """Compute overall fitness score across all traits.

        Returns:
            Average expression value of all genes.
        """
        if not self.genes:
            return 0.0
        return sum(g.express() for g in self.genes.values()) / len(self.genes)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DNA to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "persona_id": self.persona_id,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "genes": {
                k.value: {
                    "base_value": v.base_value,
                    "dominance": v.dominance.value,
                    "mutation_rate": v.mutation_rate,
                    "allele_variants": list(v.allele_variants),
                    "expression_threshold": v.expression_threshold,
                }
                for k, v in self.genes.items()
            },
            "timestamp": self.timestamp,
            "mutation_history": self.mutation_history,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArtistDNA:
        """Deserialize DNA from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ArtistDNA instance.
        """
        genes = {}
        for gene_name, gene_data in data.get("genes", {}).items():
            gene_type = GeneType(gene_name)
            genes[gene_type] = SkillGene(
                gene_type=gene_type,
                base_value=gene_data["base_value"],
                dominance=GeneDominance(gene_data["dominance"]),
                mutation_rate=gene_data["mutation_rate"],
                allele_variants=tuple(gene_data.get("allele_variants", [])),
                expression_threshold=gene_data.get("expression_threshold", 0.3),
            )

        return cls(
            persona_id=data["persona_id"],
            generation=data.get("generation", 0),
            parent_ids=tuple(data.get("parent_ids", [])),
            genes=genes,
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            mutation_history=data.get("mutation_history", []),
            signature=data.get("signature", ""),
        )


# ---------------------------------------------------------------------------
# TwinDiffChecker
# ---------------------------------------------------------------------------


class TwinDiffChecker:
    """Detects collisions between virtual twins (DNA siblings).

    Compares two DNA profiles to find genetic similarity and
    potential collision points where twins are too similar.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def compute_similarity(self, dna1: ArtistDNA, dna2: ArtistDNA) -> float:
        """Compute genetic similarity between two DNA profiles.

        Uses cosine similarity over the gene expression vectors.

        Args:
            dna1: First DNA profile.
            dna2: Second DNA profile.

        Returns:
            Similarity score (0.0-1.0).
        """
        all_genes = set(dna1.genes.keys()) | set(dna2.genes.keys())
        if not all_genes:
            return 0.0

        vec1 = [dna1.genes.get(g, SkillGene(g)).base_value for g in all_genes]
        vec2 = [dna2.genes.get(g, SkillGene(g)).base_value for g in all_genes]

        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def check_collision(self, dna1: ArtistDNA, dna2: ArtistDNA) -> Dict[str, Any]:
        """Check for collision between two DNA profiles.

        Args:
            dna1: First DNA profile.
            dna2: Second DNA profile.

        Returns:
            Collision report dictionary.
        """
        similarity = self.compute_similarity(dna1, dna2)
        is_collision = similarity >= self.similarity_threshold

        # Find specific gene collisions
        gene_collisions: List[Dict[str, Any]] = []
        for gene_type in set(dna1.genes.keys()) & set(dna2.genes.keys()):
            g1 = dna1.genes[gene_type]
            g2 = dna2.genes[gene_type]
            gene_sim = 1.0 - abs(g1.base_value - g2.base_value)
            if gene_sim >= self.similarity_threshold:
                gene_collisions.append({
                    "gene_type": gene_type.value,
                    "similarity": gene_sim,
                    "values": [g1.base_value, g2.base_value],
                })

        return {
            "collision_detected": is_collision,
            "similarity_score": similarity,
            "threshold": self.similarity_threshold,
            "gene_collisions": gene_collisions,
            "dna1_id": dna1.persona_id,
            "dna2_id": dna2.persona_id,
        }

    def find_all_collisions(
        self,
        dna_profiles: List[ArtistDNA],
    ) -> List[Dict[str, Any]]:
        """Find all collision pairs in a population.

        Args:
            dna_profiles: List of DNA profiles to check.

        Returns:
            List of collision reports.
        """
        collisions = []
        for i in range(len(dna_profiles)):
            for j in range(i + 1, len(dna_profiles)):
                report = self.check_collision(dna_profiles[i], dna_profiles[j])
                if report["collision_detected"]:
                    collisions.append(report)
        return collisions


# ---------------------------------------------------------------------------
# GenealogicalTree
# ---------------------------------------------------------------------------


class GenealogicalTree:
    """Tracks family lineage across generations.

    Maintains a directed graph of parent-child relationships
    and provides genealogical queries.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Tuple[str, str]] = []
        self._generation_map: Dict[str, int] = {}

    def register_birth(
        self,
        child_id: str,
        parent_ids: List[str],
        generation: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new persona birth in the family tree.

        Args:
            child_id: New persona ID.
            parent_ids: List of parent persona IDs.
            generation: Generation number.
            metadata: Optional metadata.
        """
        self._nodes[child_id] = {
            "id": child_id,
            "parents": parent_ids,
            "children": [],
            "generation": generation,
            "metadata": metadata or {},
        }
        self._generation_map[child_id] = generation

        for parent_id in parent_ids:
            self._edges.append((parent_id, child_id))
            if parent_id in self._nodes:
                self._nodes[parent_id]["children"].append(child_id)

    def get_ancestors(self, persona_id: str, depth: int = 3) -> List[str]:
        """Get ancestors of a persona up to specified depth.

        Args:
            persona_id: Persona to query.
            depth: Maximum ancestor depth.

        Returns:
            List of ancestor IDs.
        """
        ancestors = []
        current_gen = [persona_id]
        for _ in range(depth):
            next_gen = []
            for pid in current_gen:
                node = self._nodes.get(pid)
                if node:
                    for parent in node["parents"]:
                        if parent not in ancestors:
                            ancestors.append(parent)
                            next_gen.append(parent)
            current_gen = next_gen
            if not current_gen:
                break
        return ancestors

    def get_descendants(self, persona_id: str, depth: int = 3) -> List[str]:
        """Get descendants of a persona up to specified depth.

        Args:
            persona_id: Persona to query.
            depth: Maximum descendant depth.

        Returns:
            List of descendant IDs.
        """
        descendants = []
        current_gen = [persona_id]
        for _ in range(depth):
            next_gen = []
            for pid in current_gen:
                node = self._nodes.get(pid)
                if node:
                    for child in node["children"]:
                        if child not in descendants:
                            descendants.append(child)
                            next_gen.append(child)
            current_gen = next_gen
            if not current_gen:
                break
        return descendants

    def get_siblings(self, persona_id: str) -> List[str]:
        """Get siblings of a persona (same parents).

        Args:
            persona_id: Persona to query.

        Returns:
            List of sibling IDs.
        """
        node = self._nodes.get(persona_id)
        if not node:
            return []

        siblings = []
        for parent in node["parents"]:
            parent_node = self._nodes.get(parent)
            if parent_node:
                for child in parent_node["children"]:
                    if child != persona_id and child not in siblings:
                        siblings.append(child)
        return siblings

    def get_generation(self, generation: int) -> List[str]:
        """Get all persona IDs in a given generation.

        Args:
            generation: Generation number.

        Returns:
            List of persona IDs.
        """
        return [
            pid for pid, gen in self._generation_map.items()
            if gen == generation
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "nodes": self._nodes,
            "edges": self._edges,
            "generation_map": self._generation_map,
        }


# ---------------------------------------------------------------------------
# GeneticPersona — Main builder class
# ---------------------------------------------------------------------------


class GeneticPersona:
    """Genetic Persona builder for GART v3.0.

    Creates and evolves artist DNA profiles through crossover,
    mutation, and selection operations.

    Attributes:
        rng: Seeded random number generator.
        twin_checker: Virtual-twin collision detector.
        genealogy: Family tree tracking.
        generation: Current generation counter.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.rng = random.Random(seed)
        self.twin_checker = TwinDiffChecker(similarity_threshold)
        self.genealogy = GenealogicalTree()
        self.generation = 0
        self._persona_counter = 0

    # --- Persona creation ---

    def create_founder(
        self,
        name: str,
        skill_profile: Optional[Dict[GeneType, float]] = None,
    ) -> ArtistDNA:
        """Create a founder (generation 0) persona.

        Args:
            name: Persona display name.
            skill_profile: Optional skill value overrides.

        Returns:
            New ArtistDNA for the founder.
        """
        self._persona_counter += 1
        persona_id = f"{name}_G0_{self._persona_counter}"

        genes: Dict[GeneType, SkillGene] = {}
        for gene_type in GeneType:
            base_value = skill_profile.get(gene_type, 0.5) if skill_profile else 0.5
            genes[gene_type] = SkillGene(
                gene_type=gene_type,
                base_value=base_value,
                dominance=self.rng.choice(list(GeneDominance)),
                mutation_rate=self.rng.uniform(0.01, 0.1),
                allele_variants=tuple(self.rng.uniform(0, 1) for _ in range(3)),
            )

        dna = ArtistDNA(
            persona_id=persona_id,
            generation=0,
            genes=genes,
        )

        self.genealogy.register_birth(
            child_id=persona_id,
            parent_ids=[],
            generation=0,
            metadata={"name": name, "type": "founder"},
        )

        return dna

    def create_offspring(
        self,
        parent1: ArtistDNA,
        parent2: ArtistDNA,
        name: Optional[str] = None,
    ) -> ArtistDNA:
        """Create an offspring through genetic crossover.

        Args:
            parent1: First parent DNA.
            parent2: Second parent DNA.
            name: Optional offspring name.

        Returns:
            New ArtistDNA for the offspring.

        Raises:
            CrossoverError: If parents are incompatible.
        """
        child_generation = max(parent1.generation, parent2.generation) + 1
        self._persona_counter += 1
        child_name = name or f"Offspring_{self._persona_counter}"
        child_id = f"{child_name}_G{child_generation}_{self._persona_counter}"

        # Crossover: inherit genes from both parents
        child_genes = self._crossover(parent1, parent2)

        # Apply mutations
        mutated_genes, mutation_log = self._mutate(child_genes)

        dna = ArtistDNA(
            persona_id=child_id,
            generation=child_generation,
            parent_ids=(parent1.persona_id, parent2.persona_id),
            genes=mutated_genes,
            mutation_history=mutation_log,
        )

        self.genealogy.register_birth(
            child_id=child_id,
            parent_ids=[parent1.persona_id, parent2.persona_id],
            generation=child_generation,
            metadata={"name": child_name, "type": "offspring"},
        )

        return dna

    def _crossover(
        self,
        parent1: ArtistDNA,
        parent2: ArtistDNA,
    ) -> Dict[GeneType, SkillGene]:
        """Perform genetic crossover between two parents.

        Args:
            parent1: First parent DNA.
            parent2: Second parent DNA.

        Returns:
            Child gene dictionary.

        Raises:
            CrossoverError: If crossover fails.
        """
        try:
            child_genes: Dict[GeneType, SkillGene] = {}
            all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())

            for gene_type in all_genes:
                g1 = parent1.genes.get(gene_type)
                g2 = parent2.genes.get(gene_type)

                if g1 and g2:
                    # Both parents have gene — apply dominance rules
                    if g1.dominance == GeneDominance.DOMINANT and g2.dominance != GeneDominance.DOMINANT:
                        child_genes[gene_type] = g1
                    elif g2.dominance == GeneDominance.DOMINANT and g1.dominance != GeneDominance.DOMINANT:
                        child_genes[gene_type] = g2
                    elif g1.dominance == GeneDominance.CODOMINANT or g2.dominance == GeneDominance.CODOMINANT:
                        # Average both
                        avg_value = (g1.base_value + g2.base_value) / 2
                        child_genes[gene_type] = SkillGene(
                            gene_type=gene_type,
                            base_value=avg_value,
                            dominance=GeneDominance.CODOMINANT,
                            mutation_rate=(g1.mutation_rate + g2.mutation_rate) / 2,
                        )
                    else:
                        # Random selection
                        child_genes[gene_type] = self.rng.choice([g1, g2])
                elif g1:
                    child_genes[gene_type] = g1
                elif g2:
                    child_genes[gene_type] = g2

            return child_genes

        except Exception as e:
            raise CrossoverError(f"Crossover failed: {e}")

    def _mutate(
        self,
        genes: Dict[GeneType, SkillGene],
    ) -> Tuple[Dict[GeneType, SkillGene], List[Dict[str, Any]]]:
        """Apply random mutations to a gene set.

        Args:
            genes: Gene dictionary to mutate.

        Returns:
            Tuple of (mutated genes, mutation log).
        """
        mutated = dict(genes)
        mutation_log: List[Dict[str, Any]] = []

        for gene_type, gene in genes.items():
            if self.rng.random() < gene.mutation_rate:
                mutation_type = self.rng.choice(list(MutationType))
                try:
                    mutated_gene = gene.mutate(mutation_type, self.rng)
                    mutated[gene_type] = mutated_gene
                    mutation_log.append({
                        "gene": gene_type.value,
                        "type": mutation_type.value,
                        "before": gene.base_value,
                        "after": mutated_gene.base_value,
                    })
                except MutationError as e:
                    logger.warning("Mutation failed for %s: %s", gene_type, e)

        return mutated, mutation_log

    # --- Selection and evolution ---

    def select_best(
        self,
        population: List[ArtistDNA],
        n: int = 2,
    ) -> List[ArtistDNA]:
        """Select the fittest individuals from a population.

        Args:
            population: List of DNA profiles.
            n: Number to select.

        Returns:
            List of selected DNA profiles.
        """
        scored = [(dna, dna.get_overall_fitness()) for dna in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [dna for dna, _ in scored[:n]]

    def evolve_generation(
        self,
        population: List[ArtistDNA],
        offspring_count: int = 4,
    ) -> List[ArtistDNA]:
        """Evolve a new generation from the current population.

        Args:
            population: Current population.
            offspring_count: Number of offspring to produce.

        Returns:
            List of new offspring DNA profiles.
        """
        if len(population) < 2:
            return []

        # Select parents
        parents = self.select_best(population, n=min(4, len(population)))

        offspring = []
        for _ in range(offspring_count):
            p1, p2 = self.rng.sample(parents, 2)
            try:
                child = self.create_offspring(p1, p2)
                offspring.append(child)
            except CrossoverError as e:
                logger.warning("Offspring creation failed: %s", e)

        self.generation += 1
        return offspring

    def check_twin_collision(
        self,
        dna1: ArtistDNA,
        dna2: ArtistDNA,
    ) -> Dict[str, Any]:
        """Check for virtual-twin collision between two personas.

        Args:
            dna1: First DNA.
            dna2: Second DNA.

        Returns:
            Collision report.
        """
        return self.twin_checker.check_collision(dna1, dna2)

    def get_genealogy_report(self, persona_id: str) -> Dict[str, Any]:
        """Get full genealogical report for a persona.

        Args:
            persona_id: Persona to query.

        Returns:
            Genealogy report dictionary.
        """
        return {
            "persona_id": persona_id,
            "ancestors": self.genealogy.get_ancestors(persona_id),
            "descendants": self.genealogy.get_descendants(persona_id),
            "siblings": self.genealogy.get_siblings(persona_id),
        }
