"""Aquaculture Biomass Estimation Engine.

Provides species-specific length-weight relationships based on peer-reviewed fisheries
literature (Froese 2006, FishBase), calibrated length-girth approximations, and batch
pond population biomass aggregation.

IMPORTANT METHODOLOGICAL NOTICE:
These mathematical models require physical metric input measurements (cm, kg) obtained
via calipers, measuring boards, or optical calibration markers. Single 2D camera images
CANNOT measure metric length or weight without physical scale calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class SpeciesBiomassParameters:
    species_id: str
    scientific_name: str
    # Length-weight formula: W(g) = a * L(cm)^b (FishBase / Froese 2006)
    a_coefficient: float
    b_exponent: float
    # Length-girth formula: W(g) = (L * G^2) / C_shape
    c_shape_divisor: float
    citation: str
    length_convention: str = "Total Length (TL, cm)"
    weight_units: str = "grams (g) output, scaled to kg"
    population_context: str = "General aquaculture / wild populations"
    limitations: str = "Assumes isometric/allometric baseline; varies with reproductive stage and feeding."
    is_experimental_girth: bool = True


# Calibrated biological parameters from FishBase (Froese 2006 standard consensus)
# Note: Length in cm, weight output in grams.
SPECIES_BIOMASS_REGISTRY: dict[str, SpeciesBiomassParameters] = {
    "tilapia": SpeciesBiomassParameters(
        species_id="tilapia",
        scientific_name="Oreochromis niloticus",
        a_coefficient=0.0177,
        b_exponent=2.99,
        c_shape_divisor=29.0,
        citation="Froese (2006), FishBase consensus for Oreochromis niloticus; Schneider (1990) FAO Field Guide.",
        length_convention="Total Length (TL, cm)",
        weight_units="grams (g), converted to kg",
        population_context="Nile tilapia farmed in tropical freshwater aquaculture ponds and earthen impoundments.",
        limitations="Assumes near-isometric growth (b=2.99). Satiation, lipid accumulation, and egg incubation cause +/-12% variance.",
        is_experimental_girth=True,
    ),
    "catfish": SpeciesBiomassParameters(
        species_id="catfish",
        scientific_name="Clarias gariepinus",
        a_coefficient=0.0068,
        b_exponent=3.05,
        c_shape_divisor=38.0,
        citation="Froese (2006), FishBase consensus for Clarias gariepinus; Teugels (1986). Slender anguilliform profile.",
        length_convention="Total Length (TL, cm)",
        weight_units="grams (g), converted to kg",
        population_context="African sharptooth catfish in intensive recirculating aquaculture (RAS) and concrete tanks.",
        limitations="Slender anguilliform morphology with large bony head plate; large fish (>50 cm) may exhibit slight positive allometry.",
        is_experimental_girth=True,
    ),
    "carp": SpeciesBiomassParameters(
        species_id="carp",
        scientific_name="Cyprinus carpio",
        a_coefficient=0.0153,
        b_exponent=3.01,
        c_shape_divisor=28.5,
        citation="Froese (2006), FishBase consensus for Cyprinus carpio; Balon (1995). Deep-bodied robust cyprinid.",
        length_convention="Total Length (TL, cm)",
        weight_units="grams (g), converted to kg",
        population_context="Common carp reared in semi-intensive earthen polyculture ponds across temperate/subtropical waters.",
        limitations="Body depth varies substantially across domesticated selective breeds (e.g. mirror carp vs scaly carp).",
        is_experimental_girth=True,
    ),
    "milkfish": SpeciesBiomassParameters(
        species_id="milkfish",
        scientific_name="Chanos chanos",
        a_coefficient=0.0091,
        b_exponent=3.04,
        c_shape_divisor=34.0,
        citation="Froese (2006), FishBase consensus for Chanos chanos; Bagarinao (1991) SEAFDEC. Streamlined fusiform.",
        length_convention="Fork Length / Total Length (FL/TL, cm)",
        weight_units="grams (g), converted to kg",
        population_context="Brackishwater and coastal marine net cages / pens in Southeast Asia (Philippines, Indonesia).",
        limitations="Streamlined fusiform body; fork length should be converted to total length if caudal fin lobes are uneven.",
        is_experimental_girth=True,
    ),
    "pangasius": SpeciesBiomassParameters(
        species_id="pangasius",
        scientific_name="Pangasianodon hypophthalmus",
        a_coefficient=0.0085,
        b_exponent=3.03,
        c_shape_divisor=35.0,
        citation="Froese (2006), FishBase consensus for Pangasianodon hypophthalmus; Roberts & Vidthayanon (1991).",
        length_convention="Standard / Total Length (SL/TL, cm)",
        weight_units="grams (g), converted to kg",
        population_context="Striped catfish in high-density commercial pond and cage culture in the Mekong Delta basin.",
        limitations="Commercial high-protein feed induces deeper abdominal belly curvature than wild riverine baselines.",
        is_experimental_girth=True,
    ),
}

# Fallback generic cyprinid parameters
DEFAULT_FALLBACK_PARAMS = SpeciesBiomassParameters(
    species_id="generic_finfish",
    scientific_name="Teleostei indet.",
    a_coefficient=0.0140,
    b_exponent=3.00,
    c_shape_divisor=32.0,
    citation="Generalized fusiform teleost baseline (Froese 2006 cube law).",
    length_convention="Total Length (TL, cm)",
    weight_units="grams (g), converted to kg",
    population_context="Generic aquaculture teleost reference.",
    limitations="Cube-law approximation without species-specific calibration.",
    is_experimental_girth=True,
)


@dataclass(frozen=True)
class BiomassEstimate:
    species: str
    estimated_biomass_kg: float
    method: str
    is_experimental: bool = False
    citation: str = ""
    notes: str = ""
    length_convention: str = "Total Length (TL, cm)"
    population_context: str = ""
    limitations: str = ""


def estimate_biomass(
    species: str,
    length_cm: float,
    girth_cm: float | None = None,
) -> BiomassEstimate:
    """Estimate individual fish biomass from physical metric measurements.

    Distinguishes two methods:
    1. Length-Only: W = a * L^b (Empirically verified from FishBase/FAO peer-reviewed literature).
    2. Length-and-Girth: W = (L * G^2) / C_shape (Experimental volumetric approximation;
       requires farm-specific calibration across growth stages).
    """
    if length_cm <= 0:
        raise ValueError("Length must be strictly positive (cm).")

    key = species.lower().strip()
    params = SPECIES_BIOMASS_REGISTRY.get(key, DEFAULT_FALLBACK_PARAMS)

    if girth_cm is not None and girth_cm > 0:
        # Experimental length-girth formula: W(g) = (L * G^2) / C_shape
        biomass_g = (length_cm * (girth_cm ** 2)) / params.c_shape_divisor
        method = "experimental_length_girth_morphology"
        is_experimental = True
        notes = (
            f"Experimental elliptical volume approximation using shape factor C={params.c_shape_divisor}. "
            "Girth-based estimates vary with feeding satiation and gonad maturity."
        )
    else:
        # Verified length-weight power relation: W(g) = a * L^b
        biomass_g = params.a_coefficient * (length_cm ** params.b_exponent)
        method = "verified_fishbase_length_weight"
        is_experimental = False
        notes = (
            f"Calculated from standard fisheries power equation W = a * L^b with a={params.a_coefficient}, "
            f"b={params.b_exponent}. Total length measured in centimeters."
        )

    biomass_kg = float(np.clip(biomass_g / 1000.0, 0.0, None))
    return BiomassEstimate(
        species=species,
        estimated_biomass_kg=round(biomass_kg, 4),
        method=method,
        is_experimental=is_experimental,
        citation=params.citation,
        notes=notes,
        length_convention=params.length_convention,
        population_context=params.population_context,
        limitations=params.limitations,
    )


@dataclass(frozen=True)
class BatchBiomassEstimate:
    species: str
    individual_biomass_kg: float
    fish_count: int
    survival_rate_percent: float
    effective_fish_count: int
    total_biomass_kg: float
    estimated_daily_feed_kg: float
    feed_rate_percent: float
    method: str = "batch_population_aggregation"
    is_experimental: bool = False


def estimate_batch_biomass(
    species: str,
    individual_weight_kg: float,
    fish_count: int,
    survival_rate_percent: float = 100.0,
    daily_feed_rate_percent: float = 2.5,
    method: str = "batch_population_aggregation",
) -> BatchBiomassEstimate:
    """Calculate total batch biomass and daily feed estimate for a pond population.

    Labels results strictly as population-level mathematical aggregations.
    """
    if individual_weight_kg < 0:
        raise ValueError("Individual weight cannot be negative")
    if fish_count < 0:
        raise ValueError("Fish count cannot be negative")
    if not (0.0 <= survival_rate_percent <= 100.0):
        raise ValueError("Survival rate must be between 0.0 and 100.0 percent")
    if daily_feed_rate_percent < 0:
        raise ValueError("Daily feed rate cannot be negative")

    effective_count = int(round(fish_count * (survival_rate_percent / 100.0)))
    total_biomass_kg = round(individual_weight_kg * effective_count, 3)
    daily_feed_kg = round(total_biomass_kg * (daily_feed_rate_percent / 100.0), 3)

    return BatchBiomassEstimate(
        species=species,
        individual_biomass_kg=round(individual_weight_kg, 4),
        fish_count=fish_count,
        survival_rate_percent=survival_rate_percent,
        effective_fish_count=effective_count,
        total_biomass_kg=total_biomass_kg,
        estimated_daily_feed_kg=daily_feed_kg,
        feed_rate_percent=daily_feed_rate_percent,
        method=method,
        is_experimental=False,
    )
