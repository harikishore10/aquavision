"""Aquaculture Species Intelligence Knowledge Base.

Contains structured biological profiles, identifying morphological traits,
habitat, diet, and farming-context-dependent husbandry information for cultured fish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WaterQualityContext:
    parameter: str
    optimal_range: str
    tolerance_range: str
    unit: str
    notes: str


@dataclass(frozen=True)
class SpeciesProfile:
    species_id: str
    common_name: str
    scientific_name: str
    family: str
    identifying_characteristics: list[str]
    habitat: str
    diet: str
    trophic_level: str
    farming_systems: list[str]
    growth_cycle_months: str
    target_harvest_weight: str
    typical_fcr: str
    water_quality_notes: str
    water_quality_parameters: list[WaterQualityContext]
    aquaculture_management: list[str]
    references: list[str]


SPECIES_REGISTRY: dict[str, SpeciesProfile] = {
    "tilapia": SpeciesProfile(
        species_id="tilapia",
        common_name="Nile Tilapia",
        scientific_name="Oreochromis niloticus (Linnaeus, 1758)",
        family="Cichlidae",
        identifying_characteristics=[
            "Deep, laterally compressed body with greyish-silver to olivaceous coloration.",
            "Distinct dark vertical bars across the caudal (tail) fin, dorsal fin, and flanks.",
            "Margin of dorsal and caudal fins often pinkish or reddish in mature specimens.",
            "Interrupted lateral line and continuous dorsal fin with 16–18 spines and 12–13 soft rays.",
            "Terminal mouth with small bifid teeth suited for grazing and filter-feeding.",
        ],
        habitat=(
            "Fresh to brackish tropical and subtropical inland waters, slow-moving rivers, "
            "lakes, reservoirs, and agricultural drainage canals. Native to the Nile basin "
            "and Central/West Africa; widely naturalized and cultivated across global tropical zones."
        ),
        diet=(
            "Omnivorous microphage and opportunistic grazer. In the wild, feeds on phytoplankton, "
            "filamentous algae, periphyton, benthic detritus, and aquatic macrophytes. In commercial "
            "culture, fed floating or sinking extruded pellets (28–34% crude protein for grow-out)."
        ),
        trophic_level="2.0 – Herbivore / Detritivore",
        farming_systems=[
            "Earthen ponds (semi-intensive greenwater)",
            "Floating net cages in lakes and reservoirs",
            "Recirculating Aquaculture Systems (RAS)",
            "Biofloc Technology (BFT) tanks",
        ],
        growth_cycle_months="5 – 8 months from fingerling to market weight",
        target_harvest_weight="450 g – 800 g (whole market) / 800 g – 1.2 kg (fillet processing)",
        typical_fcr="1.2 – 1.7 (pelleted feed)",
        water_quality_notes=(
            "Water-quality thresholds are strongly dependent on the farming environment. "
            "For example, high-density biofloc systems require continuous oxygenation (>5.0 mg/L) "
            "and active alkalinity buffering (100–150 mg/L CaCO3), whereas extensive ponds "
            "tolerate wider diurnal fluctuations."
        ),
        water_quality_parameters=[
            WaterQualityContext(
                parameter="Water Temperature",
                optimal_range="27 – 31",
                tolerance_range="20 – 35 (Critical lower: 11–12)",
                unit="°C",
                notes="Feeding ceases below 18°C; prolonged exposure below 12°C is lethal.",
            ),
            WaterQualityContext(
                parameter="Dissolved Oxygen (DO)",
                optimal_range="> 4.0",
                tolerance_range="1.0 – 8.0+",
                unit="mg/L",
                notes="Tolerates severe short-term hypoxia by surface skimming; prolonged low DO halts growth.",
            ),
            WaterQualityContext(
                parameter="pH",
                optimal_range="6.5 – 8.5",
                tolerance_range="5.0 – 9.5",
                unit="pH units",
                notes="Values outside 6.0–9.0 induce osmotic stress and inhibit feeding.",
            ),
            WaterQualityContext(
                parameter="Unionized Ammonia (NH3)",
                optimal_range="< 0.05",
                tolerance_range="< 0.10",
                unit="mg/L NH3-N",
                notes="High pH increases the toxic unionized fraction rapidly.",
            ),
            WaterQualityContext(
                parameter="Salinity",
                optimal_range="0 – 5",
                tolerance_range="0 – 20",
                unit="ppt (g/L)",
                notes="Euryhaline species; slow acclimation enables brackish culture.",
            ),
        ],
        aquaculture_management=[
            "All-male monosex culture (via hormonal sex-reversal or YY technology) is standard to prevent uncontrolled pond reproduction and stunting.",
            "Frequent grading during nursery stages ensures uniform biomass and lowers feed competition.",
            "Stop feeding 24–48 hours prior to harvest to purge gastrointestinal tract.",
        ],
        references=[
            "FAO Fisheries and Aquaculture Department (2020). Cultured Aquatic Species Fact Sheets: Oreochromis niloticus.",
            "Boyd, C. E. (1998). Water Quality for Pond Aquaculture. Auburn University, Research and Development Series.",
            "WorldFish (2018). Genetic Improvement of Farmed Tilapia (GIFT): Project summary and dissemination guide.",
        ],
    ),
    "catfish": SpeciesProfile(
        species_id="catfish",
        common_name="African Sharptooth Catfish",
        scientific_name="Clarias gariepinus (Burchell, 1822)",
        family="Clariidae",
        identifying_characteristics=[
            "Elongated, cylindrical scaleless body with tough, mucus-covered dark grey/black skin.",
            "Broad, heavily flattened and ossified bony head with small lateral eyes.",
            "Four pairs of sensory barbels around the mouth (nasal, maxillary, inner and outer mandibular).",
            "Continuous, long dorsal fin extending across majority of trunk without hard dorsal spine.",
            "Specialized suprabranchial organ (arborescent organ) enabling atmospheric air-breathing.",
        ],
        habitat=(
            "Freshwater floodplain wetlands, muddy bottom rivers, swamps, and seasonal pans. "
            "Native across Africa and the Middle East; extensively cultivated across Africa, Europe, "
            "and Southeast Asia due to extreme tolerance for hypoxia and turbidity."
        ),
        diet=(
            "Opportunistic carnivore and scavenger. Wild prey includes aquatic insects, mollusks, "
            "smaller fish, amphibians, and organic refuse. Farmed stock requires high-protein feed "
            "(38–42% crude protein for fingerlings, 32–36% for grow-out) in sinking or slow-floating form."
        ),
        trophic_level="3.2 – Secondary / Tertiary Consumer",
        farming_systems=[
            "Intensive concrete / plastic circular tanks",
            "Earthen grow-out ponds",
            "Recirculating Aquaculture Systems (RAS) with biofiltration",
            "Flow-through earthen raceways",
        ],
        growth_cycle_months="4 – 6 months from fingerling to market weight",
        target_harvest_weight="800 g – 1.5 kg (local live market and smoking/processing)",
        typical_fcr="1.0 – 1.4 (intensive feed management)",
        water_quality_notes=(
            "Air-breathing capacity allows Clarias to survive in near-anoxic water, but commercial "
            "growth rate and feed efficiency collapse if water quality is poor. High stocking "
            "density requires diligent management of total ammonia nitrogen and nitrite."
        ),
        water_quality_parameters=[
            WaterQualityContext(
                parameter="Water Temperature",
                optimal_range="26 – 30",
                tolerance_range="18 – 35",
                unit="°C",
                notes="Growth declines significantly below 22°C; thermal shock above 36°C.",
            ),
            WaterQualityContext(
                parameter="Dissolved Oxygen (DO)",
                optimal_range="> 3.0",
                tolerance_range="0.5 – 8.0",
                unit="mg/L",
                notes="Survives low DO via air-breathing organ, but optimal feed ingestion requires >3.0 mg/L.",
            ),
            WaterQualityContext(
                parameter="pH",
                optimal_range="6.5 – 8.0",
                tolerance_range="5.5 – 9.0",
                unit="pH units",
                notes="Sensitive to rapid swings; acid soils can depress pond water pH below safe limits.",
            ),
            WaterQualityContext(
                parameter="Unionized Ammonia (NH3)",
                optimal_range="< 0.05",
                tolerance_range="< 0.20",
                unit="mg/L NH3-N",
                notes="More tolerant to TAN than scaled fish, but elevated nitrite triggers brown-blood disease.",
            ),
            WaterQualityContext(
                parameter="Salinity",
                optimal_range="0 – 2",
                tolerance_range="0 – 8",
                unit="ppt (g/L)",
                notes="Strictly freshwater adapted; salinity > 10 ppt is lethal.",
            ),
        ],
        aquaculture_management=[
            "Strict size grading every 2 to 3 weeks is essential to suppress severe cannibalism.",
            "Cover high-density tanks with netting as catfish readily jump during night hours or feeding frenzy.",
            "Maintain consistent feeding schedule to prevent aggressive fin-nipping.",
        ],
        references=[
            "Hecht, T., Uys, W., & Britz, P. J. (1988). The culture of Sharptooth Catfish, Clarias gariepinus in Southern Africa. South African National Scientific Programmes Report No. 153.",
            "FAO Fisheries Technical Paper (1996). A manual on the culture of the African catfish (Clarias gariepinus). Rome, FAO.",
        ],
    ),
    "carp": SpeciesProfile(
        species_id="carp",
        common_name="Common Carp",
        scientific_name="Cyprinus carpio (Linnaeus, 1758)",
        family="Cyprinidae",
        identifying_characteristics=[
            "Robust, deep-bodied fish with arched dorsal profile and slightly compressed flanks.",
            "Thick lips with two pairs of maxillary barbels (the upper pair shorter, the lower at corner of mouth).",
            "Long dorsal fin with 17–22 soft rays preceded by a stout, serrated spine.",
            "Golden-yellow to brownish-green coloration on back, fading to yellowish-white on belly.",
            "Large, thick cycloid scales (in scaled variety) or sparse large scales (mirror carp strain).",
        ],
        habitat=(
            "Lowland rivers, backwaters, floodplain lakes, and warm eutrophic ponds with gentle "
            "water flow and abundant submerged vegetation. Eurythermal and native across Eurasia; "
            "cultivated on every continent except Antarctica."
        ),
        diet=(
            "Benthic omnivore and detritivore. Grubbing behavior in soft substrate targets chironomid "
            "larvae, tubificids, small mollusks, benthic microcrustaceans, seeds, and plant detritus. "
            "In culture, responds efficiently to cereal grains and balanced pelleted feeds (28–32% CP)."
        ),
        trophic_level="2.4 – Primary / Secondary Consumer",
        farming_systems=[
            "Traditional composite polyculture earthen ponds",
            "Integrated rice-fish farming systems",
            "Semi-intensive fertilized nursery and grow-out ponds",
            "Floating cages in eutrophic reservoirs",
        ],
        growth_cycle_months="8 – 14 months (depending on climate zone: faster in tropics)",
        target_harvest_weight="1.0 kg – 2.5 kg",
        typical_fcr="1.5 – 2.2 (supplementary feed in fertilized ponds)",
        water_quality_notes=(
            "Extremely eurythermal. Survives freezing temperatures under surface ice as well "
            "as tropical warmth. Benthic foraging stirs up mud, increasing water turbidity, "
            "which should be managed in multi-species systems."
        ),
        water_quality_parameters=[
            WaterQualityContext(
                parameter="Water Temperature",
                optimal_range="23 – 28",
                tolerance_range="3 – 35",
                unit="°C",
                notes="Feeding and growth peak between 24–28°C; enters torpor below 5°C.",
            ),
            WaterQualityContext(
                parameter="Dissolved Oxygen (DO)",
                optimal_range="> 5.0",
                tolerance_range="1.5 – 9.0",
                unit="mg/L",
                notes="Tolerates hypoxia well down to 2.0 mg/L; prolonged hypoxia stunts growth.",
            ),
            WaterQualityContext(
                parameter="pH",
                optimal_range="6.8 – 8.2",
                tolerance_range="5.5 – 9.5",
                unit="pH units",
                notes="Pond liming recommended if bottom soil or water pH drops below 6.5.",
            ),
            WaterQualityContext(
                parameter="Unionized Ammonia (NH3)",
                optimal_range="< 0.03",
                tolerance_range="< 0.08",
                unit="mg/L NH3-N",
                notes="Chronic high ammonia damages gill lamellae and suppresses feed intake.",
            ),
            WaterQualityContext(
                parameter="Salinity",
                optimal_range="0 – 1",
                tolerance_range="0 – 6",
                unit="ppt (g/L)",
                notes="Freshwater fish; tolerates low-salinity brackish water up to 5 ppt.",
            ),
        ],
        aquaculture_management=[
            "Ideal foundation species for polyculture alongside surface-feeders (Silver Carp) and column-feeders (Rohu, Grass Carp).",
            "Organic manuring and inorganic fertilization promote benthic natural food organisms.",
            "Provide artificial spawning substrates (kakabans) for hatchery propagation.",
        ],
        references=[
            "FAO Cultured Aquatic Species Fact Sheets (2018). Cyprinus carpio. FAO Fisheries Division.",
            "Jhingran, V. G., & Pullin, R. S. V. (1985). A Hatchery Manual for the Common, Chinese and Indian Major Carps. ICLARM Studies and Reviews 11.",
        ],
    ),
    "milkfish": SpeciesProfile(
        species_id="milkfish",
        common_name="Milkfish (Bangus)",
        scientific_name="Chanos chanos (Forsskål, 1775)",
        family="Chanidae",
        identifying_characteristics=[
            "Elongated, torpedo-shaped streamlined body with brilliant silvery flanks and greenish-blue back.",
            "Large, deeply forked homocercal caudal fin with distinctive scale flaps at base.",
            "Small, toothless terminal mouth without sensory barbels.",
            "Smooth, firmly attached cycloid scales covering the body.",
            "Adipose tissue covers the eye; single dorsal fin situated near mid-body.",
        ],
        habitat=(
            "Marine coastal waters, coral reef lagoons, estuaries, mangrove bays, and tidal flats "
            "throughout the tropical Indo-Pacific. Euryhaline species that enters rivers and lakes as "
            "fingerlings and returns to the sea to mature and spawn."
        ),
        diet=(
            "Herbivorous and microphagous grazer. Feeds on 'lablab' (benthic biological mat composed "
            "of blue-green algae, diatoms, protozoans, and nematodes) and filamentous green algae ('lumut'). "
            "In intensive pen and cage culture, accepts extruded commercial floating pellets (24–30% CP)."
        ),
        trophic_level="2.0 – Herbivore / Grazer",
        farming_systems=[
            "Traditional shallow brackishwater ponds ('lablab' culture model)",
            "Deep-water marine net pens and cages in coastal bays",
            "Freshwater pens in large inland lakes (e.g., Laguna de Bay)",
            "Extensive mangrove intertidal ponds",
        ],
        growth_cycle_months="4 – 6 months from fingerling to market size",
        target_harvest_weight="300 g – 600 g (traditional table fish) / 600 g – 1.0 kg (deboned fillet)",
        typical_fcr="1.3 – 1.8 (commercial feed) / ~0.8 – 1.2 in natural lablab ponds",
        water_quality_notes=(
            "Highly euryhaline: adapts seamlessly from 0 ppt to 40+ ppt salinity with gradual "
            "acclimatization. Extremely vulnerable to cold temperatures below 16°C and sudden "
            "nighttime oxygen depletion in shallow ponds."
        ),
        water_quality_parameters=[
            WaterQualityContext(
                parameter="Water Temperature",
                optimal_range="26 – 32",
                tolerance_range="15 – 36",
                unit="°C",
                notes="Tropical stenotherm; sudden drops below 16°C induce shock and mass mortality.",
            ),
            WaterQualityContext(
                parameter="Dissolved Oxygen (DO)",
                optimal_range="> 4.0",
                tolerance_range="1.5 – 8.0+",
                unit="mg/L",
                notes="Prone to morning piping; paddlewheel aerators crucial during dawn.",
            ),
            WaterQualityContext(
                parameter="pH",
                optimal_range="7.5 – 8.5",
                tolerance_range="6.5 – 9.0",
                unit="pH units",
                notes="Tolerates marine alkalinity well; acidic soils require agricultural lime.",
            ),
            WaterQualityContext(
                parameter="Unionized Ammonia (NH3)",
                optimal_range="< 0.02",
                tolerance_range="< 0.05",
                unit="mg/L NH3-N",
                notes="Elevated unionized ammonia in alkaline water impairs respiratory epithelium.",
            ),
            WaterQualityContext(
                parameter="Salinity",
                optimal_range="10 – 32",
                tolerance_range="0 – 45",
                unit="ppt (g/L)",
                notes="Truly euryhaline; can be farmed in freshwater lakes, brackish ponds, or marine cages.",
            ),
        ],
        aquaculture_management=[
            "Dry and prepare pond bottom carefully to cultivate rich 'lablab' mats prior to fingerling stocking.",
            "Avoid handling during overcast days or midday heat; milkfish lose scales easily, inviting secondary infection.",
            "Implement ice-slurry immersion at harvest for rapid cold shock and superior meat freshness.",
        ],
        references=[
            "Southeast Asian Fisheries Development Center (SEAFDEC). Aquaculture Department (2001). Biology and Culture of Penaeid Shrimps and Milkfish. Iloilo, Philippines.",
            "FAO Cultured Aquatic Species Fact Sheets (2012). Chanos chanos. FAO Fisheries Department.",
        ],
    ),
    "pangasius": SpeciesProfile(
        species_id="pangasius",
        common_name="Striped Catfish (Pangasius / Tra)",
        scientific_name="Pangasianodon hypophthalmus (Sauvage, 1878)",
        family="Pangasiidae",
        identifying_characteristics=[
            "Long, laterally compressed scaleless body with dark grey/blue dorsal side and silvery iridescent sheen.",
            "Small, broad head with large eyes situated somewhat low on the skull.",
            "Short barbels (maxillary barbels shorter than head length; mental barbels very short).",
            "Small adipose fin present on the back between dorsal and caudal fins.",
            "Juveniles exhibit two distinct dark horizontal lateral stripes that fade upon maturity.",
        ],
        habitat=(
            "Large freshwater river channels, deep pools, and connected floodplain wetlands in the "
            "Mekong and Chao Phraya basins of Southeast Asia. Benthopelagic freshwater fish, "
            "poorly tolerant of brackish water."
        ),
        diet=(
            "Omnivore and opportunistic bottom-column feeder. Consumes algae, aquatic plants, "
            "zooplankton, insects, mollusks, and detritus. Under commercial culture in the Mekong Delta, "
            "fed extruded floating pellets (26–30% CP) with controlled fat content to avoid yellow fillet pigmentation."
        ),
        trophic_level="2.5 – Omnivore",
        farming_systems=[
            "Deep intensive earthen ponds (3.5 – 4.5 m depth, Mekong Delta model)",
            "Floating net cages in large river channels",
            "Fenced river pens and raceways",
        ],
        growth_cycle_months="6 – 8 months from fingerling to market size",
        target_harvest_weight="800 g – 1.3 kg (standard export processing size)",
        typical_fcr="1.5 – 1.7 (floating commercial extruded feed)",
        water_quality_notes=(
            "Possesses a vascularized swim bladder adapted for facultative air breathing, "
            "allowing it to endure lower oxygen levels in heavily stocked ponds. Strict freshwater "
            "species: salinity above 8 ppt severely retards development."
        ),
        water_quality_parameters=[
            WaterQualityContext(
                parameter="Water Temperature",
                optimal_range="28 – 31",
                tolerance_range="22 – 35",
                unit="°C",
                notes="Optimal metabolism at warm tropical range; feeding reduces noticeably below 24°C.",
            ),
            WaterQualityContext(
                parameter="Dissolved Oxygen (DO)",
                optimal_range="> 2.5",
                tolerance_range="0.8 – 7.0",
                unit="mg/L",
                notes="Supplementary air-breathing via swim bladder mitigates acute hypoxia.",
            ),
            WaterQualityContext(
                parameter="pH",
                optimal_range="6.5 – 7.8",
                tolerance_range="5.5 – 8.5",
                unit="pH units",
                notes="Prefers neutral to slightly alkaline water; acid water compromises skin mucus barrier.",
            ),
            WaterQualityContext(
                parameter="Unionized Ammonia (NH3)",
                optimal_range="< 0.05",
                tolerance_range="< 0.15",
                unit="mg/L NH3-N",
                notes="High stocking density requires rigorous pond water exchanges to flush ammonia.",
            ),
            WaterQualityContext(
                parameter="Salinity",
                optimal_range="0 – 1",
                tolerance_range="0 – 6",
                unit="ppt (g/L)",
                notes="Freshwater obligate; prolonged salinity > 6 ppt causes osmotic breakdown.",
            ),
        ],
        aquaculture_management=[
            "Perform regular tidal water exchange or pumping (20–40% pond volume daily in late grow-out) to maintain hygiene.",
            "Use floating extruded feed to observe feeding vigor and minimize feed waste decomposition on deep pond bottoms.",
            "Ensure low soybean meal pigments and xanthophylls in feed formulation to guarantee white meat fillet grade.",
        ],
        references=[
            "Phan, L. T., et al. (2009). Current status of Pangasius aquaculture in Vietnam. Aquaculture, 295(3-4), 252-260.",
            "FAO Fisheries and Aquaculture Department (2010). Cultured Aquatic Species Fact Sheets: Pangasianodon hypophthalmus.",
            "De Silva, S. S., & Phuong, N. T. (2011). Striped catfish farming in the Mekong Delta, Vietnam: a tumultuous path to a global success. Reviews in Aquaculture, 3(2), 45-73.",
        ],
    ),
}


def get_species_profile(species_name: str) -> SpeciesProfile | None:
    """Retrieve structured species intelligence profile by common or scientific name."""
    if not species_name:
        return None
    key = species_name.lower().strip()
    if key in SPECIES_REGISTRY:
        return SPECIES_REGISTRY[key]
    for profile in SPECIES_REGISTRY.values():
        if key in profile.common_name.lower() or key in profile.scientific_name.lower():
            return profile
    return None


def get_all_species_profiles() -> dict[str, SpeciesProfile]:
    """Retrieve full registry of species intelligence profiles."""
    return SPECIES_REGISTRY


def search_species_profiles(query: str) -> list[SpeciesProfile]:
    """Search species intelligence registry across names, characteristics, systems, and diet."""
    if not query or not query.strip():
        return list(SPECIES_REGISTRY.values())
    q = query.lower().strip()
    matches: list[SpeciesProfile] = []
    for profile in SPECIES_REGISTRY.values():
        haystack = " ".join(
            [
                profile.common_name.lower(),
                profile.scientific_name.lower(),
                profile.family.lower(),
                profile.habitat.lower(),
                profile.diet.lower(),
                " ".join(profile.identifying_characteristics).lower(),
                " ".join(profile.farming_systems).lower(),
            ]
        )
        if q in haystack:
            matches.append(profile)
    return matches
