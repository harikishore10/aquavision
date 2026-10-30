from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.api_client import FarmApiClient
from fishfarm_ai.biomass import estimate_batch_biomass
from fishfarm_ai.fishbase_service import FISHBASE_ATTRIBUTION, get_fishbase_service
from fishfarm_ai.gemini_service import GeminiVisionService
from fishfarm_ai.services import (
    BiomassEstimationService,
    FishPredictionService,
    SpeciesIdentificationService,
)
from fishfarm_ai.species_intelligence import (
    get_all_species_profiles,
    get_species_profile,
    search_species_profiles,
)

st.set_page_config(
    page_title="AquaVision: Aquaculture AI Hub",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@600;700&display=swap');

      :root {
        --bg: #061826;
        --bg-2: #0d2b3f;
        --panel: rgba(255, 255, 255, 0.07);
        --panel-border: rgba(255, 255, 255, 0.12);
        --text: #ecf7fb;
        --muted: #b4ccd8;
        --accent: #5ef0c2;
        --accent-2: #7cc9ff;
        --accent-3: #ffd166;
        --warning: #ffb703;
      }

      .stApp {
        background:
          radial-gradient(circle at top left, rgba(94, 240, 194, 0.15), transparent 30%),
          radial-gradient(circle at top right, rgba(124, 201, 255, 0.12), transparent 32%),
          linear-gradient(145deg, var(--bg) 0%, var(--bg-2) 100%);
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
      }

      .hero {
        padding: 2.2rem 2.2rem 1.5rem 2.2rem;
        border: 1px solid var(--panel-border);
        border-radius: 28px;
        background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
        backdrop-filter: blur(12px);
        box-shadow: 0 24px 80px rgba(0,0,0,0.22);
        margin-bottom: 1.25rem;
      }

      .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(94, 240, 194, 0.14);
        color: var(--accent);
        font-size: 0.8rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 1rem;
      }

      .hero h1 {
        font-family: 'Source Serif 4', serif;
        font-size: clamp(2.2rem, 3.8vw, 4.0rem);
        line-height: 1.05;
        margin: 0 0 0.6rem 0;
        color: var(--text);
      }

      .hero p {
        font-size: 1.02rem;
        line-height: 1.7;
        color: var(--muted);
        max-width: 62rem;
      }

      .card {
        padding: 1.1rem 1.15rem;
        border-radius: 22px;
        border: 1px solid var(--panel-border);
        background: var(--panel);
        backdrop-filter: blur(8px);
        margin-bottom: 0.8rem;
      }

      .card h3 {
        margin: 0 0 0.35rem 0;
        color: var(--text);
      }

      .card p, .card li {
        color: var(--muted);
      }

      .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1rem;
      }

      .pill {
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        color: var(--text);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.82rem;
      }

      .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.2rem 0 0.8rem 0;
        color: var(--text);
      }

      .small-note {
        color: var(--muted);
        font-size: 0.92rem;
      }

      div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 0.9rem 1rem;
      }

      div[data-testid="stTabs"] {
        background: rgba(255,255,255,0.03);
        border-radius: 24px;
        padding: 0.35rem;
        border: 1px solid rgba(255,255,255,0.09);
      }

      div[data-testid="stForm"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 24px;
        padding: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_services():
    return (
        SpeciesIdentificationService(),
        BiomassEstimationService(),
        FishPredictionService(),
    )

species_service, biomass_service, combined_service = get_services()

species_model_is_ready = species_service.is_ready()
biomass_model_is_ready = biomass_service.is_ready()

if "backend_token" not in st.session_state:
    st.session_state.backend_token = ""
if "backend_signed_in" not in st.session_state:
    st.session_state.backend_signed_in = False

st.sidebar.markdown("### AquaVision Control")
execution_mode = st.sidebar.radio("Run predictions using", ["Local services", "Backend API"], index=0)
default_backend_url = os.getenv("AQUAVISION_BACKEND_URL", "")
if not default_backend_url and hasattr(st, "secrets") and "AQUAVISION_BACKEND_URL" in st.secrets:
    default_backend_url = str(st.secrets["AQUAVISION_BACKEND_URL"]).strip()
if not default_backend_url:
    default_backend_url = "http://127.0.0.1:8000"
backend_url = st.sidebar.text_input("Backend URL", value=default_backend_url)
username = st.sidebar.text_input("Username", value="admin")
password = st.sidebar.text_input("Password", value="", type="password")

api_client: FarmApiClient | None = None
if execution_mode == "Backend API" and backend_url:
    api_client = FarmApiClient(base_url=backend_url, token=st.session_state.backend_token or "dev-token")
    if st.sidebar.button("Sign in to backend"):
        if not username or not password:
            st.sidebar.error("Enter both username and password.")
        else:
            try:
                token = api_client.login(username, password)
                st.session_state.backend_token = token
                st.session_state.backend_signed_in = True
                api_client = FarmApiClient(base_url=backend_url, token=token)
                st.sidebar.success("Signed in successfully.")
            except Exception as exc:
                st.session_state.backend_signed_in = False
                st.sidebar.error(f"Backend sign-in failed: {exc}")

    if st.session_state.backend_signed_in:
        st.sidebar.success("Backend session active")
        if st.sidebar.button("Sign out"):
            st.session_state.backend_token = ""
            st.session_state.backend_signed_in = False
            st.sidebar.info("Signed out.")

if execution_mode == "Backend API" and not st.session_state.backend_signed_in:
    st.sidebar.warning("Sign in to the backend to use API mode. Local services remain available.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model System Status")
st.sidebar.markdown(
    f"**Species Classifier:** {'🟢 Online (EfficientNet-B0 Transfer Bundle)' if species_model_is_ready else '🔴 Offline'}"
)
st.sidebar.markdown(
    f"**Biomass Regressor:** {'🟢 Online (GradientBoosting Bundle)' if biomass_model_is_ready else '🟡 Formula Fallback'}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Intelligent Species Assessment")
gemini_service = GeminiVisionService()
detected_key = GeminiVisionService.get_api_key()
gemini_key_input = st.sidebar.text_input(
    "Assessment API Key (Optional)",
    value=detected_key or "",
    type="password",
    help="Leave blank to use configured environment variable. Never saved in code.",
)
active_gemini_key = gemini_key_input.strip() or detected_key

if active_gemini_key:
    st.sidebar.caption("🟢 Assessment API Key active")
else:
    st.sidebar.caption("⚪ No API key configured (Optional assessment)")


def predict_species(image: Image.Image):
    if api_client is not None and st.session_state.backend_signed_in:
        try:
            return api_client.predict_species(image)
        except Exception as exc:
            st.warning(f"Backend species prediction failed, falling back to local service: {exc}")
    return species_service.predict(image)


def predict_biomass(species: str, length_cm: float, girth_cm: float | None):
    if api_client is not None and st.session_state.backend_signed_in:
        try:
            return api_client.predict_biomass(species, length_cm, girth_cm)
        except Exception as exc:
            st.warning(f"Backend biomass prediction failed, falling back to local service: {exc}")
    return biomass_service.predict(species, length_cm, girth_cm)


def predict_combined(image: Image.Image, length_cm: float, girth_cm: float | None, species_override: str | None = None):
    if api_client is not None and st.session_state.backend_signed_in:
        try:
            return api_client.predict_combined(image, length_cm, girth_cm, species=species_override)
        except Exception as exc:
            st.warning(f"Backend combined prediction failed, falling back to local service: {exc}")
    return combined_service.predict(image, length_cm, girth_cm, species_override=species_override)


st.markdown(
    """
    <div class="hero">
      <span class="eyebrow">AquaVision • Aquaculture Intelligence</span>
      <h1>AI-Based Fish Species Identification, Biomass Estimation & Species Intelligence</h1>
      <p>
        Unified aquaculture management hub providing real image-based species classification,
        biological intelligence profiles, and field-verified biomass calculators.
      </p>
      <div class="pill-row">
        <span class="pill">🔍 Real Image Classifier (EfficientNet-B0)</span>
        <span class="pill">🐟 Species Intelligence Profiles</span>
        <span class="pill">⚖️ Biomass & Batch Estimator</span>
        <span class="pill">🛡️ Strict Scientific Integrity</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Classifier Status", "🟢 Online" if species_model_is_ready else "🔴 Offline")
    with col2:
        st.metric("Species Explorer", "Knowledge Base", "Searchable Profiles & Biology")
    with col3:
        st.metric("Biomass Engine", "🟢 Trained Regressor" if biomass_model_is_ready else "🟡 Formula Fallback")
    with col4:
        st.metric("System Mode", execution_mode)

st.markdown("<div class='section-title'>Workflows</div>", unsafe_allow_html=True)
mode = st.radio(
    "Choose a workflow",
    [
        "Species Identification (Image)",
        "Combined Farm Workflow",
        "Biomass & Batch Estimation",
        "Species Intelligence Explorer",
        "Documentation & Model Evaluation",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# -----------------------------------------------------------------------------
# WORKFLOW 1: Species Identification (Image)
# -----------------------------------------------------------------------------
if mode == "Species Identification (Image)":
    st.markdown(
        """
        <div class="card">
          <h3>Computer Vision Fish Species Classifier</h3>
          <p>
            Real image-based fish species identification trained on verified GBIF/iNaturalist research observations.
            Utilizes an ImageNet-pretrained EfficientNet-B0 backbone with 1,280-dimensional deep semantic feature extraction
            and group-calibrated probability estimation across supported aquaculture finfish.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not species_model_is_ready:
        st.error("🛑 **Model Offline:** No trained classifier found. Run `python train.py` to train model bundles.")
    else:
        classes_str = ", ".join(c.capitalize() for c in sorted(species_service.species_model.class_names))
        st.info(
            "🐟 **Comprehensive Fish Species Platform:** AquaVision combines deep computer vision for direct image classification, "
            "an interactive Species Intelligence Knowledge Base for biological exploration, and cloud AI consultation for broader aquatic organism identification."
        )
        with st.expander("ℹ️ Local Classifier Specifications & Trained Target Species", expanded=False):
            st.markdown(
                f"**Target Aquaculture Finfish Trained in Local Model:** **{classes_str}**\n\n"
                "*Scope & Boundary:* The on-device EfficientNet-B0 convolutional network is trained specifically on verified observation records for these five primary production species. "
                "It does not recognize every wild or aquarium fish species. For specimens outside these five classes or for open-world verification, utilize the **Intelligent Species Assessment** tool below or explore the **Species Intelligence Explorer** database."
            )

    img_col, ana_col = st.columns([1.0, 1.0])
    with img_col:
        img_upload = st.file_uploader("Upload fish photo (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], key="single_img_upload")
        if img_upload is not None:
            image = Image.open(img_upload)
            st.image(image, caption="Uploaded Fish Image", use_container_width=True)

    with ana_col:
        st.markdown("#### Identification Results")
        if img_upload is not None:
            try:
                image = Image.open(img_upload)
                if image.width < 16 or image.height < 16:
                    st.error("Uploaded image is too small (minimum 16x16 pixels required).")
                else:
                    result = predict_species(image)

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Predicted Species", result.species.capitalize())
                    with m2:
                        st.metric("Model Confidence", f"{result.confidence:.2%}")
                    with m3:
                        status_color = "🟢" if result.uncertainty_status == "CONFIDENT" else ("🟡" if result.uncertainty_status == "MARGINAL" else "🔴")
                        st.metric("Prediction State", f"{status_color} {result.uncertainty_status}")

                    st.caption(
                        f"**Decision Metrics:** Margin to alternative: `{result.margin_to_second:.1%}` | "
                        f"Predictive Entropy: `{result.entropy:.2f} bits` (Uniform baseline: 2.32 bits)"
                    )

                    if result.warning:
                        st.warning(f"⚠️ {result.warning}")

                    st.info(f"🔒 **Closed-World Constraint:** {result.closed_world_caveat}")

                    if result.top_alternatives:
                        st.markdown("##### Top Alternative Classes")
                        alt_df = pd.DataFrame([
                            {"Alternative Species": a[0].capitalize(), "Probability": f"{a[1]:.2%}"}
                            for a in result.top_alternatives
                        ])
                        st.dataframe(alt_df, use_container_width=True, hide_index=True)

                    with st.expander("All Class Probabilities"):
                        st.json({k: round(v, 4) for k, v in sorted(result.scores.items(), key=lambda item: item[1], reverse=True)})

                    # Automatic Linkage to Species Intelligence
                    profile = get_species_profile(result.species)
                    if profile:
                        st.markdown("---")
                        with st.expander(f"🐟 Species Intelligence: {profile.common_name} Husbandry & Biology", expanded=True):
                            st.markdown(f"**Scientific Name:** *{profile.scientific_name}* | **Family:** `{profile.family}`")
                            st.markdown(f"**Target Harvest Weight:** `{profile.target_harvest_weight}` | **Growth Cycle:** `{profile.growth_cycle_months}`")
                            st.markdown("##### Diagnostic Traits:")
                            for trait in profile.identifying_characteristics[:3]:
                                st.markdown(f"- {trait}")
                            st.info(f"**Operational Water Quality:** {profile.water_quality_notes}")

                    # Intelligent Species Assessment
                    st.markdown("---")
                    st.markdown("#### 🤖 Intelligent Species Assessment")
                    st.caption("Obtain an additional AI-powered assessment of the uploaded organism to complement the primary image-based prediction.")

                    col_btn, col_hint = st.columns([0.45, 0.55])
                    with col_btn:
                        request_gemini = st.button("Run Additional Assessment", key="gemini_btn_workflow1")
                    with col_hint:
                        if not active_gemini_key:
                            st.info("💡 Set API key in environment or sidebar to run additional assessment.")

                    if request_gemini:
                        if not active_gemini_key:
                            st.warning("⚠️ No API key detected. Please configure your key in the sidebar or via the environment variable.")
                        else:
                            with st.spinner("Running additional intelligent assessment..."):
                                gemini_res = gemini_service.analyze_image(image, api_key=active_gemini_key)

                            if gemini_res.is_success:
                                st.success("✅ Additional assessment completed successfully")
                                g_col1, g_col2 = st.columns(2)
                                with g_col1:
                                    st.metric("AI-Assisted Identification", gemini_res.species_name)
                                    if gemini_res.scientific_name:
                                        st.caption(f"*Scientific Name:* **{gemini_res.scientific_name}**")
                                with g_col2:
                                    st.metric("Assessment Confidence", gemini_res.confidence)
                                    match_badge = (
                                        f"🟢 Farm Species Match: {gemini_res.matched_supported_species.capitalize()}"
                                        if gemini_res.is_supported_farm_species and gemini_res.matched_supported_species
                                        else "🟡 Open-World / Non-Farm Organism"
                                    )
                                    st.caption(match_badge)

                                if gemini_res.explanation:
                                    st.markdown(f"**Biological Assessment:** {gemini_res.explanation}")

                                if gemini_res.anatomical_observations:
                                    st.markdown("**Observed Diagnostic Traits:**")
                                    for obs in gemini_res.anatomical_observations:
                                        st.markdown(f"- {obs}")

                                if gemini_res.notes:
                                    st.info(f"📝 *Advisory Notes:* {gemini_res.notes}")

                                st.caption("⚠️ *Advisory Notice:* AI-assisted assessments are secondary estimates and may be incorrect. Always confirm diagnostic anatomical traits prior to operational interventions.")
                            elif gemini_res.status == "BILLING_BLOCKED":
                                st.warning(f"⚠️ {gemini_res.error_message}")
                            elif gemini_res.status == "QUOTA_EXCEEDED":
                                st.warning(f"⏳ {gemini_res.error_message}")
                            else:
                                st.error(f"❌ Additional assessment failed: {gemini_res.error_message}")
                                st.info("The local EfficientNet-B0 model predictions above remain fully active and valid.")

            except Exception as exc:
                st.error(f"Inference error: {exc}")
        else:
            st.info("Upload a fish image to run automated species identification.")

# -----------------------------------------------------------------------------
# WORKFLOW 2: Combined Farm Workflow
# -----------------------------------------------------------------------------
elif mode == "Combined Farm Workflow":
    st.markdown(
        """
        <div class="card">
          <h3>Combined Identification & Field Biomass Flow</h3>
          <p>
            Upload a fish photo to automatically detect species, then enter field length/girth to compute individual and batch biomass.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("combined_form"):
        c_left, c_right = st.columns([1.0, 1.0])
        with c_left:
            image_file = st.file_uploader("Fish Photo (triggers automated species identification)", type=["jpg", "jpeg", "png", "webp"])
            species_override = st.selectbox(
                "Manual Species Override (optional)",
                ["auto", "tilapia", "catfish", "carp", "milkfish", "pangasius"],
                format_func=lambda s: "Use Automated Detection" if s == "auto" else s.capitalize(),
            )
        with c_right:
            c_length = st.number_input("Field Measured Length (cm)", min_value=1.0, value=25.0, step=0.5)
            c_girth = st.number_input("Field Measured Girth (cm)", min_value=0.0, value=12.0, step=0.5)
            c_use_girth = st.checkbox("Include girth measurement", value=True)
            c_count = st.number_input("Pond Stock Count (fish count)", min_value=1, value=1500, step=100)

        run_combined = st.form_submit_button("Run Farm Workflow")

    if run_combined:
        girth_val = c_girth if (c_use_girth and c_girth > 0) else None

        col_img, col_metrics = st.columns([1.0, 1.0])
        chosen_species = "tilapia"
        spec_conf = None

        with col_img:
            if image_file is not None:
                img = Image.open(image_file)
                st.image(img, caption="Inspected Fish Image", use_container_width=True)
                if species_model_is_ready:
                    spec_res = predict_species(img)
                    if species_override == "auto":
                        chosen_species = spec_res.species
                        spec_conf = spec_res.confidence
                    else:
                        chosen_species = species_override
                    if spec_res.warning and species_override == "auto":
                        st.warning(f"⚠️ {spec_res.warning}")
                else:
                    chosen_species = "tilapia" if species_override == "auto" else species_override
            else:
                chosen_species = "tilapia" if species_override == "auto" else species_override
                st.info("No image uploaded. Operating on manual species selection.")

        with col_metrics:
            biomass_res = predict_biomass(chosen_species, c_length, girth_val)
            batch_res = estimate_batch_biomass(chosen_species, biomass_res.biomass_kg, c_count)

            cm1, cm2 = st.columns(2)
            with cm1:
                st.metric("Species Profile", chosen_species.capitalize())
            with cm2:
                if spec_conf is not None:
                    st.metric("Species Confidence", f"{spec_conf:.2%}")
                else:
                    st.metric("Species Mode", "Manual Selection")

            st.metric("Individual Weight Estimate", f"{biomass_res.biomass_kg:.3f} kg", f"{biomass_res.biomass_kg*1000:.0f} g")
            st.metric("Total Pond Biomass (Est.)", f"{batch_res.total_biomass_kg:,.2f} kg", f"For {batch_res.effective_fish_count:,} fish")
            st.metric("Daily Feed Ration", f"{batch_res.estimated_daily_feed_kg:.2f} kg / day", "At 2.5% body weight")
            st.caption(f"Biomass method: `{biomass_res.method}`")

        spec_profile = get_species_profile(chosen_species)
        if spec_profile:
            with st.expander(f"Species Intelligence: {spec_profile.common_name} Husbandry Guidelines", expanded=True):
                st.markdown(f"**Target Harvest Size:** {spec_profile.target_harvest_weight} | **Growth Cycle:** {spec_profile.growth_cycle_months}")
                st.markdown(f"**Water Quality Criteria:** {spec_profile.water_quality_notes}")

# -----------------------------------------------------------------------------
# WORKFLOW 3: Biomass & Batch Estimation
# -----------------------------------------------------------------------------
elif mode == "Biomass & Batch Estimation":
    st.markdown(
        """
        <div class="card">
          <h3>Field-Measurement Biomass & Batch Calculator</h3>
          <p>
            Estimate individual fish weight and aggregate pond/tank biomass using verified mathematical
            length-weight growth equations and machine learning regression.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    species_list = ["tilapia", "catfish", "carp", "milkfish", "pangasius"]

    b_left, b_right = st.columns([1.0, 1.0])

    with b_left:
        with st.form("biomass_form"):
            st.markdown("#### 1. Individual Fish Measurements")
            b_species = st.selectbox("Species", species_list, format_func=lambda s: s.capitalize())
            b_length = st.number_input("Measured Length (cm)", min_value=1.0, max_value=200.0, value=25.0, step=0.5)
            b_girth = st.number_input("Measured Maximum Girth (cm)", min_value=0.0, max_value=150.0, value=12.0, step=0.5)
            b_use_girth = st.checkbox("Include girth measurement in formula", value=True)

            st.markdown("#### 2. Population / Batch Parameters (Optional)")
            b_enable_batch = st.checkbox("Calculate total pond/tank biomass", value=True)
            b_fish_count = st.number_input("Estimated Fish Stock (count)", min_value=1, max_value=1000000, value=2500, step=100)
            b_survival_rate = st.slider("Estimated Survival Rate (%)", min_value=10.0, max_value=100.0, value=90.0, step=1.0)
            b_daily_feed_rate = st.slider("Target Daily Feeding Rate (% body weight)", min_value=0.5, max_value=10.0, value=2.5, step=0.1)

            submit_biomass = st.form_submit_button("Calculate Biomass Estimates")

    with b_right:
        st.markdown("#### Calculation Results")
        if submit_biomass:
            girth_arg = b_girth if (b_use_girth and b_girth > 0) else None
            try:
                single_result = predict_biomass(b_species, b_length, girth_arg)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Individual Weight Estimate", f"{single_result.biomass_kg:.3f} kg", f"{single_result.biomass_kg * 1000:.1f} g")
                with m2:
                    st.metric("Target Species", single_result.species.capitalize())

                if getattr(single_result, "is_experimental", False):
                    st.warning(f"🧪 **Experimental Model:** `{single_result.method}`\n\n{getattr(single_result, 'notes', '')}")
                else:
                    st.success(f"📚 **Verified Fisheries Model:** `{single_result.method}`\n\n{getattr(single_result, 'notes', '')}")

                if getattr(single_result, "citation", None):
                    st.caption(f"**Literature Citation:** *{single_result.citation}*")

                if b_enable_batch:
                    batch_result = estimate_batch_biomass(
                        species=b_species,
                        individual_weight_kg=single_result.biomass_kg,
                        fish_count=b_fish_count,
                        survival_rate_percent=b_survival_rate,
                        daily_feed_rate_percent=b_daily_feed_rate,
                    )
                    st.markdown("---")
                    st.markdown("#### Batch Population Summary")
                    bm1, bm2 = st.columns(2)
                    with bm1:
                        st.metric("Effective Population", f"{batch_result.effective_fish_count:,} fish", f"{batch_result.survival_rate_percent:.0f}% survival")
                    with bm2:
                        st.metric("Total Batch Biomass", f"{batch_result.total_biomass_kg:,.2f} kg")

                    st.metric(
                        "Estimated Daily Feed Requirement",
                        f"{batch_result.estimated_daily_feed_kg:.2f} kg / day",
                        f"Based on {batch_result.feed_rate_percent:.1f}% daily ration",
                    )

                st.success("Calculations completed successfully.")
            except Exception as exc:
                st.error(f"Calculation error: {exc}")
        else:
            st.info("Submit the form to compute individual weight and batch biomass estimates.")

        st.warning(
            "⚠️ **Methodology & Integrity Notice:** Length and girth must be obtained via physical field "
            "calibration or tape measurement. Uncalibrated 2D monocular images cannot compute metric length or weight "
            "without fixed reference markers and depth calibration."
        )

# -----------------------------------------------------------------------------
# WORKFLOW 4: Species Intelligence Explorer
# -----------------------------------------------------------------------------
elif mode == "Species Intelligence Explorer":
    st.markdown(
        """
        <div class="card">
          <h3>Aquaculture Species Intelligence & Global Taxonomy</h3>
          <p>
            Searchable biological reference database and global fish taxonomy explorer. Distinct from image-based classification,
            this knowledge hub allows you to query diagnostic morphological traits, operational water quality criteria,
            and explore over 36,000 species indexed in FishBase v25.04.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("🔍 **Interactive Knowledge Hub:** Search species by biological traits, culture systems, or taxonomy — independent of image uploads.")

    tab_core, tab_fishbase = st.tabs([
        "🐟 Core Aquaculture Profiles (Detailed Husbandry)",
        "🌐 Global FishBase Registry (36,132 Species)",
    ])

    with tab_core:
        search_col, select_col = st.columns([1.2, 0.8])
        with search_col:
            search_query = st.text_input(
                "Search species knowledge base",
                placeholder="Type species name, trait (e.g. barbels), culture system (e.g. biofloc), or diet...",
                key="core_species_search",
            )
        with select_col:
            all_profiles = get_all_species_profiles()
            species_choice = st.selectbox(
                "Or jump directly to species",
                options=list(all_profiles.keys()),
                format_func=lambda k: all_profiles[k].common_name,
                key="core_species_select",
            )

        matching_profiles = search_species_profiles(search_query) if search_query else [all_profiles[species_choice]]

        if not matching_profiles:
            st.warning(f"No species profiles found matching query: '{search_query}'. Showing default species.")
            matching_profiles = [all_profiles[species_choice]]

        for profile in matching_profiles:
            with st.expander(f"🐟 {profile.common_name} ({profile.scientific_name})", expanded=True):
                st.markdown(f"**Family:** `{profile.family}` | **Trophic Level:** `{profile.trophic_level}`")
                st.markdown(f"**Target Harvest Weight:** `{profile.target_harvest_weight}` | **Growth Cycle:** `{profile.growth_cycle_months}` | **Typical FCR:** `{profile.typical_fcr}`")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Diagnostic Characteristics",
                    "Aquaculture & Husbandry",
                    "Water Quality (Context-Dependent)",
                    "Habitat & Natural Diet",
                    "Scientific References",
                ])

                with tab1:
                    st.markdown("#### Identifying Morphological Traits")
                    for trait in profile.identifying_characteristics:
                        st.markdown(f"- {trait}")

                with tab2:
                    st.markdown("#### Farming Systems & Commercial Management")
                    st.markdown("**Culture Systems:**")
                    for sys_item in profile.farming_systems:
                        st.markdown(f"- {sys_item}")
                    st.markdown("**Husbandry Best Practices:**")
                    for mgmt in profile.aquaculture_management:
                        st.markdown(f"- {mgmt}")

                with tab3:
                    st.markdown("#### Operational Water-Quality Parameters")
                    st.info(
                        f"**Contextual Notice:** {profile.water_quality_notes}\n\n"
                        "*Values represent commercial operational guidance and vary across rearing stages, stocking densities, and system designs.*"
                    )
                    wq_data = []
                    for p in profile.water_quality_parameters:
                        wq_data.append({
                            "Parameter": p.parameter,
                            "Optimal Range": p.optimal_range,
                            "Tolerance Limit": p.tolerance_range,
                            "Unit": p.unit,
                            "Operational Notes": p.notes,
                        })
                    st.dataframe(pd.DataFrame(wq_data), use_container_width=True, hide_index=True)

                with tab4:
                    st.markdown("#### Native Habitat")
                    st.write(profile.habitat)
                    st.markdown("#### Natural & Commercial Diet")
                    st.write(profile.diet)

                with tab5:
                    st.markdown("#### Factual Biological References")
                    for ref in profile.references:
                        st.markdown(f"- *{ref}*")

    with tab_fishbase:
        st.markdown("### Global FishBase Species Explorer")
        st.caption(
            "Search and inspect global fish species records from FishBase v25.04. "
            "This is a taxonomic and biological reference lookup — distinct from automated image classification."
        )
        st.info(f"📚 **License & Attribution:** {FISHBASE_ATTRIBUTION}")

        fb_service = get_fishbase_service()

        if not fb_service.is_available:
            st.warning(f"FishBase registry is currently unavailable: {fb_service.load_error}")
        else:
            fb_col_search, fb_col_limit, fb_col_filter = st.columns([2.0, 0.7, 1.3])
            with fb_col_search:
                fb_query = st.text_input(
                    "Search by Scientific Name, Common Name, or Genus",
                    placeholder="e.g. Nile tilapia, Salmo salar, Cyprinus, Catfish...",
                    key="fishbase_search_input",
                )
            with fb_col_limit:
                fb_limit = st.number_input(
                    "Result Limit",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key="fishbase_limit_input",
                )
            with fb_col_filter:
                st.write("")
                st.write("")
                fb_aqua_only = st.checkbox(
                    "Aquaculture use only",
                    value=False,
                    help="Filter to species with commercial, experimental, or future aquaculture records",
                    key="fishbase_aqua_checkbox",
                )

            results = fb_service.search(
                query=fb_query,
                limit=int(fb_limit),
                aquaculture_only=fb_aqua_only,
            )

            if not results:
                st.warning(
                    f"No species found matching '{fb_query}'. "
                    "Try another common name, genus, or scientific binomial."
                )
            else:
                total_label = f"Showing top {len(results)} matches"
                if fb_query.strip():
                    total_label += f" for query '{fb_query}'"
                st.caption(f"**{total_label}** (Total database: {fb_service.total_count:,} species)")

                for rec in results:
                    with st.expander(f"🐟 {rec.display_title}", expanded=len(results) == 1):
                        r_col1, r_col2, r_col3 = st.columns(3)
                        with r_col1:
                            st.markdown(f"**Scientific Name:** *{rec.scientific_name}*")
                            if rec.author:
                                st.caption(f"Authority: {rec.author}")
                            st.markdown(f"**Genus:** `{rec.genus}`")
                            fam_text = f"FamCode `{rec.family_code}`" if rec.family_code is not None else "Not recorded"
                            st.markdown(f"**Family Code:** {fam_text}")
                            st.caption("*Note: Family code from FishBase schema; family name string not in table.*")

                        with r_col2:
                            com_name_text = rec.common_name or "No official common name recorded in FishBase"
                            st.markdown(f"**Common Name:** {com_name_text}")
                            aqua_badge = rec.aquaculture_use.capitalize() if rec.aquaculture_use else "Not specified"
                            st.markdown(f"**Aquaculture Use:** `{aqua_badge}`")
                            st.markdown(f"**Habitats:** {rec.habitat_display}")
                            if rec.demersal_pelagic:
                                st.caption(f"Zonation: {rec.demersal_pelagic}")

                        with r_col3:
                            st.markdown(f"**Length Dimensions:** {rec.length_display}")
                            st.markdown(f"**Max Weight:** {rec.weight_display}")
                            if rec.body_shape:
                                st.markdown(f"**Body Shape:** `{rec.body_shape}`")

                        if rec.comments:
                            st.markdown("---")
                            st.markdown(f"**Biological & Ecological Notes:** {rec.comments}")

# -----------------------------------------------------------------------------
# WORKFLOW 5: Documentation & Model Evaluation
# -----------------------------------------------------------------------------
elif mode == "Documentation & Model Evaluation":
    st.markdown(
        """
        <div class="card">
          <h3>AquaVision Model Architecture & Held-Out Evaluation Report</h3>
          <p>
            Detailed specifications, actual evaluation metrics measured on held-out test data, and execution guide.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_report_file = Path("models/evaluation_report.json")
    if eval_report_file.exists():
        with open(eval_report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        sp_data = report_data.get("species_model", {}).get("metrics", {})
        st.markdown("### Measured Species Classifier Test Performance")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Test Accuracy", f"{sp_data.get('test_accuracy', 0.0):.2%}")
        with m_c2:
            macro_f1 = sp_data.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0)
            st.metric("Macro F1-Score", f"{macro_f1:.3f}")
        with m_c3:
            st.metric("Total Real Samples", sp_data.get("sample_count_total", 0))
        with m_c4:
            st.metric("Held-Out Test Samples", sp_data.get("sample_count_test", 0))

        st.markdown("#### Per-Class Classification Report (Held-Out Test Set)")
        cls_rows = []
        for cname in sp_data.get("classes", []):
            c_metric = sp_data.get("classification_report", {}).get(cname, {})
            cls_rows.append({
                "Species": cname.capitalize(),
                "Precision": f"{c_metric.get('precision', 0.0):.3f}",
                "Recall": f"{c_metric.get('recall', 0.0):.3f}",
                "F1-Score": f"{c_metric.get('f1-score', 0.0):.3f}",
                "Test Support": int(c_metric.get("support", 0)),
            })
        st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Confusion Matrix")
        classes = sp_data.get("classes", [])
        cm_data = sp_data.get("confusion_matrix", [])
        st.dataframe(
            pd.DataFrame(cm_data, index=[c.capitalize() for c in classes], columns=[c.capitalize() for c in classes]),
            use_container_width=True,
        )

        cv_info = sp_data.get("cv_validation", {})
        if cv_info:
            st.markdown("#### 5-Fold Cross-Validation on Training Set (Pre-Test Model Selection)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("CV Accuracy", f"{cv_info.get('mean_accuracy', 0.0):.2%} ± {cv_info.get('std_accuracy', 0.0):.2%}")
            with c2:
                st.metric("CV Macro F1", f"{cv_info.get('mean_macro_f1', 0.0):.3f} ± {cv_info.get('std_macro_f1', 0.0):.3f}")
            with c3:
                st.metric("Validation Folds", cv_info.get("cv_folds", 5))

        leakage_info = sp_data.get("leakage_audit", {})
        if leakage_info:
            st.markdown("#### Evaluation Leakage Audit & Scientific Limitations")
            st.info(
                f"**Integrity Protocol:** {leakage_info.get('observation_id_integrity', 'N/A')}\n\n"
                f"**Known Risk:** {leakage_info.get('leakage_risk_limitation', 'N/A')}"
            )

        st.caption(f"Dataset source: {report_data.get('dataset_source')} | License: {report_data.get('license')}")
    else:
        st.info("No evaluation report found in `models/evaluation_report.json`.")

    st.markdown("---")
    st.markdown("### Execution Command")
    st.code(
        "streamlit run app.py",
        language="bash",
    )

st.markdown("---")
st.markdown(
    "<span class='small-note'>AquaVision: AI-Based Fish Species Identification, Biomass Estimation and Species Intelligence System.</span>",
    unsafe_allow_html=True,
)
