from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.gemini_service import GeminiVisionService
from fishfarm_ai.services import SpeciesIdentificationService

st.set_page_config(page_title="Fish Species Identification", page_icon="🐟", layout="wide")

st.title("Fish Species Identification")
st.caption("Real computer vision classifier trained on verified aquaculture species observations.")

service = SpeciesIdentificationService()

if not service.is_ready():
    st.error("🛑 Species classifier model is offline. No trained weights found at `models/species_model.joblib`.")
    st.info("Run `python train.py` to train the classifier bundle on `data/species_images`.")
else:
    classes_str = ", ".join(s.capitalize() for s in sorted(service.species_model.class_names))
    st.info(
        "🐟 **Fish Species Identification Platform:** Upload an aquatic specimen image to run local computer vision classification, "
        "inspect biological profiles, and obtain optional cloud AI assessments."
    )
    with st.expander("ℹ️ Local Classifier Specifications & Trained Target Species", expanded=False):
        st.markdown(
            f"**Target Aquaculture Finfish Trained in Local Model:** **{classes_str}**\n\n"
            "*Scope & Boundary:* The local vision model is trained specifically on five core aquaculture finfish classes. "
            "For specimens outside this training scope or for second opinions on open-world aquatic organisms, utilize the Intelligent Species Assessment feature below."
        )

left, right = st.columns([1.05, 0.95])

with left:
    uploaded = st.file_uploader("Upload fish image (JPG, JPEG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])

with right:
    st.markdown("### Output")
    if uploaded is None:
        st.info("Upload an image to identify the fish species.")
    else:
        try:
            image = Image.open(uploaded)
            if image.width < 16 or image.height < 16:
                st.error("Uploaded image is too small. Please upload a clear image of at least 16x16 resolution.")
            else:
                st.image(image, caption="Uploaded fish image", use_container_width=True)
                result = service.predict(image)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Predicted Species", result.species.capitalize())
                with m2:
                    st.metric("Model Confidence", f"{result.confidence:.2%}")
                with m3:
                    state_icon = "🟢" if result.uncertainty_status == "CONFIDENT" else ("🟡" if result.uncertainty_status == "MARGINAL" else "🔴")
                    st.metric("State", f"{state_icon} {result.uncertainty_status}")

                st.caption(
                    f"**Metrics:** Margin to runner-up: `{result.margin_to_second:.1%}` | "
                    f"Entropy: `{result.entropy:.2f} bits` | "
                    f"Calibration: `{getattr(result, 'calibration_method', 'Standard')}`"
                )

                if result.warning:
                    st.warning(f"⚠️ {result.warning}")

                st.info(f"🔒 **Closed-World Scope:** {result.closed_world_caveat}")

                if result.top_alternatives:
                    st.markdown("#### Top Alternative Classes")
                    alt_data = [{"Species": alt[0].capitalize(), "Probability": f"{alt[1]:.2%}"} for alt in result.top_alternatives]
                    st.dataframe(pd.DataFrame(alt_data), use_container_width=True, hide_index=True)

                with st.expander("All Class Probabilities"):
                    st.json({k: round(v, 4) for k, v in sorted(result.scores.items(), key=lambda item: item[1], reverse=True)})

                # Species Intelligence Integration
                if result.species_profile:
                    p = result.species_profile
                    with st.expander(f"📖 Species Intelligence: {p.common_name} Profile", expanded=True):
                        st.markdown(f"**Scientific Name:** *{p.scientific_name}*")
                        st.markdown(f"**Family:** `{p.family}` | **Trophic Level:** `{p.trophic_level}`")
                        st.markdown(f"**Target Harvest Weight:** `{p.target_harvest_weight}` | **FCR:** `{p.typical_fcr}`")
                        st.markdown("##### Distinctive Traits:")
                        for trait in p.identifying_characteristics[:3]:
                            st.markdown(f"- {trait}")
                        st.markdown(f"*{p.water_quality_notes}*")

                # Intelligent Species Assessment
                st.markdown("---")
                st.markdown("#### 🤖 Intelligent Species Assessment")
                st.caption("Obtain an additional AI-powered assessment of the uploaded organism to complement the primary image-based prediction.")

                detected_key = GeminiVisionService.get_api_key()
                col_g_btn, col_g_key = st.columns([0.45, 0.55])
                with col_g_key:
                    gem_key_input = st.text_input(
                        "Assessment API Key (Optional)",
                        value=detected_key or "",
                        type="password",
                        key="gemini_key_dedicated",
                        help="Leave blank if environment variable is set. Never saved to source code.",
                    )
                with col_g_btn:
                    st.write("")
                    ask_gemini = st.button("Run Additional Assessment", key="gemini_btn_dedicated")

                active_key = gem_key_input.strip() or detected_key
                if ask_gemini:
                    if not active_key:
                        st.warning("⚠️ No API key provided. Set the environment variable or enter your key above.")
                    else:
                        with st.spinner("Running additional intelligent assessment..."):
                            g_service = GeminiVisionService()
                            gemini_res = g_service.analyze_image(image, api_key=active_key)

                        if gemini_res.is_success:
                            st.success("✅ Additional assessment completed successfully")
                            gc1, gc2 = st.columns(2)
                            with gc1:
                                st.metric("AI-Assisted Identification", gemini_res.species_name)
                                if gemini_res.scientific_name:
                                    st.caption(f"*Scientific Name:* **{gemini_res.scientific_name}**")
                            with gc2:
                                st.metric("Assessment Confidence", gemini_res.confidence)
                                match_badge = (
                                    f"🟢 Matches Farm Class: {gemini_res.matched_supported_species.capitalize()}"
                                    if gemini_res.is_supported_farm_species and gemini_res.matched_supported_species
                                    else "🟡 Non-Farm / Open-World Organism"
                                )
                                st.caption(match_badge)

                            if gemini_res.explanation:
                                st.markdown(f"**Biological Assessment:** {gemini_res.explanation}")
                            if gemini_res.anatomical_observations:
                                st.markdown("**Observed Traits:**")
                                for o in gemini_res.anatomical_observations:
                                    st.markdown(f"- {o}")
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
            st.error(f"Image processing error: {exc}")

st.markdown("---")
st.markdown("AquaVision: Real fish species identification pipeline.")
