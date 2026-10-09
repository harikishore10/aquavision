from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from fishfarm_ai.services import BiomassEstimationService

st.set_page_config(page_title="Fish Biomass Estimation", page_icon="⚖️", layout="wide")

st.title("Fish Biomass Estimation")
st.caption("Enter species and measurements to estimate fish biomass." )

species_options = ["tilapia", "catfish", "carp", "milkfish", "pangasius"]

left, right = st.columns([1.0, 1.0])

with left:
    species = st.selectbox("Species", species_options)
    length_cm = st.number_input("Length (cm)", min_value=0.0, value=25.0, step=0.5)
    girth_cm = st.number_input("Girth (cm)", min_value=0.0, value=12.0, step=0.5)
    use_girth = st.checkbox("Use girth in biomass estimate", value=True)

with right:
    st.markdown("### Output")
    service = BiomassEstimationService()
    result = service.predict(species, length_cm, girth_cm if use_girth else None)
    st.metric("Species", result.species.capitalize())
    st.metric("Estimated Biomass", f"{result.biomass_kg:.3f} kg", f"{result.biomass_kg * 1000:.1f} g")

    if getattr(result, "is_experimental", False):
        st.warning(f"🧪 **Experimental Model:** `{result.method}`\n\n{getattr(result, 'notes', '')}")
    else:
        st.success(f"📚 **Verified Fisheries Model:** `{result.method}`\n\n{getattr(result, 'notes', '')}")

    if getattr(result, "citation", None):
        st.caption(f"**Literature Citation:** *{result.citation}*")

    st.info(
        "⚖️ **Physical Metric Measurement:** Length and girth are manual physical inputs. "
        "Camera images alone cannot measure physical metric dimensions without in-situ optical calibration markers."
    )
