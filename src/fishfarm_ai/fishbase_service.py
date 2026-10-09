"""FishBase Species Registry Service for AquaVision.

Provides fast, in-memory, case-insensitive exploration across the global FishBase
species taxonomy (36,132 species) using partitioned parquet data.

License & Attribution:
Data source: FishBase v25.04 (Froese & Pauly, 2025).
Distributed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
Not licensed or cleared for commercial operational use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_FISHBASE_PATH


FISHBASE_ATTRIBUTION = (
    "Data source: FishBase v25.04 (Froese & Pauly, 2025). "
    "Distributed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). "
    "Academic and non-commercial research use only; not cleared for commercial use."
)

REQUIRED_COLUMNS = [
    "SpecCode",
    "Genus",
    "Species",
    "FBname",
    "Author",
    "FamCode",
    "BodyShapeI",
    "Fresh",
    "Brack",
    "Saltwater",
    "DemersPelag",
    "Length",
    "CommonLength",
    "Weight",
    "UsedforAquaculture",
    "Comments",
]


@dataclass(frozen=True)
class FishBaseRecord:
    spec_code: int
    genus: str
    species: str
    scientific_name: str
    common_name: str | None = None
    author: str | None = None
    family_code: int | None = None
    body_shape: str | None = None
    habitats: list[str] = field(default_factory=list)
    demersal_pelagic: str | None = None
    max_length_cm: float | None = None
    common_length_cm: float | None = None
    max_weight_g: float | None = None
    aquaculture_use: str | None = None
    comments: str | None = None

    @property
    def display_title(self) -> str:
        if self.common_name:
            return f"{self.common_name} ({self.scientific_name})"
        return self.scientific_name

    @property
    def habitat_display(self) -> str:
        if self.habitats:
            return ", ".join(self.habitats)
        return "Not recorded"

    @property
    def length_display(self) -> str:
        parts = []
        if self.max_length_cm is not None:
            parts.append(f"Max: {self.max_length_cm:.1f} cm")
        if self.common_length_cm is not None:
            parts.append(f"Typical: {self.common_length_cm:.1f} cm")
        return " | ".join(parts) if parts else "Not recorded"

    @property
    def weight_display(self) -> str:
        if self.max_weight_g is not None:
            if self.max_weight_g >= 1000.0:
                return f"{self.max_weight_g / 1000.0:.2f} kg ({self.max_weight_g:,.0f} g)"
            return f"{self.max_weight_g:.1f} g"
        return "Not recorded"


class FishBaseService:
    """Service for searching and retrieving species records from FishBase v25.04."""

    def __init__(self, parquet_path: Path | str = DEFAULT_FISHBASE_PATH) -> None:
        self.parquet_path = Path(parquet_path)
        self._df: pd.DataFrame | None = None
        self._load_error: str | None = None
        self._load_data()

    def _load_data(self) -> None:
        """Load required columns from parquet and build search indices."""
        if not self.parquet_path.exists():
            self._load_error = f"FishBase dataset file not found at: {self.parquet_path}"
            self._df = None
            return

        try:
            df = pd.read_parquet(self.parquet_path, columns=REQUIRED_COLUMNS)

            # Build cleaned fields
            genus_clean = df["Genus"].fillna("").astype(str).str.strip()
            species_clean = df["Species"].fillna("").astype(str).str.strip()
            df["ScientificName"] = genus_clean + " " + species_clean

            # Pre-compute unified lowercase search index
            fbname_clean = df["FBname"].fillna("").astype(str).str.strip()
            df["_search_index"] = (
                df["ScientificName"].str.lower()
                + " "
                + fbname_clean.str.lower()
                + " "
                + genus_clean.str.lower()
            )

            self._df = df
            self._load_error = None
        except Exception as exc:
            self._load_error = f"Error reading FishBase parquet dataset: {exc}"
            self._df = None

    @property
    def is_available(self) -> bool:
        return self._df is not None and len(self._df) > 0

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def total_count(self) -> int:
        return len(self._df) if self._df is not None else 0

    def _row_to_record(self, row: pd.Series) -> FishBaseRecord:
        """Convert a dataframe row into a strongly typed FishBaseRecord."""
        # Habitats
        habitats: list[str] = []
        if row.get("Fresh") == 1:
            habitats.append("Freshwater")
        if row.get("Brack") == 1:
            habitats.append("Brackish")
        if row.get("Saltwater") == 1:
            habitats.append("Marine / Saltwater")

        # Missing values safely parsed
        fbname = str(row["FBname"]).strip() if pd.notna(row.get("FBname")) and str(row["FBname"]).strip() else None
        author = str(row["Author"]).strip() if pd.notna(row.get("Author")) and str(row["Author"]).strip() else None
        fam_code = int(row["FamCode"]) if pd.notna(row.get("FamCode")) else None
        body_shape = str(row["BodyShapeI"]).strip() if pd.notna(row.get("BodyShapeI")) and str(row["BodyShapeI"]).strip() else None
        dem_pel = str(row["DemersPelag"]).strip() if pd.notna(row.get("DemersPelag")) and str(row["DemersPelag"]).strip() else None

        length_cm = float(row["Length"]) if pd.notna(row.get("Length")) and float(row["Length"]) > 0 else None
        com_length_cm = float(row["CommonLength"]) if pd.notna(row.get("CommonLength")) and float(row["CommonLength"]) > 0 else None
        weight_g = float(row["Weight"]) if pd.notna(row.get("Weight")) and float(row["Weight"]) > 0 else None

        aqua_use = str(row["UsedforAquaculture"]).strip() if pd.notna(row.get("UsedforAquaculture")) and str(row["UsedforAquaculture"]).strip() else None
        comments = str(row["Comments"]).strip() if pd.notna(row.get("Comments")) and str(row["Comments"]).strip() else None

        return FishBaseRecord(
            spec_code=int(row["SpecCode"]),
            genus=str(row["Genus"]).strip(),
            species=str(row["Species"]).strip(),
            scientific_name=str(row["ScientificName"]).strip(),
            common_name=fbname,
            author=author,
            family_code=fam_code,
            body_shape=body_shape,
            habitats=habitats,
            demersal_pelagic=dem_pel,
            max_length_cm=length_cm,
            common_length_cm=com_length_cm,
            max_weight_g=weight_g,
            aquaculture_use=aqua_use,
            comments=comments,
        )

    def search(
        self,
        query: str = "",
        limit: int = 50,
        aquaculture_only: bool = False,
    ) -> list[FishBaseRecord]:
        """Search FishBase species by scientific name, common name, or genus.
        
        Args:
            query: Case-insensitive search string.
            limit: Maximum number of records to return (default 50).
            aquaculture_only: Filter strictly to species labeled with commercial or experimental aquaculture use.
        """
        if not self.is_available or self._df is None:
            return []

        df = self._df

        # Filter by aquaculture utility if requested
        if aquaculture_only:
            aqua_mask = df["UsedforAquaculture"].isin(["commercial", "experimental", "likely future use"])
            df = df[aqua_mask]

        if not query or not query.strip():
            matched_df = df.head(max(1, limit))
        else:
            q_clean = query.lower().strip()
            mask = df["_search_index"].str.contains(q_clean, regex=False, na=False)
            sub_df = df[mask]
            if len(sub_df) > 1:
                exact_genus = sub_df["Genus"].str.lower() == q_clean
                exact_common = sub_df["FBname"].fillna("").str.lower() == q_clean
                starts_sci = sub_df["ScientificName"].str.lower().str.startswith(q_clean)
                prio = (exact_genus | exact_common).astype(int) * 2 + starts_sci.astype(int)
                sub_df = sub_df.assign(_prio=prio).sort_values(by="_prio", ascending=False)
            matched_df = sub_df.head(max(1, limit))

        return [self._row_to_record(row) for _, row in matched_df.iterrows()]

    def get_by_spec_code(self, spec_code: int) -> FishBaseRecord | None:
        """Lookup a species by its unique FishBase SpecCode."""
        if not self.is_available or self._df is None:
            return None
        m = self._df[self._df["SpecCode"] == spec_code]
        if len(m) > 0:
            return self._row_to_record(m.iloc[0])
        return None

    def get_by_scientific_name(self, scientific_name: str) -> FishBaseRecord | None:
        """Lookup a species by exact scientific binomial name (case-insensitive)."""
        if not self.is_available or self._df is None or not scientific_name:
            return None
        q = scientific_name.lower().strip()
        m = self._df[self._df["ScientificName"].str.lower() == q]
        if len(m) > 0:
            return self._row_to_record(m.iloc[0])
        return None


# Global cached singleton instance
_GLOBAL_FISHBASE_SERVICE: FishBaseService | None = None


def get_fishbase_service(parquet_path: Path | str = DEFAULT_FISHBASE_PATH) -> FishBaseService:
    """Retrieve or initialize the cached global FishBase service."""
    global _GLOBAL_FISHBASE_SERVICE
    if _GLOBAL_FISHBASE_SERVICE is None or _GLOBAL_FISHBASE_SERVICE.parquet_path != Path(parquet_path):
        _GLOBAL_FISHBASE_SERVICE = FishBaseService(parquet_path)
    return _GLOBAL_FISHBASE_SERVICE
