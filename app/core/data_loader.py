"""
Data loader module for ContextPilot.
Loads and validates chart metadata and verifies disk images.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.config import (
    CHARTS_DIR,
    CHART_METADATA_CSV,
    COL_CHART_TYPE,
    COL_DESCRIPTION,
    COL_FILENAME,
    COL_QUERY_INTENT,
    COL_RELEVANT_TABLES,
)


@dataclass
class ChartRecord:
    filename: str
    chart_type: str
    query_intent: str
    relevant_tables: list[str]  # split by '|'
    description: str
    image_path: Path

    @property
    def chart_id(self) -> str:
        """Returns filename without .png extension as clean ID."""
        return Path(self.filename).stem

    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "filename": self.filename,
            "chart_type": self.chart_type,
            "query_intent": self.query_intent,
            "relevant_tables": "|".join(self.relevant_tables),
            "description": self.description,
            "image_path": str(self.image_path),
        }


def load_chart_dataframe() -> pd.DataFrame:
    """Loads raw metadata DataFrame from CSV and verifies required columns."""
    if not CHART_METADATA_CSV.exists():
        raise FileNotFoundError(f"Chart metadata CSV missing at {CHART_METADATA_CSV}")

    df = pd.read_csv(CHART_METADATA_CSV)

    required_cols = [
        COL_FILENAME,
        COL_CHART_TYPE,
        COL_QUERY_INTENT,
        COL_RELEVANT_TABLES,
        COL_DESCRIPTION,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    return df


def load_chart_records() -> list[ChartRecord]:
    """
    Loads all chart records and verifies corresponding PNG files exist.
    Raises FileNotFoundError if any chart PNG is missing.
    """
    df = load_chart_dataframe()
    records: list[ChartRecord] = []
    missing_images: list[str] = []

    for _, row in df.iterrows():
        fname = str(row[COL_FILENAME]).strip()
        img_path = CHARTS_DIR / fname

        if not img_path.is_file():
            missing_images.append(fname)
            continue

        raw_tables = str(row[COL_RELEVANT_TABLES]).strip()
        tables = [t.strip() for t in raw_tables.split("|") if t.strip()]

        records.append(
            ChartRecord(
                filename=fname,
                chart_type=str(row[COL_CHART_TYPE]).strip(),
                query_intent=str(row[COL_QUERY_INTENT]).strip(),
                relevant_tables=tables,
                description=str(row[COL_DESCRIPTION]).strip(),
                image_path=img_path,
            )
        )

    if missing_images:
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing image files. First few: {missing_images[:3]}"
        )

    return records


def get_sample_queries(limit: int = 10) -> list[dict]:
    """
    Returns sample queries with chart details for UI dropdowns/quick selection.
    Uses chart description as the query text.
    """
    records = load_chart_records()
    samples = []
    for rec in records[:limit]:
        samples.append({
            "query_text": rec.description,
            "chart_id": rec.chart_id,
            "filename": rec.filename,
            "chart_type": rec.chart_type,
            "query_intent": rec.query_intent,
        })
    return samples


if __name__ == "__main__":
    recs = load_chart_records()
    print(f"Successfully loaded {len(recs)} chart records.")
    print("Sample record:", recs[0])
