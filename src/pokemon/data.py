from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_card_data(data_dir: str = "data/raw/") -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    en = pd.read_csv(data_path / "EN_Card_Data.csv")
    jp = data_path / "JP_Card_Data.csv"
    jp_df = pd.read_csv(jp) if jp.exists() else None

    logger.info("Loaded EN card data: %d cards", len(en))
    if jp_df is not None:
        logger.info("Loaded JP card data: %d cards", len(jp_df))

    return {"en": en, "jp": jp_df}
