import pandas as pd
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'."
        )
    df = pd.read_csv(p)
    logger.info(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
    return df


