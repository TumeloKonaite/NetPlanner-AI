import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    base_dir: Path = Path(__file__).resolve().parents[2]
    source_data_path: Path = base_dir / "data" / "telecom_dataset.csv"
    train_data_path: Path = base_dir / "artifacts" / "train.csv"
    test_data_path: Path = base_dir / "artifacts" / "test.csv"
    raw_data_path: Path = base_dir / "artifacts" / "raw_measurements.csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.required_columns = [
            "tower_id",
            "latitude",
            "longitude",
            "city",
            "operator",
            "network_type",
            "signal_strength_dbm",
            "coverage_quality",
            "download_speed_mbps",
            "upload_speed_mbps",
            "latency_ms",
            "distance_to_tower_km",
            "indoor_outdoor",
        ]

    def _validate_dataset(self, df: pd.DataFrame) -> None:
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise CustomException(
                f"Missing required telecom columns: {missing_cols}",
                sys,
            )

    def initiate_data_ingestion(self):
        logging.info("Entered the telecom data ingestion component")

        try:
            source_path = self.ingestion_config.source_data_path
            if not source_path.exists():
                raise FileNotFoundError(f"Source dataset not found at {source_path}")

            df = pd.read_csv(source_path)
            self._validate_dataset(df)
            logging.info("Read telecom dataset from %s with shape %s", source_path, df.shape)

            artifacts_dir = self.ingestion_config.raw_data_path.parent
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw measurement dataset to %s", self.ingestion_config.raw_data_path)

            tower_ids = df["tower_id"].dropna().astype(str).unique()
            if len(tower_ids) < 2:
                raise CustomException("Not enough unique tower_ids to create a train/test split.", sys)

            train_tower_ids, test_tower_ids = train_test_split(
                tower_ids,
                test_size=0.2,
                random_state=42,
            )

            tower_id_series = df["tower_id"].astype(str)
            train_set = df[tower_id_series.isin(train_tower_ids)].copy()
            test_set = df[tower_id_series.isin(test_tower_ids)].copy()

            if train_set.empty or test_set.empty:
                raise CustomException("Tower-based split produced an empty train or test dataset.", sys)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(
                "Data ingestion completed with %s train rows across %s towers and %s test rows across %s towers",
                len(train_set),
                train_set["tower_id"].nunique(),
                len(test_set),
                test_set["tower_id"].nunique(),
            )

            return (
                str(self.ingestion_config.train_data_path),
                str(self.ingestion_config.test_data_path),
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
