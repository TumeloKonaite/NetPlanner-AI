import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        self.model = load_object(file_path=self.model_path)
        preprocessor = load_object(file_path=self.preprocessor_path)

        self.scaler = preprocessor["scaler"]
        self.fill_values = preprocessor.get("fill_values", {})
        self.feature_columns = preprocessor.get(
            "feature_columns",
            ["upload_speed_mbps", "distance_to_tower_km", "num_measurements"],
        )

    def _prepare_features(self, tower_features: pd.DataFrame):
        missing_cols = [col for col in self.feature_columns if col not in tower_features.columns]
        if missing_cols:
            raise CustomException(
                f"Missing required NetPlanner AI feature columns: {missing_cols}",
                sys,
            )

        prepared_df = tower_features[self.feature_columns].copy()
        for column in self.feature_columns:
            prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")

        if self.fill_values:
            prepared_df = prepared_df.fillna(self.fill_values)

        return self.scaler.transform(prepared_df)

    def predict(self, tower_features: pd.DataFrame):
        try:
            prepared_features = self._prepare_features(tower_features)
            return self.model.predict(prepared_features)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, tower_features: pd.DataFrame):
        try:
            if not hasattr(self.model, "predict_proba"):
                raise CustomException(
                    "The trained NetPlanner AI model does not support probability scores.",
                    sys,
                )

            prepared_features = self._prepare_features(tower_features)
            return self.model.predict_proba(prepared_features)[:, 1]
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        upload_speed_mbps: float,
        distance_to_tower_km: float,
        num_measurements: int,
    ):
        self.upload_speed_mbps = upload_speed_mbps
        self.distance_to_tower_km = distance_to_tower_km
        self.num_measurements = num_measurements

    def get_data_as_data_frame(self):
        try:
            tower_metrics = {
                "upload_speed_mbps": [self.upload_speed_mbps],
                "distance_to_tower_km": [self.distance_to_tower_km],
                "num_measurements": [self.num_measurements],
            }

            return pd.DataFrame(tower_metrics)
        except Exception as e:
            raise CustomException(e, sys)
