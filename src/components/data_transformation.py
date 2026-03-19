import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    schema_file_path: str = os.path.join("artifacts", "schema.json")
    feature_columns_file_path: str = os.path.join(
        "artifacts", "feature_columns.json"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_column_name = "upgrade_needed"
        self.aggregation_columns = [
            "signal_strength_dbm",
            "download_speed_mbps",
            "upload_speed_mbps",
            "latency_ms",
            "distance_to_tower_km",
        ]
        self.feature_columns = [
            "upload_speed_mbps",
            "distance_to_tower_km",
            "num_measurements",
        ]
        self.required_raw_columns = ["tower_id", *self.aggregation_columns]

    def _build_tower_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.required_raw_columns if col not in df.columns]
        if missing_cols:
            raise CustomException(
                f"Missing required telecom columns: {missing_cols}",
                sys,
            )

        working_df = df[self.required_raw_columns].copy()
        for column in self.aggregation_columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

        tower_stats = (
            working_df.groupby("tower_id", as_index=False)[self.aggregation_columns]
            .mean()
            .merge(
                working_df.groupby("tower_id")
                .size()
                .reset_index(name="num_measurements"),
                on="tower_id",
                how="left",
            )
        )

        tower_stats[self.target_column_name] = (
            (tower_stats["signal_strength_dbm"] < -90)
            | (tower_stats["download_speed_mbps"] < 20)
            | (tower_stats["latency_ms"] > 100)
        ).astype(int)

        logging.info(
            "Built tower-level dataset with shape %s and target distribution %s",
            tower_stats.shape,
            tower_stats[self.target_column_name].value_counts().to_dict(),
        )

        return tower_stats

    def _prepare_features_and_target(self, df: pd.DataFrame):
        missing_feature_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_feature_cols:
            raise CustomException(
                f"Missing required columns for transformation: {missing_feature_cols}",
                sys,
            )

        X = df[self.feature_columns].copy()
        y = pd.to_numeric(df[self.target_column_name], errors="coerce").fillna(0).astype(int)

        return X, y

    def initiate_data_transformation(self, train_path, test_path):
        try:
            raw_train_df = pd.read_csv(train_path)
            raw_test_df = pd.read_csv(test_path)

            logging.info(
                "Loaded raw telecom train split with shape %s and test split with shape %s",
                raw_train_df.shape,
                raw_test_df.shape,
            )

            train_tower_df = self._build_tower_dataset(raw_train_df)
            test_tower_df = self._build_tower_dataset(raw_test_df)

            X_train, y_train = self._prepare_features_and_target(train_tower_df)
            X_test, y_test = self._prepare_features_and_target(test_tower_df)

            fill_values = X_train.median()
            X_train = X_train.fillna(fill_values)
            X_test = X_test.fillna(fill_values)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            logging.info(
                "Notebook-style transformation prepared with train shape %s and test shape %s",
                X_train.shape,
                X_test.shape,
            )

            preprocessor = {
                "scaler": scaler,
                "fill_values": fill_values.to_dict(),
                "feature_columns": self.feature_columns,
                "target_column": self.target_column_name,
            }
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            os.makedirs(
                os.path.dirname(self.data_transformation_config.schema_file_path),
                exist_ok=True,
            )

            schema = {
                "aggregation_level": "tower_id",
                "aggregation_columns": self.aggregation_columns,
                "feature_columns": self.feature_columns,
                "target_column": self.target_column_name,
            }
            with open(self.data_transformation_config.schema_file_path, "w") as file_obj:
                json.dump(schema, file_obj, indent=2)

            with open(
                self.data_transformation_config.feature_columns_file_path, "w"
            ) as file_obj:
                json.dump(self.feature_columns, file_obj, indent=2)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
