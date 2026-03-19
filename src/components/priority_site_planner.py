import os
import sys
from pathlib import Path

import pandas as pd

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PrioritySitePlanner:
    def __init__(self):
        self.preprocessor_path = Path("artifacts") / "preprocessor.pkl"
        self.model_path = Path("artifacts") / "model.pkl"

    @staticmethod
    def _mode_or_missing(series):
        non_null = series.dropna()
        if non_null.empty:
            return "missing"

        mode = non_null.mode()
        if not mode.empty:
            return str(mode.iloc[0])

        return str(non_null.iloc[0])

    def build_scored_tower_dataset(self, raw_data_path):
        try:
            raw_df = pd.read_csv(raw_data_path)

            transformation = DataTransformation()
            tower_stats = transformation._build_tower_dataset(raw_df)

            location_df = raw_df[["tower_id", "latitude", "longitude"]].copy()
            for column in ["latitude", "longitude"]:
                location_df[column] = pd.to_numeric(location_df[column], errors="coerce")

            tower_locations = (
                location_df.groupby("tower_id", as_index=False)[["latitude", "longitude"]]
                .mean()
            )
            tower_stats = tower_stats.merge(tower_locations, on="tower_id", how="left")

            metadata_columns = [
                column
                for column in [
                    "city",
                    "operator",
                    "network_type",
                    "coverage_quality",
                    "indoor_outdoor",
                ]
                if column in raw_df.columns
            ]
            if metadata_columns:
                tower_metadata = raw_df.groupby("tower_id", as_index=False)[metadata_columns].agg(
                    self._mode_or_missing
                )
                tower_stats = tower_stats.merge(tower_metadata, on="tower_id", how="left")

            preprocessor = load_object(str(self.preprocessor_path))
            model = load_object(str(self.model_path))

            features = tower_stats[preprocessor["feature_columns"]].copy()
            for column in preprocessor["feature_columns"]:
                features[column] = pd.to_numeric(features[column], errors="coerce")

            features = features.fillna(preprocessor.get("fill_values", {}))
            scaled_features = preprocessor["scaler"].transform(features)

            if hasattr(model, "predict_proba"):
                tower_stats["upgrade_probability"] = model.predict_proba(scaled_features)[:, 1]
            else:
                tower_stats["upgrade_probability"] = model.predict(scaled_features)

            tower_stats["risk_level"] = tower_stats["upgrade_probability"].apply(
                lambda probability: (
                    "High"
                    if probability >= 0.70
                    else "Medium" if probability >= 0.40 else "Low"
                )
            )

            return tower_stats.dropna(subset=["latitude", "longitude"])
        except Exception as e:
            raise CustomException(e, sys)

    def save_upgrade_map(self, tower_stats, output_path=None, max_markers=1000):
        try:
            import folium
            from folium.plugins import MarkerCluster
        except ImportError as e:
            raise CustomException(
                "folium is required to save the NetPlanner AI HTML map.",
                sys,
            ) from e

        map_rows = (
            tower_stats.sort_values("upgrade_probability", ascending=False)
            .head(max_markers)
            .copy()
        )

        if map_rows.empty:
            raise CustomException("No tower data available to build the HTML map.", sys)

        map_center = [map_rows["latitude"].mean(), map_rows["longitude"].mean()]
        tower_map = folium.Map(location=map_center, zoom_start=6)
        marker_cluster = MarkerCluster().add_to(tower_map)

        for _, row in map_rows.iterrows():
            if row["risk_level"] == "High":
                color = "red"
            elif row["risk_level"] == "Medium":
                color = "orange"
            else:
                color = "green"

            popup_text = f"""
            <b>Tower ID:</b> {row['tower_id']}<br>
            <b>Upgrade Probability:</b> {row['upgrade_probability']:.2f}<br>
            <b>Risk Level:</b> {row['risk_level']}<br>
            <b>Upload Speed:</b> {row['upload_speed_mbps']:.2f}<br>
            <b>Distance to Tower:</b> {row['distance_to_tower_km']:.2f} km<br>
            <b>Measurements:</b> {int(row['num_measurements'])}
            """

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
            ).add_to(marker_cluster)

        output_path = Path(output_path) if output_path else Path.cwd() / "tower_upgrade_map.html"
        tower_map.save(str(output_path))

        logging.info("Saved NetPlanner AI upgrade map to %s", output_path.resolve())
        return str(output_path.resolve())

    def save_priority_sites_csv(self, tower_stats, output_path=None):
        ranked_sites = tower_stats.sort_values(
            ["upgrade_probability", "num_measurements"],
            ascending=[False, False],
        ).copy()
        ranked_sites["priority_rank"] = range(1, len(ranked_sites) + 1)
        ranked_sites["priority_level"] = ranked_sites["risk_level"]

        selected_columns = [
            "priority_rank",
            "priority_level",
            "tower_id",
            "upgrade_probability",
            "upgrade_needed",
            "latitude",
            "longitude",
            "city",
            "operator",
            "network_type",
            "coverage_quality",
            "indoor_outdoor",
            "signal_strength_dbm",
            "download_speed_mbps",
            "upload_speed_mbps",
            "latency_ms",
            "distance_to_tower_km",
            "num_measurements",
        ]
        export_columns = [column for column in selected_columns if column in ranked_sites.columns]

        output_path = Path(output_path) if output_path else Path.cwd() / "priority_sites.csv"
        ranked_sites[export_columns].to_csv(output_path, index=False)

        logging.info("Saved NetPlanner AI priority sites CSV to %s", output_path.resolve())
        return str(output_path.resolve())
