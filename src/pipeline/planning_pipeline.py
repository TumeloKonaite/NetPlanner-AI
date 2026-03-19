import sys
from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.priority_site_planner import PrioritySitePlanner
from src.exception import CustomException
from src.logger import logging
from src.pipeline.train_pipeline import TrainPipeline


class PlanningPipeline:
    def __init__(self):
        self.planner = PrioritySitePlanner()

    def _ensure_scoring_artifacts(self):
        try:
            data_ingestion = DataIngestion()
            raw_data_path = data_ingestion.ingestion_config.raw_data_path
            preprocessor_path = Path("artifacts") / "preprocessor.pkl"
            model_path = Path("artifacts") / "model.pkl"

            if not raw_data_path.exists():
                logging.info(
                    "Raw measurement data not found. Running ingestion before generating planning outputs."
                )
                data_ingestion.initiate_data_ingestion()

            if not preprocessor_path.exists() or not model_path.exists():
                logging.info(
                    "Model artifacts missing. Running training pipeline before generating planning outputs."
                )
                TrainPipeline().run_pipeline()

            return raw_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def build_priority_sites(self):
        try:
            raw_data_path = self._ensure_scoring_artifacts()
            return self.planner.build_scored_tower_dataset(raw_data_path)
        except Exception as e:
            raise CustomException(e, sys)

    def generate_upgrade_map(self, output_path=None, max_map_markers=1000):
        try:
            tower_stats = self.build_priority_sites()
            return self.planner.save_upgrade_map(
                tower_stats=tower_stats,
                output_path=output_path,
                max_markers=max_map_markers,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def generate_priority_sites_csv(self, output_path=None):
        try:
            tower_stats = self.build_priority_sites()
            return self.planner.save_priority_sites_csv(
                tower_stats=tower_stats,
                output_path=output_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, map_output_path=None, priority_csv_output_path=None, max_map_markers=1000):
        try:
            tower_stats = self.build_priority_sites()
            map_path = self.planner.save_upgrade_map(
                tower_stats=tower_stats,
                output_path=map_output_path,
                max_markers=max_map_markers,
            )
            csv_path = self.planner.save_priority_sites_csv(
                tower_stats=tower_stats,
                output_path=priority_csv_output_path,
            )

            return {
                "map_output_path": map_path,
                "priority_sites_csv_path": csv_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
