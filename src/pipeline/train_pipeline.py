import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("Starting NetPlanner AI training pipeline")

            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(train_path, test_path)
            )

            model_trainer = ModelTrainer()
            metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(
                "NetPlanner AI training pipeline completed with best model %s",
                metrics["best_model_name"],
            )

            return {
                "train_data_path": train_path,
                "test_data_path": test_path,
                "preprocessor_path": preprocessor_path,
                "model_metrics": metrics,
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    print(pipeline.run_pipeline())
