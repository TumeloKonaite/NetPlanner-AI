import json
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_metrics_file_path: str = os.path.join("artifacts", "model_metrics.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test arrays into X/y components")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                raise CustomException(
                    "Both train and test data must contain at least two target classes.",
                    sys,
                )

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42,
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 5.0],
                },
                "Decision Tree": {
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
            }

            model_report, trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise CustomException("Model evaluation did not return results.", sys)

            best_model_name = max(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]

            if hasattr(best_model, "predict_proba"):
                y_scores = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_scores = best_model.decision_function(X_test)
            else:
                y_scores = best_model.predict(X_test)

            y_pred = best_model.predict(X_test)

            metrics = {
                "best_model_name": best_model_name,
                "roc_auc": float(roc_auc_score(y_test, y_scores)),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "all_model_scores": {
                    name: float(score) for name, score in model_report.items()
                },
            }

            logging.info(
                "Best telecom upgrade model: %s with ROC-AUC %.4f",
                metrics["best_model_name"],
                metrics["roc_auc"],
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            with open(self.model_trainer_config.model_metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=2)

            return metrics
        except Exception as e:
            raise CustomException(e, sys)
