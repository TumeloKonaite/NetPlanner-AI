from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline


def test_train_pipeline():
    result = TrainPipeline().run_pipeline()

    assert "model_metrics" in result
    assert "best_model_name" in result["model_metrics"]

    return result


def test_predict_pipeline():
    sample_data = CustomData(
        upload_speed_mbps=8.5,
        distance_to_tower_km=5.2,
        num_measurements=3,
    ).get_data_as_data_frame()

    predictor = PredictPipeline()
    prediction = predictor.predict(sample_data)
    probability = predictor.predict_proba(sample_data)

    assert len(prediction) == 1
    assert len(probability) == 1

    return {
        "prediction": prediction.tolist(),
        "probability": probability.tolist(),
    }


if __name__ == "__main__":
    train_result = test_train_pipeline()
    predict_result = test_predict_pipeline()

    print("Train pipeline OK")
    print(train_result["model_metrics"])
    print("Predict pipeline OK")
    print(predict_result)
