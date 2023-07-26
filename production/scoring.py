"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import pandas as pd


from ta_lib.core.api import (get_dataframe,
                             get_feature_names_from_column_transformer,
                             get_package_path, hash_object, load_dataset,
                             load_pipeline, register_processor, save_dataset, DEFAULT_ARTIFACTS_PATH)


@register_processor("model_eval", "score_model")
def score_model(context, params):
    """Score a pre-trained model."""

    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_columns = load_pipeline(op.join(artifacts_folder, "features_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    # transform the test dataset
    test_X = pd.DataFrame(
        features_transformer.transform(test_X),
        columns=features_columns,
    )
    test_X = test_X[curated_columns]

    # make a prediction
    test_X["yhat"] = model_pipeline.predict(test_X)

    # store the predictions for any further processing.
    save_dataset(context, test_X, output_ds)
            
    # evaluationg the model
    final_mae = mean_absolute_error(test_y, test_X["yhat"])
    final_mse = mean_squared_error(test_y, test_X["yhat"])
    final_rmse = np.sqrt(final_mse)

    # Configuring mlflow
    mlflow.set_experiment("House Price Prediction")
    with mlflow.start_run(run_name="Model Evaluation"):
        mlflow.log_params(params)
        mlflow.log_metric("Model - MAE", final_mae)
        mlflow.log_metric("Model - MSE", final_mse)
        mlflow.log_metric("Model - RMSE", final_rmse)
