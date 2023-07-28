"""Processors for the model training step of the worklow."""
import logging
import os.path as op
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model_gen", "train_model")
def train_model(context, params):
    """Train a regression model."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    curated_columns = load_pipeline(op.join(artifacts_folder, "curated_columns.joblib"))
    features_columns = load_pipeline(op.join(artifacts_folder, "features_columns.joblib"))
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))

    # sample data if needed. Useful for debugging/profiling purposes.
    sample_frac = params.get("sampling_fraction", None)
    if sample_frac is not None:
        logger.warn(f"The data has been sample by fraction: {sample_frac}")
        sample_X = train_X.sample(frac=sample_frac, random_state=context.random_seed)
    else:
        sample_X = train_X
    sample_y = train_y.loc[sample_X.index]

    # transform the training data
    train_X = pd.DataFrame(
        features_transformer.fit_transform(train_X),
        columns=features_columns,
    )
    train_X = train_X[curated_columns]

    # create training pipeline
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )

    # Configuring mlflow
    mlflow.set_experiment("House Price Prediction")
    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_params(params)
        mlflow.log_param("Input Features", train_X.columns)
        mlflow.log_param("Input Target", train_y.columns)

        # Log the model
        mlflow.sklearn.log_model(reg_ppln_ols, "reg_ppln_ols")

        # Log the artifacts
        mlflow.log_artifact(op.join(artifacts_folder, "curated_columns.joblib"))
        mlflow.log_artifact(op.join(artifacts_folder, "features.joblib"))
        mlflow.log_artifact(op.abspath(op.join(artifacts_folder, "train_pipeline.joblib")))
