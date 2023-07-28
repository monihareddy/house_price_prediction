"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scripts import CombinedAttributesAdder

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier

logger = logging.getLogger(__name__)




@register_processor("feat_engg", "transform_features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )

    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a
    # ``OnehotEncoder`` and a ``SimpleImputer`` to first encode the
    # categorical variable into a numerical values and then impute any missing
    # values using ``most_frequent`` strategy.

    col_names = "total_rooms", "total_bedrooms", "population", "households"
    rooms_ix, bedrooms_ix, population_ix, households_ix = [
        train_X.columns.get_loc(c) for c in col_names]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder(rooms_ix, bedrooms_ix, population_ix, households_ix)),
        ('std_scaler', StandardScaler()),
    ])
    features_transformer = ColumnTransformer([
        ("num", num_pipeline, num_columns),
        ("cat", OneHotEncoder(sparse=False, handle_unknown='ignore', drop='first'), list(set(cat_columns))),
    ])
    _ = features_transformer.fit_transform(train_X)

    # Combine the column names
    # Add the new column names created by the CombinedAttributesAdder
    new_columns = ['rooms_per_household', 'population_per_household']
    if num_pipeline.named_steps['attribs_adder'].add_bedrooms_per_room:
        new_columns.append('bedrooms_per_room')
    cat_columns = features_transformer.named_transformers_['cat'].get_feature_names_out(cat_columns)
    all_columns = np.concatenate((num_columns.to_list() + new_columns, cat_columns))

    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required.
    train_X = pd.DataFrame(
        features_transformer.fit_transform(train_X),
        columns=all_columns
    )

    # Note: we can create a transformer/feature selector that simply drops
    # a specified set of columns. But, we don't do that here to illustrate
    # what to do when transformations don't cleanly fall into the sklearn
    # pattern.
    curated_columns = list(
        set(train_X.columns.to_list()) - set([])
    )

    # saving the list of relevant columns and the pipeline.
    save_pipeline(
        curated_columns, op.abspath(op.join(artifacts_folder, "curated_columns.joblib"))
    )
    save_pipeline(
        all_columns, op.abspath(op.join(artifacts_folder, "features_columns.joblib"))
    )
    save_pipeline(
        features_transformer, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )

    # Configuring mlflow
    mlflow.set_experiment("House Price Prediction")
    with mlflow.start_run(run_name="Feature Engineering"):
        mlflow.log_param("outliers_method", params["outliers"]["method"])
        mlflow.log_param("outliers_drop", params["outliers"]["drop"])

        # metric
        mlflow.log_metric("num_features", train_X.shape[1])

        # Log the artifacts
        #mlflow.log_artifact(input_features_ds, "input_features_ds")
        #mlflow.log_artifact(input_target_ds, "input_target_ds")

        # Log the output data
        mlflow.log_artifact(op.join(artifacts_folder, "curated_columns.joblib"))
        mlflow.log_artifact(op.join(artifacts_folder, "features.joblib"))
