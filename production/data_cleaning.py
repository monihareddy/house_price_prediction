"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning
)
from scripts import binned_median_house_value


@register_processor("data_cleaning", "housing")
def clean_housing_table(context, params):
    """Clean the ``housing`` data table.

    The table containts the housing price data and has information on the latitude, longitude,
    population, bedrooms etc.
    """

    input_dataset = "raw/housing"
    output_dataset = "cleaned/housing"

    # load dataset
    housing_df = load_dataset(context, input_dataset)

    # list of columns that we want string cleaning op to be performed on.
    str_cols = list(
    set(housing_df.select_dtypes('object').columns.to_list())
    )

    housing_df_clean = (
        housing_df
        # set dtypes
        .change_type(['housing_median_age', 'total_rooms', 'population', 'households'], np.int64)

        # clean string columns
        .transform_columns(str_cols, string_cleaning, elementwise=False)

        # clean column names
        .clean_names(case_type='snake')
    )

    # save dataset
    save_dataset(context, housing_df_clean, output_dataset)
    return housing_df_clean


@register_processor("data_cleaning", "train_test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    housing_df_processed = load_dataset(context, input_dataset)

    # split the data
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_processed, splitter, by=binned_median_house_value
    )

    # split train dataset into features and target
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)

    # Configuring mlflow
    mlflow.set_experiment("House Price Prediction")
    with mlflow.start_run(run_name="Data Cleaning"):
        mlflow.log_params(params)
