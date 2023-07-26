"""Module for listing down additional custom functions required for the notebooks."""

import pandas as pd


def binned_selling_price(df):
    """Bin the selling price column using quantiles."""
    return pd.qcut(df["unit_price"], q=10)


def binned_housing_price(df):
    """Bin the median house value column using quantiles"""
    return pd.qcut(df["median_house_value"], q=10)