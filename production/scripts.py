"""Module for listing down additional custom functions required for production."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# def binned_selling_price(df):
#     """Bin the selling price column using quantiles."""
#     return pd.qcut(df["unit_price"], q=10)


def binned_median_house_value(df):
    """Bin the housing price column using quantiles."""
    return pd.qcut(df["median_house_value"], q=10)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    A transformer class that computes additional attributes and combines them
    with the original dataset.

    Parameters
    ----------
    rooms_ix : int
        The index of column containing the total number of rooms in dataset.
    bedrooms_ix : int
        The index of column containing the total number of bedrooms in dataset.
    population_ix : int
        The index of the column containing the total population in the dataset.
    households_ix : int
        The index of column containing total number of households in dataset.
    add_bedrooms_per_room : bool, default=True
        A flag indicating whether to add bedrooms_per_room attribute or not.

    Attributes
    ----------
    rooms_ix : int
        See the 'Parameters' section.
    bedrooms_ix : int
        See the 'Parameters' section.
    population_ix : int
        See the 'Parameters' section.
    households_ix : int
        See the 'Parameters' section.
    add_bedrooms_per_room : bool
        See the 'Parameters' section.
    """

    def __init__(self, rooms_ix, bedrooms_ix, population_ix, households_ix, add_bedrooms_per_room=True):
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        y : None
            Unused. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self  # nothing else to do

    def transform(self, X):
        """
        Transform the input data by adding the computed attributes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features + n_new_features)
            The transformed data with additional features.
        """
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def inverse_transform(self, X_transformed):
        """
        Revert the transformation applied by `transform`.

        Parameters
        ----------
        X_transformed : array-like, shape (n_samples, n_features + n_new_features)
            The transformed data.

        Returns
        -------
        X_original : array, shape (n_samples, n_features)
            The original data before transformation.
        """
        X_original = X_transformed.copy()

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X_transformed[:, -1]
            X_original[:, self.bedrooms_ix] = X_transformed[:, self.rooms_ix] * bedrooms_per_room
            X_transformed = np.delete(X_transformed, -1, axis=1)

        population_per_household = X_transformed[:, -1]
        X_original[:, self.population_ix] = X_transformed[:, self.households_ix] * population_per_household
        X_transformed = np.delete(X_transformed, -1, axis=1)

        rooms_per_household = X_transformed[:, -1]
        X_original[:, self.rooms_ix] = X_transformed[:, self.households_ix] * rooms_per_household
        X_transformed = np.delete(X_transformed, -1, axis=1)

        # Remove the additional columns from the X_original array
        X_original = np.delete(X_original, -1, axis=1)
        X_original = np.delete(X_original, -1, axis=1)
        if self.add_bedrooms_per_room:
            X_original = np.delete(X_original, -1, axis=1)

        return X_original