import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator
from production.scripts import CombinedAttributesAdder

# Unit test for CombinedAttributesAdder class
def test_combined_attributes_adder():
    # Create an instance of the CombinedAttributesAdder class
    adder = CombinedAttributesAdder(0, 1, 2, 3)

    # Test the fit method
    result = adder.fit(None)
    assert result is adder

    # Define input data
    X = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    # Test the transform method with add_bedrooms_per_room=True
    expected_output = np.array([
        [1, 2, 3, 4, 0.25, 0.75],
        [5, 6, 7, 8, 0.625, 0.875]
    ])
    transformed_output = adder.transform(X)
    assert np.array_equal(transformed_output, expected_output)

    # Test the transform method with add_bedrooms_per_room=False
    adder = CombinedAttributesAdder(0, 1, 2, 3, add_bedrooms_per_room=False)
    expected_output = np.array([
        [1, 2, 3, 4, 0.25],
        [5, 6, 7, 8, 0.625]
    ])
    transformed_output = adder.transform(X)
    assert np.array_equal(transformed_output, expected_output)

    # Test the inverse_transform method with add_bedrooms_per_room=True
    expected_output = X
    inverse_transformed_output = adder.inverse_transform(transformed_output)
    assert np.array_equal(inverse_transformed_output, expected_output)

    # Test the inverse_transform method with add_bedrooms_per_room=False
    adder = CombinedAttributesAdder(0, 1, 2, 3, add_bedrooms_per_room=False)
    expected_output = X
    inverse_transformed_output = adder.inverse_transform(transformed_output)
    assert np.array_equal(inverse_transformed_output, expected_output)

# Run the test
pytest.main()
