import pytest
from .dummy_model import MathModel, Affine, Quadratic

@pytest.fixture
def base_model() -> MathModel:
    """Fixture to provide a fresh model for each test."""
    match = Affine()
    res = Quadratic()
    return MathModel(affine=match, quadratic=res)