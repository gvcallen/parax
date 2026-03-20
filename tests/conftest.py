import pytest
from tests.dummy_model import CircuitModel, MatchingNetwork, Resonator

@pytest.fixture
def base_model() -> CircuitModel:
    """Fixture to provide a fresh model for each test."""
    match = MatchingNetwork()
    res = Resonator()
    return CircuitModel(input_match=match, resonator=res)