import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize("name", ["simple", "stronger"])
def test_build_model_output_shape(name):
    model = build_model(name)
    X = torch.randn(2, 3, 32, 32)
    y = model(X)
    assert y.shape == (2, 10)


def test_build_model_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("not-a-real-model")
