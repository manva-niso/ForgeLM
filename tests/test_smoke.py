def test_package_imports():
    import forger  # noqa: F401


def test_torch_cpu_available():
    import torch

    assert torch.__version__ >= "2.4"