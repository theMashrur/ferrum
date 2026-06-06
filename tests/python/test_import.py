import importlib


def test_import_package() -> None:
    module = importlib.import_module("ferrum")
    assert module.__name__ == "ferrum"
