import adashell_renderer


def test_version_string() -> None:
    assert isinstance(adashell_renderer.__version__, str)
