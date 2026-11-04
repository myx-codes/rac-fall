from racfall.env import diagnostic


def test_diagnostic_minimal():
    result = diagnostic()
    assert isinstance(result["apple_mps_available"], bool)
    assert "platform" in result
    assert "python" in result
