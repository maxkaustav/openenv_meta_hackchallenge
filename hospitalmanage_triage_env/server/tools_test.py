from tools import ToolSpec
import pytest
import sys

@pytest.fixture
def tools():
    return ToolSpec()

# Test for get_opd_doctor method
def test_get_opd_doctor(tools):
    result = tools.get_opd_doctor("Cardiology")
    assert result is not None, "Expected a non-None result for get_opd_doctor"
    # Add more specific assertions based on the expected behavior of get_opd_doctor


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))