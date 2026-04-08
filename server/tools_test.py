from .tools import ToolSpec
import pytest
import sys


@pytest.fixture
def tools():
    return ToolSpec()

def test_get_department(tools):
    result = tools.get_department("chest pain, heart attack, hypertension")
    assert result.message =='The proper department for the patient is: Cardiology', "get_department not returning the same result"

# Test for get_opd_doctor method
def test_get_opd_doctor(tools):
    result = tools.get_opd_doctor("Cardiology","Friday")
    assert result is not None, "Expected a non-None result for get_opd_doctor"
    # Add more specific assertions based on the expected behavior of get_opd_doctor

def test_get_appointment(tools):
    result = tools.get_appointment(doctor_id=1, patient_id=1001)
    assert result is not None, "Expected a non-None result for get_appointment"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))