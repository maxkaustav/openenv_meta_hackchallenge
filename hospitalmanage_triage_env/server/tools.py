from typing import Dict,List, Any, Optional

class ToolSpec:
    def __init__(self): pass

    def get_department(self, conditions: str) -> str:

        return sommething
    
    def get_doctor_tool(self,department : str):
        return {"tool": "doctor", "department": department}

    def assign_doctor(self, doctor: str, patient_id: str) -> Dict[str, str]:
        return {"action": "assign_doctor", "doctor": doctor, "patient_id": patient_id}