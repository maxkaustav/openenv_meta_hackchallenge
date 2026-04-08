from client import HospitalmanageTriageEnv
from openenv.core.env_server.mcp_types import CallToolAction

# with HospitalmanageTriageEnv(base_url="http://localhost:8001").sync() as env:
    
    ### testing check reset state
    # env.reset()
    # env.state() 
    # # OUTPUT: episode_id='5542f934-3c32-484f-8606-0a704ad6fe85' step_count=0 patient_id='1294' doctor_id='' department='' tool_call_sequence=[] tool_state_step=0
    
    # ### testing check reset state
    # env.reset(patient_id='8118', tool_call_sequence=['get_department'])
    # env.state()
    # # OUTPUT: episode_id='5542f934-3c32-484f-8606-0a704ad6fe85' step_count=0 patient_id='8118' doctor_id='' department='' tool_call_sequence=['get_department'] tool_state_step=0
    
    ### check if tool call works
    # env.reset(patient_id='8118', tool_call_sequence=['get_department'],output_sequence=['The proper department for the patient is: Neurology'])
    # result = env.step(CallToolAction(tool_name="get_department", arguments={"conditions": "headache"}))
    # print(result.observation)  # No more timeout :)
    # env.state()
    #### check if invalid tool call
    # env.reset(patient_id='8118', tool_call_sequence=['get_department'])
    # result = env.step(CallToolAction(tool_name="get_room", arguments={"conditions": "headache"}))
    # print(result.observation)  # No more timeout :)
    # env.state()
    # check for edge tool errors

with HospitalmanageTriageEnv(base_url="http://localhost:8001").sync() as env:
    ### check for appointment booking
    # env.reset(patient_id='8118', tool_call_sequence=['make_appointment'],output_sequence=['Appointment made successfully.'])
    # doctor_id: int = 1234
    # patient_id: int = 8118
    # doctor_name: str = "Dr. Smith"
    # patient_name: str = "John Doe"
    # result = env.step(CallToolAction(tool_name="make_appointment", arguments={"doctor_id": doctor_id, "patient_id": patient_id, "doctor_name": doctor_name, "patient_name": patient_name}))
    # print(result.observation)  # No more timeout :)
    # env.state()

    ### opd +make appointment
    # env.reset(patient_id='8118', tool_call_sequence=['make_appointment'],output_sequence=['Appointment made successfully.'])
    # env.reset(patient_id=8119, tool_call_sequence=['get_opd_doctor','make_appointment'],output_sequence=['The proper doctor for the patient is: Dr. Maria Lopez with doctor ID: 3','Appointment made successfully.'])
    # department = "Neurology"
    # doctor_id: int = 3
    # patient_id: int = 8119
    # doctor_name: str = "Dr. Maria Lopez"
    # patient_name: str = "John Doe"
    # result = env.step(CallToolAction(tool_name="get_opd_doctor", arguments={"department" :department}))
    # print(result.observation)  # No more timeout :)
    # result = env.step(CallToolAction(tool_name="make_appointment", arguments={"doctor_id": doctor_id, "patient_id": patient_id, "doctor_name": doctor_name, "patient_name": patient_name}))
    # print(result.observation)
    
    ### find department + find doctor + make appointment
    env.reset(patient_id=8144 , tool_call_sequence=['get_department','get_opd_doctor','make_appointment'],
              output_sequence=['The proper department for the patient is: Neurology',
                               'The proper doctor for the patient is:',
                               'Appointment made successfully.'])
    condition = "headache"
    department = "Neurology"
    doctor_id: int = 3
    patient_id: int = 8144
    doctor_name: str = "Dr. Maria Lopez"
    patient_name: str = "John Doe Trump"
    result = env.step(CallToolAction(tool_name="get_department", arguments={"conditions": condition}))
    print(result.observation)  # No more timeout :)
    result = env.step(CallToolAction(tool_name="get_opd_doctor", arguments={"department" :department}))
    print(result.observation)  # No more timeout :)
    result = env.step(CallToolAction(tool_name="make_appointment", arguments={"doctor_id": doctor_id, "patient_id": patient_id, "doctor_name": doctor_name, "patient_name": patient_name}))
    print(result.observation)