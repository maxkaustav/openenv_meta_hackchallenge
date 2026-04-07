"""
REBOOK Task — Healthcare Appointment Scheduling

User request: "I need to rebook my appointment with Dr. Priya Patel"
Expected path:
  check_availability("Dr. Priya Patel") → book_appointment("Dr. Priya Patel", <any valid slot>)

Ground truth:
  department : Dermatology
  doctor     : Dr. Priya Patel
"""

try:
    from healthcare_scheduling.server.environment import HealthcareAppointmentEnvironment
    from healthcare_scheduling.models import AppointmentAction
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from server.environment import HealthcareAppointmentEnvironment  # type: ignore
    from models import AppointmentAction  # type: ignore


TASK_ID = "rebook_priya_patel"
USER_REQUEST = "I need to rebook my appointment with Dr. Priya Patel"
CORRECT_DEPARTMENT = "Dermatology"
CORRECT_DOCTOR = "Dr. Priya Patel"


def make_env() -> HealthcareAppointmentEnvironment:
    """Return a fresh environment pre-loaded with the rebook task."""
    env = HealthcareAppointmentEnvironment()
    env.reset(
        user_request=USER_REQUEST,
        correct_department=CORRECT_DEPARTMENT,
        correct_doctor=CORRECT_DOCTOR,
    )
    return env


def get_task_config() -> dict:
    """Return the task configuration dictionary used by agents and graders."""
    return {
        "task_id": TASK_ID,
        "difficulty": "rebook",
        "user_request": USER_REQUEST,
        "correct_department": CORRECT_DEPARTMENT,
        "correct_doctor": CORRECT_DOCTOR,
        "description": (
            "The user explicitly names the doctor to rebook with. "
            "The agent should skip get_departments and get_doctors, "
            "and directly call check_availability to start."
        ),
        "max_steps": 10,
        "expected_min_steps": 2,
        "is_rebook": True,
        "hints": [
            "Start directly with check_availability",
            "Book with Dr. Priya Patel",
        ],
    }


if __name__ == "__main__":
    from tasks.graders import grade_episode

    env = HealthcareAppointmentEnvironment()
    obs = env.reset(
        user_request=USER_REQUEST,
        correct_department=CORRECT_DEPARTMENT,
        correct_doctor=CORRECT_DOCTOR,
    )
    print(f"[REBOOK] Episode started: {obs.message}")

    # Simulate an oracle agent following the optimal path
    actions = [
        AppointmentAction(
            tool="check_availability", parameters={"doctor": "Dr. Priya Patel"}
        ),
        AppointmentAction(
            tool="book_appointment",
            parameters={"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"},
        ),
    ]

    for action in actions:
        obs = env.step(action)
        print(f"  [{action.tool}] reward={obs.reward:.2f} | {obs.message}")
        if obs.done:
            break

    score = grade_episode(env.state, get_task_config())
    print(f"\n[REBOOK] Final Score: {score:.3f}")
