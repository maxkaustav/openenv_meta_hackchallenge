---
title: Hospital Reception Support ENV
sdk: docker
---

# Healthcare Appointment Scheduling OpenEnv Project

## Overview

This project provides a robust, production-grade Reinforcement Learning environment for healthcare appointment scheduling, compliant with the OpenEnv specification.
An AI agent receives a natural-language patient request, identifies the correct medical department and doctor, and books an appointment using a set of structured tool calls.

## Features

- **OpenEnv Compliant**: Fully implements the OpenEnv standard with `models.py`, `client.py`, and `server/` architecture.
- **Interactive Streamlit UI**: Includes a feature-rich Streamlit frontend (`streamlit_app.py`) for observing the agent taking actions and responding to user clarifications.
- **Dynamic Tool Dispatch**: The agent interacts with the environment through five core tools:
  - `get_departments()`
  - `get_doctors(department)`
  - `check_availability(doctor)`
  - `book_appointment(doctor, slot)`
  - `ask_user_clarification(question)`
- **Advanced State Management**: Tracks conversations, identified departments, doctor selections, intent parsing (e.g., rebooking requests), and multi-turn clarifications.
- **Supported Baselines**: Contains a built-in Groq-powered LLM agent baseline and Oracle tools for deterministic evaluations.

## Environment Details

The environment is structured as a Markov Decision Process where each interaction loop is composed of:

1. **Observation**: Status of the request, last tool result, and reward details.
2. **Action**: The tool selected by the agent.
3. **Reward**: Calculated at every step to guide the agent toward correct scheduling.
4. **Done Condition**: The episode stops when an appointment is booked or after a maximum of 10 steps.

The dataset includes automated matching of patient symptoms to the correct departments and doctors, acting as a ground-truth verifier.

## Reward System

The environment offers dense rewards at every step to ensure proper training gradients and strict verification:

- **Correct Department (+1.0)**: Correctly retrieving the right department by matching symptoms.
- **Incorrect Department (-0.50)**: Penalty for querying the wrong department.
- **Correct Doctor (+1.0)**: Checking availability for the correct doctor within the specific department.
- **Incorrect/Wrong Specialization (-0.50)**: Selecting a doctor outside the proper bounds of the requested symptoms.
- **Successful Booking (+1.0)**: For finalizing the booking with the correct specifications.
- **Invalid Actions (-0.50)**: Penalty for failing to conform to predefined schemas or exceeding max steps.

Certain workflows, such as "Rebooking", allow the agent to bypass initial steps without penalty if the user explicitly references an existing doctor in their request.

## Setup Instructions

### Prerequisites

- Python 3.10+
- `uv` package manager (recommended for handling the lock file) or `pip`.
- Groq API Key (if you intend to run the LLM Baseline agent).

### Local Installation

Navigate to the root directory and install dependencies:

```bash
pip install -e .
pip install -e '.[dev]'
```

Alternatively, if `uv` is installed, you can utilize the `uv.lock` file directly.

## How to Run

### Running the API Server (OpenEnv Backend)

To start the OpenEnv FastAPI backend daemon locally:

```bash
uv run server
```

Or use the standard Uvicorn command if installed globally:

```bash
python -m uvicorn server.app:app --port 8000
```

### Running the Interactive UI (Streamlit)

To test the environment interactively or visualize agent interactions, start the UI:

```bash
pip install streamlit
streamlit run streamlit_app.py
```
