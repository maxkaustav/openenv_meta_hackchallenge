---
title: Hospital Reception Support ENV
sdk: docker
---

# Healthcare Appointment Scheduling OpenEnv Project

## Overview

This project provides a robust, production-grade Reinforcement Learning environment for healthcare appointment scheduling, compliant with the OpenEnv specification.
An AI agent receives a natural-language patient request, identifies the correct medical department and doctor, and books an appointment, also rebooking an appointment using a set of structured tool calls.

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

### Environment Setup

Create a `.env` file in the root directory and add your API keys required for execution:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=your_model_name
HF_TOKEN=your_hf_token_here
```

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

### Inference Output

```
==============================================================
  Healthcare Appointment Scheduling — Baseline Evaluation
==============================================================
  Mode: GROQ (llama-3.1-8b-instant)
==============================================================

  TASK [EASY]
  User request: "I have chest pain"
  Expected: Cardiology -> Dr. Sarah Smith
  ----------------------------------------------------------
    [LLM raw] {
  "tool": "get_departments",
  "parameters": {},
  "reasoning": "Begin by determining available departments to narrow down the possible causes of chest pain."
}...
    Step 1: get_departments({})
           → reward=1.0000 | Retrieved department list. Now call get_doctors(department). (+1.0)
    [LLM raw] {
  "tool": "ask_user_clarification",
  "parameters": {"question": "Is the chest pain related to the heart or is it a sharp pain in the chest?"},
  "reasoning": "The symptom 'chest pain' is ambiguous ...
    Step 2: ask_user_clarification({"question": "Is the chest pain related to the heart or is it a sharp pain in the chest?"})
           → reward=0.0000 | Got user clarification.
    [LLM raw] {
  "tool": "get_departments",
  "parameters": {},
  "reasoning": "Based on the user's response, the chest pain seems to be related to the heart, so we should focus on the Cardiology department."
}...
    Step 3: get_departments({})
           → reward=0.0000 | Retrieved department list again.
    [LLM raw] {
  "tool": "get_doctors",
  "parameters": {"department": "Cardiology"},
  "reasoning": "Based on the user's symptoms, Cardiology is the most likely department to handle the issue, so we should get th...
    Step 4: get_doctors({"department": "Cardiology"})
           → reward=1.0000 | Correct department 'Cardiology' identified! (+1.0)
    [LLM raw] {
  "tool": "check_availability",
  "parameters": {"doctor": "Dr. Sarah Smith"},
  "reasoning": "Based on the user's symptoms, Dr. Sarah Smith's specialization in general cardiology and chest pain mak...
    Step 5: check_availability({"doctor": "Dr. Sarah Smith"})
           → reward=1.0000 | Checking availability for the right doctor (Dr. Sarah Smith). (+1.0)
    [LLM raw] {
  "tool": "book_appointment",
  "parameters": {"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"},
  "reasoning": "Based on the available slots, we should book an appointment with Dr. Sarah...
    Step 6: book_appointment({"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"})
           → reward=1.0000 | Booking successful with correct department and doctor. (+1.0)

  [OK] Score: 4.000
      Got departments:     True  (+1.00)
      Department correct:  True  (+1.00)
      Doctor correct:      True  (+1.00)
      Booking successful:  True  (+1.00)
      Clarification penalty: 0.00
      Steps taken:         6
      Env cumulative reward:4.0000

  TASK [MEDIUM]
  User request: "I have a skin rash for 2 weeks"
  Expected: Dermatology -> Dr. Priya Patel
  ----------------------------------------------------------
    [LLM raw] {
  "tool": "get_departments",
  "parameters": {},
  "reasoning": "Begin by determining available departments to narrow down the possible specialists."
}...
    Step 1: get_departments({})
           → reward=1.0000 | Retrieved department list. Now call get_doctors(department). (+1.0)
    [LLM raw] {
  "tool": "get_doctors",
  "parameters": {"department": "Dermatology"},
  "reasoning": "Match the symptom to the correct department, which in this case is Dermatology."
}...
    Step 2: get_doctors({"department": "Dermatology"})
           → reward=1.0000 | Correct department 'Dermatology' identified! (+1.0)
    [LLM raw] {
  "tool": "check_availability",
  "parameters": {"doctor": "Dr. Priya Patel"},
  "reasoning": "Choose Dr. Priya Patel as the specialist for skin rashes, which matches the user's symptom."
}...
    Step 3: check_availability({"doctor": "Dr. Priya Patel"})
           → reward=1.0000 | Checking availability for the right doctor (Dr. Priya Patel). (+1.0)
    [LLM raw] {
  "tool": "book_appointment",
  "parameters": {"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"},
  "reasoning": "Book an appointment with Dr. Priya Patel on January 15th at 8:00 AM, which...
    Step 4: book_appointment({"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"})
           → reward=1.0000 | Booking successful with correct department and doctor. (+1.0)

  [OK] Score: 4.000
      Got departments:     True  (+1.00)
      Department correct:  True  (+1.00)
      Doctor correct:      True  (+1.00)
      Booking successful:  True  (+1.00)
      Clarification penalty: 0.00
      Steps taken:         4
      Env cumulative reward:4.0000

  TASK [HARD]
  User request: "I feel pain but not sure where"
  Expected: Cardiology -> Dr. Sarah Smith
  ----------------------------------------------------------
    [LLM raw] {
  "tool": "get_departments",
  "parameters": {},
  "reasoning": "Begin by getting a list of available medical departments to narrow down the options"
}...
    Step 1: get_departments({})
           → reward=1.0000 | Retrieved department list. Now call get_doctors(department). (+1.0)
    [LLM raw] {
  "tool": "ask_user_clarification",
  "parameters": {"question": "Can you tell me where the pain is located (e.g. head, back, arm)?"},
  "reasoning": "The user's symptom is ambiguous, so we need to ...
    Step 2: ask_user_clarification({"question": "Can you tell me where the pain is located (e.g. head, back, arm)?"})
           → reward=0.0000 | Got user clarification.
    [LLM raw] {
  "tool": "get_departments",
  "parameters": {},
  "reasoning": "Based on the user's response, we can now determine the appropriate department. Chest pain is often related to Cardiology or Orthopedi...
    Step 3: get_departments({})
           → reward=0.0000 | Retrieved department list again.
    [LLM raw] {
  "tool": "get_doctors",
  "parameters": {"department": "Cardiology"},
  "reasoning": "Based on the user's description of chest pain, we're going to focus on Cardiology to find a suitable doctor"
}...
    Step 4: get_doctors({"department": "Cardiology"})
           → reward=1.0000 | Correct department 'Cardiology' identified! (+1.0)
    [LLM raw] {
  "tool": "check_availability",
  "parameters": {"doctor": "Dr. Sarah Smith"},
  "reasoning": "We'll start by checking the availability of Dr. Sarah Smith, a general cardiologist with a specializati...
    Step 5: check_availability({"doctor": "Dr. Sarah Smith"})
           → reward=1.0000 | Checking availability for the right doctor (Dr. Sarah Smith). (+1.0)
    [LLM raw] {
  "tool": "book_appointment",
  "parameters": {"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"},
  "reasoning": "We've found an available slot for Dr. Sarah Smith, so we'll book the appoi...
    Step 6: book_appointment({"doctor": "Dr. Sarah Smith", "slot": "2024-01-15 09:00 AM"})
           → reward=1.0000 | Booking successful with correct department and doctor. (+1.0)

  [OK] Score: 4.000
      Got departments:     True  (+1.00)
      Department correct:  True  (+1.00)
      Doctor correct:      True  (+1.00)
      Booking successful:  True  (+1.00)
      Clarification penalty: 0.00
      Steps taken:         6
      Env cumulative reward:4.0000

  TASK [REBOOK]
  User request: "I need to rebook my appointment with Dr. Priya Patel"
  Expected: Dermatology -> Dr. Priya Patel
  ----------------------------------------------------------
    [LLM raw] {
  "tool": "check_availability",
  "parameters": {"doctor": "Dr. Priya Patel"},
  "reasoning": "User has explicitly named a specific doctor, so we can directly check their availability."
}...
    Step 1: check_availability({"doctor": "Dr. Priya Patel"})
           → reward=1.0000 | Checking availability for the right doctor (Dr. Priya Patel). (+1.0)
    [LLM raw] {
  "tool": "book_appointment",
  "parameters": {"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"},
  "reasoning": "We have found a suitable doctor and available slot, so we can book the app...
    Step 2: book_appointment({"doctor": "Dr. Priya Patel", "slot": "2024-01-15 08:00 AM"})
           → reward=1.0000 | Booking successful with correct department and doctor. (+1.0)

  [OK] Score: 4.000
      Got departments:     False  (+0.00)
      Department correct:  True  (+0.00)
      Doctor correct:      True  (+1.00)
      Booking successful:  True  (+1.00)
      Clarification penalty: 0.00
      Steps taken:         2
      Env cumulative reward:2.0000

==============================================================
  SUMMARY
==============================================================
  EASY    4.000  |####################|
  MEDIUM  4.000  |####################|
  HARD    4.000  |####################|
  REBOOK  4.000  |####################|

  Average Score: 4.000
==============================================================
```
