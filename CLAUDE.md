# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

"AI in Robotics" university course (2026 edition). Contains lecture notebooks, homework assignments with automated grading, and infrastructure tools (Telegram bot, autograder, LLM oracle). Python 3.10+, Jupyter notebooks, Docker-based grading.

## Repository Structure

- `01-intro-and-kinematics/` through `06-simulation-and-sim2real/` — Course modules, each with `class.ipynb`, `homework/`, and `lib/`
- `tools/` — Infrastructure: bot, autograder, oracle, shared schemas, config

Each homework directory contains:
- `homework.ipynb` — Student-facing notebook
- `lib/` — Stub implementations students fill in
- `solutions/` — Reference solutions
- `tests/` — Pytest test suite
- `container/` — Docker config for grading (Dockerfile, docker-compose.yaml, entrypoint.sh)
- `autograder.yaml` — Points, metrics, limits, problem IDs

## Commands

### Install dependencies
```bash
pip install -r requirements.txt          # Course notebooks
pip install -r tools/requirements.txt    # Tools (bot, autograder, oracle)
```

### Run homework tests locally
```bash
cd 01-intro-and-kinematics/homework && pytest
cd 02-dynamics/homework && pytest
```

### Run a single test file
```bash
cd 01-intro-and-kinematics/homework && pytest tests/test_beads.py -v
```

### Run tools tests
```bash
./tools/run_tests.sh
# Or:
cd tools && PYTHONPATH="$(pwd):$PYTHONPATH" python -m pytest tests/ -v
```

### Run tools integration tests (requires Docker + dev/ mount)
```bash
cd tools && pytest tests/ -m integration -v
```

### Build and run a homework Docker container
```bash
export REPO_ROOT="$(pwd -P)"
docker compose -f 01-intro-and-kinematics/homework/container/docker_compose.yaml build
docker compose -f 01-intro-and-kinematics/homework/container/docker_compose.yaml up
```

### Run services
```bash
./tools/run_bot.sh           # Telegram bot
./tools/run_autograder.sh    # Grading daemon
./tools/run_oracle.sh        # LLM oracle (optional)
./tools/run_dashboards.sh    # Admin dashboards (ports 5001, 5002)
```

## Architecture

### Three-Service Design (tools/)
1. **Bot** (`tools/bot/`) — Telegram interface via python-telegram-bot. Accepts submissions, pushes jobs to Redis queue.
2. **Autograder** (`tools/autograder/`) — Redis BLPOP consumer. Spawns Docker containers per submission, runs pytest, parses output, stores grades in SQLite.
3. **Oracle** (`tools/oracle/`) — FastAPI server for LLM-powered Q&A and homework feedback. Optional.

Shared schemas in `tools/shared/schemas.py` (Job, GradeRow). Config in `tools/config/` (weeks.yaml, .env.example).

### Grading Flow
Student sends files via Telegram → Bot queues Redis job → Autograder spawns Docker container → pytest runs inside container → results stored in SQLite + sent back via Telegram.

Environment variable `GRADING_STUDENT_SUBMISSION=1` switches container from running reference solutions to student code.

### Course Library Modules
Each module's `lib/` provides robotics primitives (rotations, SE(2)/SE(3) transforms, kinematics, dynamics). Homework `lib/` files contain stubs that students implement; `solutions/` has the reference implementations.

## Testing Conventions

- Tests use parametrized pytest with open and hidden test cases
- Hidden tests load from `dev/<topic>/homework/hidden_tests/` (not in repo, mounted at grading time)
- Tests print `METRIC:problem_id:float_value` for leaderboard metrics
- Each homework has its own `pytest.ini` with `pythonpath = .`
- Tools tests use markers: `@pytest.mark.integration` for Docker-dependent tests

## Key Dependencies

numpy, scipy, sympy, torch, opencv-python, pytorch-kinematics, cvxpy, trimesh, lerobot[feetech], matplotlib, mediapy
