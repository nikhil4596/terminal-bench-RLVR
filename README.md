# Terminal-Bench RLVR Environment Design

This repo is a no-training implementation package for a Terminal-Bench 2.0
RLVR environment. It is designed to satisfy the take-home assignment boundary:
define the environment, action space, observation space, reward functions,
rollout schema, trainer handoff, model/training plan, curriculum, metrics, and
large-scale readiness without running an RL job.

The repo is intentionally environment-first. Terminal-Bench tasks are executed
through Harbor, while TRL and verl are treated as downstream training backends.
That split keeps the core submission focused on verifiable terminal-agent
rollouts and leaves only compute-dependent training work for later.

## Current Status

- Core action, observation, reward, safety, rollout, and mock environment code
  is implemented.
- Rollout records include prompt, model output, scalar reward, reward
  components, terminal reason, and hashes needed for training export and audit.
- Trainer-neutral export helpers convert rollout records into GRPO/PPO sample
  rows.
- Pilot and large-scale training configuration stubs are included under
  `configs/training/`.
- Unit tests cover parsing, reward composition, safety, rollout serialization,
  mock environment behavior, training export, and readiness reporting.

No RL training run is performed in this repo.

## Main Files

```text
docs/
  submission_writeup.md
  implementation_plan.md
  training_readiness.md
configs/training/
  harbor_rollout.toml
  trl_grpo_pilot.toml
  verl_grpo_large_scale.toml
src/tb_rlvr/
  actions.py
  observations.py
  rewards.py
  safety.py
  rollout.py
  env.py
  readiness.py
  trainers/export.py
tests/
  test_actions.py
  test_rewards.py
  test_safety.py
  test_rollout.py
  test_mock_env.py
  test_training_export.py
  test_readiness.py
examples/
  run_mock_rollout.py
  check_training_readiness.py
```

## Run The Checks

```bash
python3 -m pytest
python3 examples/run_mock_rollout.py
python3 examples/check_training_readiness.py
```

## Training Backend Position

The selected environment substrate is Harbor because Terminal-Bench 2.0 tasks
are terminal/Docker environments with verifiable outcomes. The trainer backend
is deliberately separated:

- TRL is the preferred small research pilot backend for GRPO because it is easy
  to run and iterate on.
- verl is the preferred large-scale backend because production RL requires
  distributed actor/reference/rollout workers, high-throughput inference, and
  robust checkpointing.
- The current package does not import TRL or verl in core code because doing so
  would make a no-training environment repo depend on GPU/distributed packages
  that are irrelevant to local validation.

The compromise is explicit: this repo proves rollout generation and trainer
handoff, not full distributed training execution.
