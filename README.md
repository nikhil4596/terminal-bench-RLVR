# Terminal-Bench RLVR Environment Design

This repo is a take-home submission for designing a reinforcement learning with
verifiable rewards (RLVR) environment for Terminal-Bench 2.0.

The repo is intentionally environment-first:

```text
Terminal-Bench task
  -> Harbor runtime/verifier
  -> tb_rlvr observation/action/reward/safety/rollout contract
  -> future TRL or verl GRPO training loop
```

No GPU or API-model access is assumed locally. On a laptop, the realistic checks
are unit tests, mock rollouts, trainer-sample export, and optionally a Harbor
oracle smoke task if Docker and Harbor/uvx are available. Actual LLM policy
rollouts and GRPO training are the next resource-dependent step.

## Current Status

- Core action, observation, reward, safety, Harbor-command, rollout, and mock
  environment code is implemented.
- Rollout records include prompt, model output, scalar reward, reward
  components, terminal reason, and hashes needed for training export and audit.
- Trainer-neutral export helpers convert rollout records into audit/SFT/DPO
  sample rows. Online GRPO still requires live policy sampling and logprobs.
- Pilot and large-scale training configuration stubs are included under
  `configs/training/`.
- Unit tests cover parsing, reward composition, safety, rollout serialization,
  Harbor command construction, mock environment behavior, training export, and
  readiness reporting.

No RL training run is performed in this repo.

## Main Files

```text
docs/
  submission_writeup.md
  implementation_plan.md
  data_lifecycle.md
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
  harbor.py
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
scripts/
  harbor_oracle_smoke.py
  export_mock_samples.py
```

## Setup

With conda:

```bash
conda env create -f environment.yml
conda activate tb-rlvr
```

Or with an existing Python 3.11 environment:

```bash
pip install -e .
pip install pytest
```

## Local Checks

```bash
python3 -m pytest
python3 examples/run_mock_rollout.py
python3 examples/check_training_readiness.py
python3 scripts/export_mock_samples.py
```

## Reading Order

For review/submission:

1. `docs/submission_writeup.md` - full answer to the take-home prompt.
2. `docs/data_lifecycle.md` - concrete task-to-rollout-to-trainer data flow.
3. `docs/implementation_plan.md` - implementation phases and acceptance
   criteria.
4. `docs/training_readiness.md` - what is ready locally versus what needs model
   access/GPU/cluster resources.

For code review:

1. `src/tb_rlvr/actions.py`
2. `src/tb_rlvr/observations.py`
3. `src/tb_rlvr/rewards.py`
4. `src/tb_rlvr/safety.py`
5. `src/tb_rlvr/rollout.py`
6. `src/tb_rlvr/harbor.py`
7. `src/tb_rlvr/env.py`

## Harbor Oracle Smoke

Oracle means Harbor runs the task's reference solution. It is not an LLM and
does not need a GPU or API key.

Dry-run the command:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert
```

If Docker and `uvx`/Harbor are available, execute it:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert --execute
```

## Training Backend Position

The selected environment substrate is Harbor because Terminal-Bench 2.0 tasks
are terminal/Docker environments with verifiable outcomes. The trainer backend
is deliberately separated:

- TRL is the preferred small research pilot backend for GRPO after a real model
  policy and GPU resources are available.
- verl is the preferred large-scale backend because production RL requires
  distributed actor/reference/rollout workers, high-throughput inference, and
  robust checkpointing.
- The current package does not import TRL or verl in core code because local
  validation does not require model training dependencies.

The compromise is explicit: this repo proves the environment contract, data
lifecycle, and trainer handoff shape. It does not pretend to run policy
rollouts or GRPO without model access and GPU compute.
