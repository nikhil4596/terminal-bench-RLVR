# Training Readiness Audit

This document separates what is ready in this repo from what requires external
resources. The key correction is that "no training run" does not mean "mock
only." The repo should be ready to validate the task runtime and data contract,
while honestly stopping before LLM policy rollouts and GRPO updates.

## Execution Boundary

Laptop-feasible without GPU or API keys:

- run the unit test suite,
- run the mock RLVR environment,
- export trainer-neutral toy samples,
- dry-run the exact Harbor oracle command,
- run a real Harbor oracle smoke task if Docker and `uvx`/Harbor are installed,
- inspect the data lifecycle and rollout schema.

Not feasible without additional resources:

- real LLM policy rollouts,
- token logprob collection for a trainable model,
- GRPO/PPO updates,
- benchmark improvement claims,
- distributed verl training.

The deliverable boundary is therefore:

```text
Validate real Terminal-Bench runtime with Harbor oracle.
Validate RLVR data contract with code/tests/mock rollout.
Specify how online GRPO attaches.
Do not pretend to run LLM rollout or training without model access/GPU.
```

## Layered System

```text
Terminal-Bench 2.0
  = task suite: instructions, Docker envs, tests, oracle solutions

Harbor
  = task runtime/evaluator: launches tasks, runs agents/oracle, runs verifiers

tb_rlvr
  = RLVR contract: observations, actions, safety, rewards, rollout JSONL

TRL
  = future small-pilot optimizer: online GRPO when model/GPU are available

verl
  = future scale optimizer: distributed rollout/training infrastructure
```

Harbor is not a replacement for this repo's environment design. Harbor is the
backend runtime. This repo defines the RL interface around that runtime.

## Current Repo Readiness

### Environment Contract

Status: ready for local validation.

Implemented:

- `AgentAction` for structured `bash`, `patch`, and `finish` actions.
- `Observation` for prompt construction.
- `RewardComponents` for decomposed reward logging.
- `RolloutRecord` for audit and trainer handoff.
- `MockTerminalBenchEnv` for fast unit tests.
- `harbor.py` for Harbor oracle command construction and dry-run execution.

The mock environment is not the proposed benchmark environment. It is a unit
test fixture. Harbor is the real Terminal-Bench runtime.

### Harbor Runtime

Status: command path represented; execution depends on local Docker/uvx/Harbor.

Dry-run:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert
```

Real oracle smoke, if dependencies are installed:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert --execute
```

Equivalent command:

```bash
uvx harbor run -d terminal-bench@2.0 -t openssl-selfsigned-cert -a oracle
```

Oracle means reference solution. It is not an LLM. This check validates the
task Docker environment and verifier, not policy behavior.

### Reward Interface

Status: ready for schema validation.

Implemented reward components:

- final success reward,
- bounded progress reward,
- integrity/safety penalty,
- step penalty,
- token penalty.

The scalar reward is suitable for trainer consumption, but component rewards
are retained for diagnosis.

### Rollout And Handoff Format

Status: ready for audit, SFT/DPO-style conversion, and future online RL
integration.

Each `RolloutRecord` stores:

- task id,
- episode id,
- backend,
- step,
- observation prompt,
- model output,
- parsed action,
- reward components,
- scalar reward,
- terminal reason,
- observation hashes,
- metadata.

Important limitation:

```text
JSONL records alone are not enough for online GRPO unless they also include
fresh policy samples and logprobs from the policy being updated.
```

For online GRPO, TRL/verl must call the environment while sampling from the
current model.

### Trainer-Neutral Export

Status: ready for toy export.

Command:

```bash
python3 scripts/export_mock_samples.py
```

This writes JSONL sample rows from the mock environment. It validates the data
shape but does not represent real Terminal-Bench training data.

### TRL

Status: future pilot backend.

TRL is not imported by core code. That is intentional because local validation
does not require GPU training dependencies.

TRL becomes relevant when:

- a model policy is available,
- rollouts can be sampled from that policy,
- token logprobs can be computed,
- rewards can be returned from the Harbor-backed environment,
- GPU compute exists for the update.

The correct future integration is online:

```text
prompt -> current policy samples G actions/trajectories -> Harbor scores them
       -> TRL computes logprobs and group advantages -> policy update
```

### verl

Status: scale-up discussion only.

verl should not be a laptop requirement. It is appropriate when moving from a
single-machine pilot to distributed actor/reference/rollout workers.

Required future resources:

- GPU cluster,
- Ray/Slurm/Kubernetes launcher,
- vLLM/SGLang rollout serving,
- checkpoint store,
- rollout data store,
- evaluation scheduler.

## Readiness Commands

Local checks:

```bash
python3 -m pytest
python3 examples/run_mock_rollout.py
python3 scripts/export_mock_samples.py
python3 examples/check_training_readiness.py
```

Harbor command dry-run:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert
```

Harbor command execution, if Docker and `uvx`/Harbor are installed:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert --execute
```

## What "Ready For Training" Means Here

Ready:

- task/runtime choice is clear,
- laptop-feasible Harbor oracle path is documented,
- RLVR action/observation/reward schema exists,
- rollout records are serializable,
- trainer-neutral sample export works on toy data,
- online GRPO handoff is specified.

Not ready by design:

- no LLM policy rollout without API/local model access,
- no GRPO update without GPU,
- no distributed verl run without cluster infrastructure.

## Verdict

The repo is ready as an environment-design take-home if it is described
accurately:

```text
Harbor validates real Terminal-Bench task execution.
Mock tests validate the RLVR interface.
Docs define the exact data lifecycle and training handoff.
Actual policy rollout and GRPO training are resource-dependent next steps.
```
