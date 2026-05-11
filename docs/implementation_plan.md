# Implementation Plan

This is the execution plan for turning the current no-training package into a
large-scale-ready Terminal-Bench RLVR system. It is written as a submission
artifact, not as a learning note.

## Goal

Build an RLVR environment for Terminal-Bench 2.0 where:

- a language-model policy observes terminal-task state,
- emits one structured action per step,
- receives verifiable reward components,
- produces auditable rollout records,
- exports trainer-neutral prompt/action/reward samples,
- can later be connected to TRL for a pilot or verl for large-scale GRPO.

The current repo implements the RLVR contract, a mock unit-test environment,
Harbor oracle command tooling, and the training handoff shape. The mock is not
the proposed benchmark runtime; it is a fast fixture for testing the contract.
Harbor is the real Terminal-Bench runtime.

## Non-Goals

- No model fine-tuning in this repo.
- No local LLM policy rollout without API or local model access.
- No claim that JSONL alone is sufficient for online GRPO.
- No leaderboard claim.
- No hidden-test leakage.
- No benchmark-specific solution memorization.
- No hard dependency on TRL or verl in core environment code.

## Architecture

```text
Policy model
  -> observation prompt
  -> structured action text
  -> action parser
  -> safety precheck
  -> Harbor task execution
  -> stdout/stderr/filesystem summary
  -> reward components
  -> rollout record
  -> audit/SFT/DPO export, or live online-GRPO handoff
  -> future TRL pilot or verl large-scale GRPO
```

See `docs/data_lifecycle.md` for sample-level walkthroughs.

## Current Repo Components

### Action Parser

File: `src/tb_rlvr/actions.py`

Responsibilities:

- parse exactly one action from model output,
- reject malformed or multi-action responses,
- serialize parsed actions back into model text for training records.

Acceptance criteria:

- `<bash>...</bash>` parses as a shell action,
- `<patch path="...">...</patch>` parses as a file mutation,
- `<finish>...</finish>` parses as a terminal action,
- outputs with zero or multiple actions fail.

### Observation Renderer

File: `src/tb_rlvr/observations.py`

Responsibilities:

- represent the task state visible to the policy,
- render a deterministic text prompt,
- avoid exposing hidden tests or oracle internals.

Acceptance criteria:

- prompt contains task instruction, cwd, directory summary, recent history,
  stdout, stderr, selected files, and remaining steps;
- prompt output is deterministic for a given observation.

### Reward Combiner

File: `src/tb_rlvr/rewards.py`

Responsibilities:

- compute scalar reward from components,
- retain component-level reward values for diagnosis,
- cap progress reward so it cannot dominate success.

Acceptance criteria:

- success maps to positive final reward,
- progress is bounded,
- integrity violation is strongly negative,
- step/token penalties are small tie-breakers.

### Safety Layer

File: `src/tb_rlvr/safety.py`

Responsibilities:

- reject protected path mutation,
- reject obvious destructive commands,
- normalize paths before checking protected prefixes.

Acceptance criteria:

- direct writes to `/app/tests` are blocked,
- traversal into protected paths is blocked,
- normal workspace edits are allowed.

### Rollout Record

File: `src/tb_rlvr/rollout.py`

Responsibilities:

- serialize step-level records to JSONL,
- preserve prompt/action/reward metadata,
- convert records into trainer-neutral sample rows.

Acceptance criteria:

- JSONL round-trips losslessly,
- records contain observation prompt and model output,
- samples expose `prompt`, `completion`, `reward`, and metadata.

### Mock Environment

File: `src/tb_rlvr/env.py`

Responsibilities:

- provide a deterministic stand-in for Harbor,
- test action/reward/safety/rollout interfaces without Docker,
- demonstrate successful and failed episodes.

Acceptance criteria:

- scripted success terminates with success reward,
- integrity violation terminates with negative reward,
- max-step timeout terminates without success.

### Harbor Oracle Smoke

Files: `src/tb_rlvr/harbor.py`, `scripts/harbor_oracle_smoke.py`

Responsibilities:

- construct the exact Harbor command for a Terminal-Bench oracle run,
- support dry-run mode with no Docker/Harbor dependency,
- execute the command when the user passes `--execute` and has Docker/uvx/Harbor.

Acceptance criteria:

- dry-run prints a command like
  `uvx harbor run -d terminal-bench@2.0 -t openssl-selfsigned-cert -a oracle`;
- no LLM, API key, or GPU is required for oracle validation;
- the command path is tested without launching Docker.

## Phase 1: Laptop-Feasible Runtime Validation

Validate the real Terminal-Bench runtime with Harbor oracle and validate the
RLVR contract with local tests.

### Commands

```bash
python3 -m pytest
python3 examples/run_mock_rollout.py
python3 scripts/export_mock_samples.py
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert
```

If Docker and Harbor/uvx are available:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert --execute
```

### Phase 1 Exit Criteria

- all unit tests pass,
- mock rollout emits a successful `RolloutRecord`,
- sample export writes prompt/completion/reward rows,
- Harbor oracle command is reproducible,
- optional real oracle run verifies one Terminal-Bench task.

## Phase 2: Harbor Policy Adapter

Implement `HarborTerminalBenchEnv` behind the same conceptual interface as
`MockTerminalBenchEnv`. This is the first step that requires a real interactive
agent/runtime integration rather than an oracle reference solution.

### Required Interface

```python
class HarborTerminalBenchEnv:
    def reset(self, task_id: str, seed: int | None = None) -> Observation: ...
    def step(self, action: AgentAction) -> StepResult: ...
```

### Reset Responsibilities

- select Terminal-Bench task,
- create isolated runtime,
- launch Harbor environment,
- capture task instruction,
- initialize step counters and episode id,
- return first observation.

### Step Responsibilities

- parse already-validated `AgentAction`,
- run safety precheck,
- execute shell command or file edit in the task container,
- collect stdout/stderr,
- summarize relevant filesystem state,
- compute progress probes if allowed,
- run final verifier only on termination,
- write `RolloutRecord`.

### Adapter Constraints

- hidden tests must not be exposed in observations;
- oracle solutions must not be readable by the policy;
- test and grader mutation must be blocked;
- network should be disabled unless the task explicitly requires it;
- timeouts must be deterministic and recorded.

### Phase 2 Exit Criteria

- one scripted policy can solve one known easy task through Harbor;
- one failing scripted policy emits a clean failure rollout;
- rollout JSONL converts to training samples;
- tests cover reset, step, timeout, safety violation, and final success.

## Phase 3: Policy Rollout Collection

Attach a frozen policy model and collect trajectories without training. This
requires either API model access or a local model server. It is not laptop-free.

### Policy Interface

```python
def policy(observation: Observation) -> str:
    return model.generate(observation.to_prompt())
```

The model output is parsed through `parse_action`.

### Collection Settings

Pilot rollout settings:

```text
tasks: 25-50
samples_per_prompt: 4-8
max_steps_per_episode: 30
temperature: 0.7
top_p: 0.95
max_completion_tokens: 2048
```

### Data To Store

- all step-level records,
- terminal records,
- task metadata,
- reward components,
- terminal reason,
- stdout/stderr snippets,
- failure taxonomy labels when reviewed.

### Phase 3 Exit Criteria

- at least 50-100 complete rollouts;
- reward component histograms look sane;
- integrity violations are rare and explainable;
- progress reward correlates with final success;
- there are enough successful and failed examples for GRPO grouping.

## Phase 4: TRL GRPO Pilot

Use TRL for a small research pilot after Harbor policy rollouts are valid. This
requires GPU compute because the trainer must sample from the current policy,
compute token logprobs, compute group-relative advantages, and update weights.

### Purpose

- validate reward scaling,
- validate group-size choice,
- validate KL coefficient,
- detect reward hacking,
- confirm action grammar remains stable after optimization.

### Configuration

See `configs/training/trl_grpo_pilot.toml`.

Initial defaults:

```text
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
algorithm: GRPO
num_generations_per_prompt: 8
learning_rate: 1e-6
kl_coefficient_start: 0.02
temperature: 0.7
top_p: 0.95
```

### Phase 4 Exit Criteria

- loss and KL are stable;
- integrity violations do not increase;
- held-out success does not regress;
- qualitative trajectories show more effective terminal use;
- reward improvements are not explained only by shorter completions or public
  probe exploitation.

## Phase 5: verl Large-Scale Training

Move to verl only after the TRL pilot validates the reward/environment design.

### Purpose

Scale rollout throughput and policy updates without changing environment
semantics.

### Required Infrastructure

- CUDA training image,
- Harbor installed in rollout workers,
- verl installed in trainer image,
- Ray or Slurm launcher,
- model checkpoint store,
- rollout object store,
- experiment tracker,
- held-out eval jobs.

### Configuration

See `configs/training/verl_grpo_large_scale.toml`.

### Phase 5 Exit Criteria

- distributed workers can collect rollouts and train from the same schema;
- checkpoints are reproducible;
- evaluation runs on held-out tasks at fixed intervals;
- reward component dashboards are stable;
- scale-up changes throughput, not task semantics.

## Failure Taxonomy

Every failed episode should be classifiable as one of:

- parse failure,
- invalid action,
- command execution failure,
- wrong file edited,
- build/test failure,
- timeout,
- safety violation,
- final verifier failure,
- reward/progress mismatch,
- environment crash.

This taxonomy should be stored as metadata after automated or manual review.

## Metrics

Primary:

- held-out pass rate,
- pass rate by task family,
- average return,
- final success reward,
- timeout rate,
- integrity violation rate.

Efficiency:

- average steps per episode,
- average steps per successful episode,
- average tokens per episode,
- command failure rate.

Training:

- KL to reference,
- reward component histograms,
- clip ratio,
- entropy,
- group reward variance.

Safety:

- protected path attempts,
- oracle access attempts,
- destructive command attempts,
- network attempts.

## Readiness Checklist

Ready now:

- action parser,
- observation schema,
- reward combiner,
- safety precheck,
- rollout schema,
- training-sample export,
- mock environment,
- Harbor oracle command dry-run,
- data lifecycle documentation,
- config skeletons,
- tests,
- submission write-up.

External or future:

- Harbor runtime execution if Docker/uvx/Harbor are unavailable locally,
- real Terminal-Bench adapter,
- frozen model/API rollout collector,
- GPU-backed TRL launch script,
- verl scale-up launch script,
- cluster deployment,
- actual RL training.

## Implementation Order

1. Keep the current tests passing.
2. Run or dry-run Harbor oracle smoke.
3. Add `HarborTerminalBenchEnv`.
4. Add one scripted policy integration test.
5. Generate rollout JSONL from a frozen model or API policy.
6. Review reward distributions.
7. Add GPU-backed TRL GRPO pilot script.
8. Move to verl only after the pilot validates reward and action semantics.

The final training job should not require redesigning the action, observation,
reward, safety, or rollout contract. It should require a real Harbor policy
adapter, model access, and a trainer backend that calls this environment while
sampling from the current policy.
