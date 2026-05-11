# Training Readiness Audit

This document evaluates whether the repo can move from a no-training take-home
submission into large-scale RL training without redesigning the environment.

## Decision Summary

The repo is organized around three separate layers:

```text
Terminal-Bench task execution
  -> Harbor rollout environment
  -> tb_rlvr action, observation, reward, safety, and rollout schema
  -> trainer-neutral dataset export
  -> TRL pilot or verl large-scale GRPO training
```

The core package is not built directly on TRL or verl. This is intentional.
TRL and verl are training systems; they are not the source of truth for the
terminal environment. Terminal-Bench 2.0 tasks, Docker state, hidden/public
tests, timeouts, and oracle discipline belong in the environment layer.

## What Is Ready

### Environment Contract

Ready.

The repo defines the required abstractions:

- `AgentAction` for structured actions.
- `Observation` for prompt construction.
- `MockTerminalBenchEnv` as a deterministic stand-in for the Harbor adapter.
- `RewardComponents` for decomposed reward logging.
- `RolloutRecord` for JSONL audit and trainer export.
- Safety checks for protected paths and destructive commands.

The mock environment is not a substitute for Harbor. Its role is to make the
interfaces executable and testable without Docker or network-dependent setup.

### Reward Interface

Ready for pilot training.

The reward schema supports:

- final success reward,
- bounded progress reward,
- integrity/safety penalty,
- step efficiency penalty,
- token penalty.

The important large-scale property is decomposition. A single scalar reward is
available for GRPO/PPO, but component-level rewards are retained for diagnosing
reward hacking, safety regressions, and curriculum failures.

### Rollout Format

Ready for trainer handoff.

Each rollout record stores:

- task id,
- step id,
- observation prompt before the action,
- model output/action text,
- parsed action,
- scalar reward,
- reward components,
- done flag,
- terminal reason,
- observation hashes,
- backend metadata.

This is enough to produce prompt/completion/reward rows for TRL, verl, or a
custom GRPO loop.

### Configuration Skeleton

Ready as a planning artifact.

The repo includes:

- `configs/training/harbor_rollout.toml`
- `configs/training/trl_grpo_pilot.toml`
- `configs/training/verl_grpo_large_scale.toml`

These are not executable launch files yet. They pin the intended boundaries and
make the remaining training work clear.

### Tests

Ready.

The current local validation suite checks the environment contract and data
flow. A real training launch would add integration tests that run one Harbor
task end to end.

## What Is Not Ready Locally

### Harbor Runtime

Not installed in the local package path.

This is expected for the current repo state. The implementation provides the
adapter boundary and mock; the real run image would install Harbor and execute
Terminal-Bench tasks through it.

Required next step:

```text
Implement HarborTerminalBenchEnv behind the same reset/step interface.
```

### TRL Runtime

Not installed in the core environment.

Reason: TRL is useful for small-scale GRPO experiments but should not be a hard
dependency of the environment package. Requiring TRL in core code would make
local validation slower and couple environment correctness to a specific
training implementation.

Compromise:

- The repo cannot launch a TRL GRPO job today.
- It can export training samples in the shape a TRL pipeline needs.
- A future `trl_train.py` can consume those samples without changing the env.

### verl Runtime

Not installed in the core environment.

Reason: verl is the right backend for large-scale RL, but it brings distributed
training assumptions: GPU workers, inference engines, checkpoint stores, Ray or
equivalent orchestration, and cluster-specific launch files.

Compromise:

- The repo cannot launch distributed GRPO today.
- It already separates rollout collection from trainer execution, which is the
  important architectural precondition for using verl later.
- The large-scale config identifies the missing cluster resources.

### Distributed Infrastructure

Not configured.

Missing pieces:

- GPU cluster allocation,
- Ray or Slurm launcher,
- Docker image with Harbor, model runtime, and trainer packages,
- shared checkpoint storage,
- rollout dataset storage,
- experiment tracking.

These are deployment decisions, not environment-design decisions.

## Why Not Make TRL Or verl The Core Framework?

The assignment asks for an RLVR environment. For Terminal-Bench 2.0, the hard
part is not calling an optimizer. The hard part is constructing a reproducible,
auditable terminal environment where a policy can:

- observe task state,
- emit terminal/file actions,
- receive bounded feedback,
- avoid test/oracle tampering,
- terminate cleanly,
- produce records that are trainable and debuggable.

TRL and verl do not replace this layer. They consume it.

### Capabilities Compromised By Not Using TRL In Core

- No immediate `GRPOTrainer` script.
- No direct Hugging Face dataset/trainer integration.
- No automatic reference-policy KL handling in the repo.
- No built-in LoRA/full-finetuning launch path.

Mitigation:

- Keep `trl_grpo_pilot.toml` as the pilot configuration.
- Export prompt/completion/reward samples from `RolloutRecord`.
- Add a small TRL adapter only after Harbor integration is working.

### Capabilities Compromised By Not Using verl In Core

- No distributed actor/critic/reference worker orchestration.
- No vLLM/SGLang rollout worker integration.
- No cluster-scale checkpointing.
- No Ray-based launch script.

Mitigation:

- Keep `verl_grpo_large_scale.toml` as the large-scale target.
- Preserve backend-neutral rollout schema.
- Add verl only in the training image, not the environment package.

## Spring-To-Training Plan

### Stage 1: Real Harbor Adapter

Implement `HarborTerminalBenchEnv` with the same interface as
`MockTerminalBenchEnv`.

Required behavior:

- launch a selected Terminal-Bench task,
- expose task instruction and workspace summary,
- execute one action per step,
- collect stdout/stderr and file summaries,
- run public progress checks when allowed,
- run final verifier only at episode termination,
- block hidden-test/oracle leakage,
- write `RolloutRecord` rows.

Exit criterion:

```text
A scripted policy can solve one toy or easy Terminal-Bench task through Harbor,
and the generated JSONL converts to training samples.
```

### Stage 2: Rollout Collection

Attach a frozen instruct model as the policy.

Collect:

- 8 samples per prompt for GRPO group comparison,
- task-family-stratified rollouts,
- terminal and non-terminal step records,
- failure taxonomy labels,
- reward component histograms.

Exit criterion:

```text
At least 50-100 low-cost rollouts exist across multiple task families,
and reward distributions are sane.
```

### Stage 3: TRL Pilot

Run a small GRPO experiment with Qwen2.5-Coder-7B-Instruct.

Purpose:

- validate reward scale,
- validate action protocol,
- test KL range,
- find reward hacking modes,
- test curriculum order.

This stage is for correctness, not final performance.

### Stage 4: verl Scale-Up

Move to distributed training after the pilot passes.

Required additions:

- Ray or Slurm launcher,
- model sharding strategy,
- vLLM or SGLang rollout workers,
- object-store backed rollout data,
- checkpoint retention policy,
- evaluation jobs on held-out task families.

Exit criterion:

```text
The only changes from pilot to scale are backend, launch, and throughput
configuration. Reward and environment semantics remain unchanged.
```

## Readiness Verdict

The repo is ready as a take-home submission and as a pre-training environment
package. It is not yet a training repo. The remaining work is operational:
install Harbor, implement the concrete Harbor adapter, run a small rollout
collection, then choose TRL for pilot or verl for scale.

That is the intended boundary. Environment design, reward decomposition,
logging, and trainer handoff are already represented in code.
