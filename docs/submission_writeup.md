# RLVR Environment Design For Terminal-Bench 2.0

## Executive Summary

This proposal designs a reinforcement learning with verifiable rewards (RLVR)
environment for improving coding-agent performance on Terminal-Bench 2.0. The
core idea is to train an LLM policy through multi-step terminal interaction,
where each episode is a real task environment and rewards come from executable
verification, progress probes, safety checks, and efficiency penalties.

The implementation package stops before model training, as requested. It
provides the environment interface, action protocol, observation format, reward
schema, safety checks, rollout record format, trainer-neutral export path,
configuration skeletons, tests, and a training-readiness audit. With compute
and a Harbor runtime available, the remaining work is to run rollouts and
connect those records to a GRPO trainer.

Selected stack:

- Benchmark: Terminal-Bench 2.0.
- Environment substrate: Harbor.
- Pilot trainer: TRL GRPO.
- Large-scale trainer: verl GRPO.
- Base model: Qwen2.5-Coder-7B-Instruct.
- RL algorithm: GRPO.

The central design choice is to keep the environment independent from the
trainer. Harbor owns task execution. The repo owns state/action/reward/logging.
TRL or verl consumes exported rollouts for training.

## Assignment Scope

The requested deliverable is a design and code package, not a completed RL run.
This submission covers:

- a selected open-source RL stack,
- benchmark selection and justification,
- observation/state space,
- action space,
- at least two virtual/verifiable reward functions,
- model and training plan,
- dataset and curriculum plan,
- hyperparameter plan,
- evaluation metrics,
- code skeleton and smoke tests,
- large-scale readiness assessment.

This submission does not claim benchmark improvement and does not fine-tune
model weights.

## Benchmark Choice

### Selected Benchmark: Terminal-Bench 2.0

Terminal-Bench 2.0 is the better fit for this assignment because it evaluates
LLM agents in terminal environments with executable task verification. The
tasks require code inspection, shell use, file editing, debugging, dependency
management, and long-horizon state tracking. These properties match the RLVR
setting: a policy takes actions, changes an environment, and receives rewards
from verifiable outcomes.

The alternative, tau2-bench, is also valuable but less aligned with this
specific environment design. tau2-bench introduces user simulation and
conversation dynamics. That is useful for customer-support or tool-use agents,
but it adds a second-agent modeling problem. Terminal-Bench gives a cleaner
first RLVR environment for coding-agent post-training because the verifier is
primarily task-state based.

### Why Terminal-Bench Works For RLVR

Terminal-Bench provides:

- task instructions,
- isolated execution environments,
- Docker-style reproducibility,
- final-state verifiers,
- time limits,
- varied task families,
- natural long-horizon interaction,
- measurable success/failure outcomes.

This creates an MDP/POMDP-like training loop:

```text
observation_t -> policy -> action_t -> terminal environment -> reward_t
```

The environment is partially observable because the model does not see the full
filesystem or hidden tests at every step. It must decide what to inspect and
when to stop.

## Framework Selection

### Environment Framework: Harbor

Harbor is the correct environment substrate because Terminal-Bench 2.0 tasks
are run as terminal environments. The environment layer must launch tasks,
execute commands, collect outputs, enforce timeouts, preserve hidden verifier
discipline, and return task-state observations. Those responsibilities belong
closer to Harbor than to a trainer.

Using Harbor-first avoids a common failure mode: building a trainer demo that
cannot faithfully execute the benchmark. In this submission, the environment
contract is defined first, and the trainer integration comes after rollouts are
valid.

### Pilot Training Framework: TRL

TRL is appropriate for a small GRPO pilot because it is a lightweight research
interface around Hugging Face models. It is useful for validating reward scale,
group size, KL settings, action formatting, and curriculum on a modest number
of rollouts.

TRL is not used as a hard dependency of the core package because the local
submission does not run training. Pulling trainer dependencies into the
environment package would make the no-training repo heavier without improving
the environment design.

### Large-Scale Training Framework: verl

verl is the better large-scale backend because serious post-training requires
distributed actor/reference workers, high-throughput rollout generation,
checkpoint management, and cluster orchestration. Those capabilities matter
once the Harbor adapter is producing reliable rollouts.

The repo therefore includes a verl-oriented large-scale config, but not a live
distributed launch. The missing pieces are cluster specific.

### Capability Tradeoff

The main compromise is explicit:

- without TRL in core, the repo cannot immediately call a local `GRPOTrainer`;
- without verl in core, the repo cannot launch distributed training;
- without Harbor installed locally, the repo cannot yet run real
  Terminal-Bench tasks.

The benefit is also explicit:

- the environment semantics are trainer-neutral;
- local tests remain fast;
- rollout records are auditable;
- the same reward/action schema can feed TRL, verl, or another backend.

## Environment Design

### Episode Definition

An episode corresponds to one Terminal-Bench task attempt.

Episode lifecycle:

```text
reset(task_id)
  -> launch task environment
  -> construct initial observation
  -> loop over structured model actions
  -> compute reward components
  -> terminate on success, finish, timeout, max steps, or integrity violation
  -> emit rollout records
```

### Observation Space

The observation is text-structured because the policy is a language model. It
contains:

- task id,
- task instruction,
- current working directory,
- directory summary,
- recent action history,
- last stdout,
- last stderr,
- selected file contents,
- remaining step budget.

The current prototype defines this as:

```python
class Observation:
    task_id: str
    instruction: str
    cwd: str
    directory_summary: str
    recent_history: tuple[str, ...]
    last_stdout: str
    last_stderr: str
    selected_files: dict[str, str]
    steps_remaining: int
```

The observation deliberately does not expose hidden tests, oracle answers, or
private verifier internals.

### Action Space

The action space is a constrained text protocol with one action per assistant
turn:

```text
<bash>command</bash>
<patch path="/app/path.py">file contents or patch body</patch>
<finish>ready for grading</finish>
```

Allowed action types:

- `bash`: run a shell command in the task environment,
- `patch`: edit a specific file,
- `finish`: ask the environment to run the final verifier.

The single-action constraint improves auditability. It also creates clearer
credit assignment for training because each reward update corresponds to one
environment mutation.

### Termination Conditions

An episode terminates when:

- final verifier succeeds,
- the model emits `finish`,
- max step budget is reached,
- wall-clock timeout is reached,
- safety or integrity violation occurs,
- environment execution fails unrecoverably.

The rollout record stores the terminal reason.

## Reward Design

The reward function is decomposed into components and then summed into a scalar
for GRPO/PPO.

```text
R_total =
  w_success   * R_success
+ w_progress  * R_progress
+ w_integrity * R_integrity
+ w_step      * R_step
+ w_token     * R_token
```

### Reward 1: Final Success Reward

Final success is the primary verifiable reward:

```text
R_success = 1.0 if the final verifier passes else 0.0
```

This is the anchor reward. It should dominate the objective because Terminal-
Bench performance is ultimately measured by task success.

### Reward 2: Progress Reward

Progress reward gives bounded intermediate credit:

```text
R_progress = clamp(progress_delta, 0.0, 0.20)
```

Possible progress signals:

- public tests pass after failing,
- expected file appears,
- syntax check improves,
- command output matches task-specific probe,
- build step progresses further than before.

Progress reward is capped so the model cannot optimize probe-chasing over final
success.

### Reward 3: Integrity Penalty

Integrity reward penalizes forbidden behavior:

```text
R_integrity = -1.0 if the action mutates tests/oracles/graders or attempts a
forbidden shortcut else 0.0
```

This is necessary because coding agents can otherwise learn to attack the
benchmark harness instead of solving the task.

### Reward 4: Efficiency Penalty

Step and token penalties discourage wasteful trajectories:

```text
R_step = -0.01 * step_count
R_token = -0.00001 * generated_tokens
```

These penalties are small relative to success. They should break ties between
successful trajectories rather than prevent useful exploration.

## Rollout Schema

Each step produces a JSONL record with:

- task id,
- episode id,
- backend,
- step,
- observation prompt,
- observation hash,
- model output,
- parsed action,
- reward components,
- scalar reward,
- done flag,
- terminal reason,
- next observation hash,
- safety/execution metadata.

This is sufficient for:

- audit,
- failure analysis,
- trajectory replay,
- reward debugging,
- GRPO/PPO dataset conversion.

The trainer-neutral sample format is:

```text
prompt: observation before action
completion: sampled model action
reward: scalar reward
metadata: task id, step, terminal reason, reward components, hashes
```

## Model Choice

### Base Model

The pilot model is Qwen2.5-Coder-7B-Instruct.

Reasons:

- strong coding prior,
- small enough for pilot-scale experiments,
- open weights,
- practical context length,
- suitable for terminal-agent instruction following.

Larger follow-up candidates:

- Qwen2.5-Coder-14B-Instruct,
- DeepSeek-Coder style models,
- CodeLlama-family baselines,
- general instruction models for cross-domain comparison.

The final submission should avoid claiming model improvement until a real
training run and held-out evaluation are complete.

## RL Algorithm

### Selected Algorithm: GRPO

GRPO is selected because it is practical for RLVR where rewards are scalar,
verifiable, and often sparse. Instead of fitting a separate value model, GRPO
compares groups of sampled completions for the same prompt and updates the
policy based on relative advantage.

This fits Terminal-Bench:

- for each task state, sample multiple candidate actions,
- execute each candidate or trajectory branch,
- compute verifiable rewards,
- compare candidates within the group,
- update toward actions that improve success/progress without violating safety.

### Alternatives

PPO is a mature alternative but requires value estimation and can be more
complex for long-horizon, sparse-reward agent tasks.

RLOO/REINFORCE++ style methods are viable if the team wants a simpler
policy-gradient baseline.

DPO-style offline training can be used after rollout collection by turning
successful and failed trajectories into preferences, but it is not the primary
online RL method here.

## Dataset And Curriculum

### Task Source

The primary task source is Terminal-Bench 2.0. Public benchmark tasks should be
used carefully because contamination and overfitting are real risks.

### Split Strategy

Use task-family stratification:

- easy training tasks,
- medium training tasks,
- hard training tasks,
- held-out validation tasks,
- held-out final evaluation tasks.

Avoid mixing near-duplicate tasks across train and validation splits.

### Curriculum

Stage 1: Smoke tasks.

- Validate action protocol.
- Validate Harbor adapter.
- Validate reward logging.
- Validate final verifier execution.

Stage 2: Easy coding/system tasks.

- Shorter horizons.
- Lower timeout risk.
- More interpretable failures.

Stage 3: Medium tasks.

- Multi-file edits.
- Debugging loops.
- Dependency and build issues.

Stage 4: Hard tasks.

- Long horizons.
- Sparse final rewards.
- Larger context requirements.
- More realistic software-engineering behavior.

### Rollout Scale Plan

Pilot:

- 25-50 tasks,
- 4-8 samples per prompt,
- 1-3 epochs,
- manual failure review.

Medium scale:

- 50-100 tasks,
- 8 samples per prompt,
- curriculum progression,
- automated metrics dashboard.

Large scale:

- full available task set plus synthetic variants,
- distributed rollout generation,
- held-out evaluation by task family,
- periodic regression checks.

## Hyperparameter Plan

Pilot defaults:

```text
algorithm: GRPO
base_model: Qwen2.5-Coder-7B-Instruct
num_generations_per_prompt: 8
learning_rate: 1e-6
temperature: 0.7
top_p: 0.95
max_prompt_tokens: 8192
max_completion_tokens: 2048
kl_coefficient_start: 0.02
max_grad_norm: 1.0
```

Environment defaults:

```text
max_steps_per_episode: 30
max_wall_time_seconds: 3600
progress_reward_cap: 0.20
success_reward: 1.0
integrity_penalty: -1.0
step_penalty: -0.01 * step_count
```

The first tuning target is not leaderboard score. It is stable reward behavior:
success should dominate, progress should correlate with final success, and
integrity violations should remain near zero.

## Evaluation Metrics

Primary metrics:

- pass rate on held-out Terminal-Bench tasks,
- pass rate by task family,
- average return,
- final success reward,
- timeout rate,
- integrity violation rate.

Training diagnostics:

- KL divergence to reference policy,
- reward component histograms,
- clip ratio,
- entropy,
- average steps per successful episode,
- average tokens per successful episode,
- command failure rate,
- public-test overfitting rate.

Qualitative diagnostics:

- trajectory review,
- failure taxonomy,
- reward hacking examples,
- safety violation review,
- cases where progress reward disagrees with final success.

## Safety And Anti-Cheating

The environment should forbid:

- editing tests,
- editing oracle solutions,
- editing graders,
- reading hidden verifier internals,
- destructive system commands,
- network access unless explicitly allowed,
- benchmark-specific memorization shortcuts.

The current code implements a small local safety layer for path and command
screening. The real Harbor adapter should additionally use container-level
permissions, filesystem mounts, network isolation, and verifier separation.

## Scaling Plan

### Local Prototype

Use the mock environment and unit tests to verify schema and reward behavior.

### Harbor Integration

Implement a real `HarborTerminalBenchEnv` behind the same interface. The mock
environment should remain as the fast unit-test target.

### TRL Pilot

Use TRL GRPO for a small training experiment after real rollouts are available.
The goal is reward validation and action-protocol validation.

### verl Large-Scale Training

Use verl when scaling beyond pilot:

- distributed actor workers,
- reference policy workers,
- high-throughput generation,
- sharded training,
- checkpoint storage,
- rollout dataset storage,
- periodic held-out evaluation.

The code is arranged so the environment and reward semantics do not change
between TRL pilot and verl scale-up.

## Code Package Overview

The implemented package contains:

- `src/tb_rlvr/actions.py`: action parser and action serialization.
- `src/tb_rlvr/observations.py`: observation schema and prompt rendering.
- `src/tb_rlvr/rewards.py`: reward components and scalar combination.
- `src/tb_rlvr/safety.py`: local safety checks.
- `src/tb_rlvr/rollout.py`: rollout JSONL and training-sample export.
- `src/tb_rlvr/env.py`: mock Terminal-Bench-style environment.
- `src/tb_rlvr/trainers/export.py`: backend-neutral trainer export.
- `src/tb_rlvr/readiness.py`: training-readiness reporting.
- `configs/training/`: Harbor, TRL, and verl configuration skeletons.
- `tests/`: fast local validation suite.

## Validation

The local validation commands are:

```bash
python3 -m pytest
python3 examples/run_mock_rollout.py
python3 examples/check_training_readiness.py
```

Expected state:

- tests pass,
- mock rollout emits a successful JSONL record,
- readiness report marks core repo files ready and external training packages
  as intentionally not installed.

## Known Limitations

This repo does not yet:

- install or run Harbor,
- execute real Terminal-Bench tasks,
- run TRL GRPO,
- run verl distributed training,
- fine-tune any model,
- claim benchmark improvement.

These are outside the no-training deliverable. The design keeps them as
straight-line follow-up work rather than unresolved environment questions.

## Next Implementation Steps

1. Implement `HarborTerminalBenchEnv` behind the current `reset/step` contract.
2. Run one real Terminal-Bench smoke task with a scripted policy.
3. Generate 50-100 pilot rollouts with a frozen model.
4. Review reward component distributions and failure cases.
5. Launch a small TRL GRPO pilot.
6. Move to verl only after the reward and action protocol are stable.

## References

- Terminal-Bench project and task harness documentation:
  https://github.com/laude-institute/terminal-bench
- Harbor project documentation:
  https://github.com/laude-institute/harbor
- TRL GRPO documentation:
  https://huggingface.co/docs/trl/main/en/grpo_trainer
- verl documentation:
  https://verl.readthedocs.io/
- PPO:
  https://arxiv.org/abs/1707.06347
- InstructGPT/RLHF:
  https://arxiv.org/abs/2203.02155
- DeepSeekMath and GRPO:
  https://arxiv.org/abs/2402.03300
- DeepSeek-R1:
  https://arxiv.org/abs/2501.12948
- CodeRL:
  https://arxiv.org/abs/2207.01780
- SWE-agent:
  https://arxiv.org/abs/2405.15793
- Agent Lightning:
  https://arxiv.org/abs/2508.03680
