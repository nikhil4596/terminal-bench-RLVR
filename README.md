# RLVR Environment Design for Terminal-Bench 2.0 Transfer

## Executive Summary

This repository defines an RLVR environment for improving terminal-agent
performance on Terminal-Bench 2.0 while keeping Terminal-Bench 2.0 strictly as
held-out evaluation. The training environment is a set of independently
generated Harbor-format terminal tasks with executable verifiers. The proposed
training stack uses Harbor for containerized task execution, Terminus-2 as the
terminal-agent scaffold, and SkyRL-Agent/SkyRL for multi-turn RL.

The core data boundary is:

```text
synthetic Harbor tasks -> SFT and RLVR training -> Terminal-Bench 2.0 evaluation
```

Selected components:

- Target benchmark: Terminal-Bench 2.0, used only for final evaluation.
- Environment substrate: Harbor task format and Harbor rollout execution.
- Agent scaffold: Terminus-2 or a Terminus-compatible terminal loop.
- RL framework: SkyRL-Agent/SkyRL.
- Base model: `Qwen/Qwen2.5-Coder-7B-Instruct` for the pilot.
- RL algorithm: GRPO after SFT warm start.
- Reward source: verifier-backed correctness from `/logs/verifier/reward.json`,
  with correctness-gated efficiency and integrity penalties.

The repository contains one concrete synthetic Harbor task, a deterministic task
generator, task validation utilities, local action/observation/reward/rollout
contract specifications, and training configuration skeletons. It does not run
model training or claim a benchmark improvement.

## Benchmark Selection

### Selected Benchmark: Terminal-Bench 2.0

Terminal-Bench 2.0 evaluates autonomous agents on realistic terminal tasks.
Tasks require command-line interaction, file inspection, code or data
manipulation, debugging, and final-state verification. This makes it a natural
target for RLVR because the environment can produce executable rewards rather
than subjective preference labels.

Terminal-Bench 2.0 is not used as training data. The public benchmark contains
89 tasks, and the benchmark paper discusses contamination risk when model
developers train on public benchmark tasks. This submission therefore treats
Terminal-Bench 2.0 as a final held-out benchmark:

- no Terminal-Bench 2.0 task instructions in SFT or RL training;
- no Terminal-Bench 2.0 tests or oracle solutions in training;
- no rollout traces from Terminal-Bench 2.0 in training;
- no generated task intentionally copies TB2 task names, prompts, tests,
  filenames, or oracle solutions;
- final evaluation reports pass@1 over all 89 TB2 tasks.

## Framework Selection

### Harbor

Harbor is selected as the environment substrate. Harbor defines tasks as
containerized directories with instructions, environments, verifiers, and
solutions. It provides the execution boundary needed for terminal-agent RL:

- isolated container runtime;
- task instruction and workspace;
- agent/verifier separation;
- `/app`, `/tests`, `/solution`, and `/logs/verifier` conventions;
- verifier reward files such as `/logs/verifier/reward.json`;
- support for ATIF trajectories;
- local and cloud sandbox execution options.

ATIF stands for **Agent Trajectory Interchange Format**. It is Harbor's
standard JSON-based trajectory format for recording the complete interaction
history of an autonomous agent: messages, tool calls, terminal observations,
token usage, logprobs, and related metadata. ATIF trajectories are useful for
debugging, supervised fine-tuning, and RL rollout analysis.

### Terminus-2

Terminus-2 is the recommended agent scaffold for production rollouts. It is
Harbor's reference terminal agent and already supports JSON parser
configuration, ATIF trajectory generation, verifier reward collection, and RL
rollout metadata. With rollout-detail collection enabled, Terminus-2 can record
prompt token ids, completion token ids, and logprobs. Those fields are required
for practical on-policy RL.

The repository's local action and observation contracts are therefore not
intended to replace Terminus-2 internals. They specify the expected semantics of
the agent loop and provide a compact validation target. If a custom Harbor
agent is implemented instead of stock Terminus-2, these contracts can be used
directly.

### SkyRL-Agent/SkyRL

SkyRL-Agent/SkyRL is selected as the RL framework because the target problem is
multi-turn, long-horizon agent RL. A terminal task rollout involves model
generation, command execution, environment feedback, verifier execution, and
trajectory logging. SkyRL's custom generator interface is a closer fit for this
workflow than a simple prompt/completion trainer.

TRL and VeRL remain useful alternatives: TRL is suitable for small or
single-turn GRPO experiments, and VeRL is a strong distributed RL backend. The
primary design here centers on the Harbor/SkyRL agent-environment interface.

## Runtime Architecture

The system has three contracts:

```text
1. Harbor task contract
   task.toml / instruction.md / environment / tests / solution
   -> consumed directly by Harbor

2. Agent contract
   visible observation -> policy -> JSON terminal action -> stdout/stderr
   -> owned by Terminus-2 or a custom Harbor agent

3. Training rollout contract
   completed Harbor trial -> rewards + ATIF + token ids + masks + logprobs
   -> converted by a SkyRL generator into training batches
```

The local Python dataclasses represent the second and third contracts as
testable specifications. Harbor itself consumes the task directories directly.
In a production run, the adapter would launch Harbor jobs with Terminus-2,
extract reward and trajectory metadata from Harbor trial results, optionally log
local `RolloutRecord` rows for audit, and return SkyRL `GeneratorOutput`
batches to the trainer.

The training loop is:

```text
SkyRL requests rollouts
  -> Harbor/SkyRL adapter selects synthetic tasks
  -> Harbor runs Terminus-2 against each task
  -> verifier writes reward.json
  -> adapter reads verifier rewards, ATIF, token ids, masks, and logprobs
  -> SkyRL/GRPO computes advantages and updates the policy
```

The trainer does not normally replay the environment for each optimizer
minibatch. The environment interaction happens during rollout generation. The
trainer then recomputes current-policy or reference-policy logprobs on the
sampled trajectories and applies the GRPO objective.

## Environment Design

### Training Task Source

Training tasks live under `tasks/synthetic-*` and follow Harbor's task layout:

```text
task.toml
instruction.md
environment/Dockerfile
environment/workspace/...
tests/test.sh
solution/solve.sh
```

The included seed task, `synthetic-event-audit-001`, asks the agent to inspect
an event log and create a structured JSON report. The verifier recomputes the
expected answer, checks the submitted report, and writes a correctness reward to
`/logs/verifier/reward.json`.

The current generator creates randomized instances of this event-audit task. A
larger training run should expand generation into multiple skill families:

- file and directory manipulation;
- log and data processing;
- dependency and environment repair;
- CLI program construction;
- repository debugging;
- service configuration.

The task validator checks that generated tasks have required Harbor files, write
a correctness reward, avoid known Terminal-Bench 2.0 task ids, and can
optionally run oracle-pass and dummy-fail checks in a Linux/Harbor environment.

### Observation Space

The policy observes the visible terminal-agent state:

```text
task id
instruction
current working directory
terminal prompt
recent terminal turns
ATIF history summary
last stdout
last stderr
remaining turn budget
```

The observation excludes tests, oracle solutions, hidden verifier internals, and
synthetic file-content leaks that would not appear in normal terminal
interaction. The model must inspect the visible workspace through terminal
commands.

### Action Space

The action space is one JSON terminal turn per model response:

```json
{"rationale": "inspect the workspace", "command": "ls -la", "task_complete": false}
```

The stop action is:

```json
{"rationale": "report created", "command": "", "task_complete": true}
```

File edits are performed through shell commands, editors, or scripts inside the
container. There is no custom privileged patch action. This keeps training
behavior aligned with the terminal-agent scaffold used for final evaluation.

## Reward Design

The reward is verifier-backed and decomposed into auditable components.

### Correctness Reward

```text
R_correctness = passed_verifier_checks / total_verifier_checks
```

Correctness is read from the task verifier's `reward.json`. It is the dominant
reward term because benchmark performance ultimately depends on solving the
task.

### Efficiency Reward

```text
R_efficiency = R_correctness * f(turn_count, token_count)
```

Efficiency is correctness-gated. A failed short attempt receives no efficiency
reward. Among equally correct attempts, shorter trajectories are preferred.

### Integrity Penalty

```text
R_integrity = -1.0
```

The integrity penalty is applied when the agent attempts to access or modify
protected paths such as `/tests`, `/solution`, `/oracle`, or verifier logs.
This discourages reward hacking and benchmark tampering.

### Combined Reward

```text
R_total = 0.80 * R_correctness + 0.20 * R_efficiency + R_integrity
```

These weights are pilot defaults. They should be tuned after inspecting reward
histograms, integrity failures, and held-out synthetic task performance.

## Model and Training Plan

### Base Model

Pilot model: `Qwen/Qwen2.5-Coder-7B-Instruct`.

This model is open, coding-specialized, practical for pilot-scale LoRA
fine-tuning, and suitable for shell/Python/JSON workflows. Larger follow-up
runs can use Qwen3-Coder or larger Qwen-Coder variants after the environment and
reward design are validated.

### SFT Warm Start

Before RL, collect successful synthetic Harbor trajectories and train the model
to imitate complete terminal workflows:

```text
inspect -> plan -> edit/run commands -> verify -> fix -> finish
```

SFT teaches the action grammar and terminal workflow before sparse RL rewards
are introduced.

Pilot SFT configuration:

```text
samples: 500-2,000 successful synthetic traces
method: LoRA SFT
lora_rank: 32
learning_rate: 2e-5
batch_size: 64-128 effective
max_context_tokens: 32768
schedule: cosine
checkpoint selection: held-out synthetic dev loss and task success
```

### RLVR Phase

RL uses SkyRL-Agent/SkyRL with Harbor rollouts and GRPO.

Initial configuration:

```text
algorithm: GRPO
group_size: 16
max_turns: 20
temperature: 0.7
top_p: 0.95
learning_rate: 1e-6
optimizer: AdamW
kl_coefficient: 0.02
max_grad_norm: 1.0
dtype: bfloat16
lora_rank: 32
reward: correctness + correctness-gated efficiency + integrity penalty
```

Task selection:

- estimate the SFT model's solve rate on generated tasks;
- keep RL tasks with approximately 10-80% solve rate;
- exclude tasks that are always solved or never solved;
- preserve group metadata so GRPO can compare multiple sampled trajectories for
  the same task/prompt.

Rollout records or SkyRL generator outputs must preserve:

- prompt token ids;
- completion token ids;
- loss masks;
- rollout logprobs;
- rewards and reward components;
- terminal reason;
- ATIF trajectory reference or content;
- task id and group id.

## Dataset Configuration

Pilot scale:

```text
synthetic task families: 3-5
generated tasks: 100-300
SFT traces: 500-2,000 successful traces
RL task filter: 10-80% SFT solve rate
synthetic dev/test: split by generator seed and skill family
Terminal-Bench 2.0: held-out final evaluation only
```

Scale-up:

```text
synthetic tasks: 1,000-5,000+
rollout samples per task: 8-16
sandbox execution: Harbor local/cloud providers
periodic evaluation: synthetic held-out tasks
final evaluation: Terminal-Bench 2.0
```

## Metrics

Training metrics:

- verifier correctness reward;
- binary synthetic task success;
- efficiency reward;
- integrity violation rate;
- timeout rate;
- verifier failure rate;
- average turns per episode;
- average generated tokens per episode;
- KL to reference policy;
- GRPO group reward variance.

Evaluation metrics:

- Terminal-Bench 2.0 pass@1 over all 89 tasks;
- repeated-run average if compute allows;
- pass@k as secondary analysis;
- failure taxonomy;
- average turns and tokens per successful task.

## Code Package

Important repository components:

- `tasks/synthetic-event-audit-001/`: one working Harbor-format synthetic task.
- `src/tb_rlvr/task_generation.py`: deterministic synthetic task generator.
- `src/tb_rlvr/validate_tasks.py`: structure, reward, and contamination checks.
- `src/tb_rlvr/contracts/actions.py`: local JSON action contract.
- `src/tb_rlvr/contracts/observations.py`: local observation contract.
- `src/tb_rlvr/contracts/rewards.py`: verifier-backed reward combiner.
- `src/tb_rlvr/contracts/safety.py`: local integrity prechecks.
- `src/tb_rlvr/contracts/rollout.py`: local audit/export rollout record.
- `configs/training/harbor_rollout.toml`: Harbor rollout skeleton.
- `configs/training/skyrl_grpo_pilot.toml`: SkyRL/GRPO pilot skeleton.

Local validation:

```bash
python -m pytest
python examples/validate_synthetic_tasks.py
```

Optional Linux/Harbor validation:

```bash
python examples/validate_synthetic_tasks.py --run-local
```

## Remaining Implementation Work

The remaining production work is:

1. Add more synthetic task families.
2. Run the synthetic tasks through Harbor with Terminus-2.
3. Implement a SkyRL custom generator that launches Harbor jobs and converts
   trial results into SkyRL training batches.
4. Collect successful ATIF trajectories for SFT.
5. Run SFT on synthetic trajectories.
6. Filter RL tasks by SFT solve rate.
7. Run SkyRL/GRPO.
8. Select checkpoints on held-out synthetic tasks.
9. Evaluate the selected checkpoint on Terminal-Bench 2.0.

The repository defines the environment, task format, reward design, action and
observation contracts, validation checks, and training plan. It does not include
completed Harbor/SkyRL training infrastructure or trained model weights.

## References

- Terminal-Bench 2.0:
  https://arxiv.org/abs/2601.11868
- Harbor task documentation:
  https://harborframework.com/docs/tasks
- Harbor RL documentation:
  https://harborframework.com/docs/training-workflows/rl
- Harbor Terminus-2 documentation:
  https://harborframework.com/docs/agents/terminus-2
- Harbor ATIF documentation:
  https://harborframework.com/docs/trajectory-format
- SkyRL custom generator documentation:
  https://docs.skyrl.ai/docs/tutorials/skyrl_gym_generator
- AfterQuery Terminal-Bench 2.0 case study:
  https://www.afterquery.com/blog/terminal-bench-improvement
- Endless Terminals:
  https://arxiv.org/abs/2601.16443
- Nemotron-Terminal:
  https://arxiv.org/abs/2602.21193
- TerminalTraj:
  https://arxiv.org/abs/2602.01244
- SkillSynth:
  https://arxiv.org/abs/2604.25727
- CLI-Gym:
  https://arxiv.org/abs/2602.10999
- GRPO / DeepSeekMath:
  https://arxiv.org/abs/2402.03300
