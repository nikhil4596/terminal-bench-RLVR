# Final Take-Home Solution Plan

This is the implementation-ready plan for the AfterQuery Research Scientist (Post-Training) take-home. It assumes we will **not train an RL model**. The deliverable is an environment/reward/training-design package plus a private repo link. Everything after this plan should be executable by another engineer or agent without needing to make major design decisions.

Assignment PDF summary:

- Select a popular, open-source RL framework.
- Choose `terminalbench2` or `tau2 bench`.
- Design an RLVR environment for improving model performance on the chosen benchmark.
- Define observation/state space.
- Define action space.
- Design at least two virtual reward functions.
- Select a base model.
- Select and justify an RL algorithm.
- Specify dataset size/configuration, curriculum, hyperparameters, and metrics.
- Submit a write-up and a private code repo link.
- Optional stretch: synthetic task, dual environment, scaling discussion.

## Final Scope

### In Scope

- Four Markdown docs:
  - ecosystem guide,
  - RL modeling foundations,
  - Terminal-Bench 2 strategy review,
  - final solution plan.
- A private repo skeleton that defines:
  - Harbor/Terminal-Bench environment adapter design,
  - structured action schema,
  - observation schema,
  - reward functions,
  - rollout log schema,
  - mock smoke tests,
  - future trainer config.
- A final take-home write-up that can be submitted as the assignment document.

### Out Of Scope

- Running GRPO/PPO.
- Fine-tuning model weights.
- Running all 89 Terminal-Bench tasks.
- Claiming benchmark improvement.
- Publishing official benchmark data, hidden tests, or oracle solutions.

### Success Criterion

At the end, the repo should prove:

```text
If we had compute and a clean task split,
we could plug in a policy model,
collect Terminal-Bench-style rollouts,
compute rewards,
and hand trajectories to a GRPO trainer.
```

The remaining work should be training, not environment design.

## Main Recommendation

Choose:

- Benchmark: **Terminal-Bench 2.0**.
- Environment harness: **Harbor**.
- Prototype implementation: **Harbor-first RLVR substrate**, not a trainer-first demo.
- Future RL framework: **TRL/GRPO for small research pilot**, **SkyRL or OpenRLHF for multi-turn agentic training**, **verl for large-scale production training**.
- Base model: **Qwen2.5-Coder-7B-Instruct** for pilot training plan.
- Algorithm: **GRPO**, with PPO/RLOO/REINFORCE++ as alternatives.

Why Harbor-first:

- The assignment is about an RLVR environment.
- Terminal-Bench 2.0 is run through Harbor.
- We are not training, so a correct environment and rollout substrate is more valuable than importing a trainer.
- Future trainer integration can be described without GPU work.

Why Terminal-Bench 2:

- It is terminal-native and directly relevant to coding/post-training.
- It has executable verifiers and outcome-based tests.
- It stresses long-horizon tool use, debugging, systems work, and code tasks.
- It maps cleanly to RLVR.

Why not tau2-bench:

- tau2 is strong, but it adds user simulation and dual-control dialogue.
- The assignment is for RLVR-style post-training, and the user's stated interest is terminal/coding agents.
- Terminal-Bench gives a cleaner first environment: shell/file actions plus verifiable final state.

## Critique Of The Naive Plan

The earlier naive plan had the right broad instincts:

- It identified Terminal-Bench 2 as a good benchmark.
- It recognized state/action/reward as core sections.
- It mentioned GRPO and code-specialized models.
- It compared frameworks at a high level.

But it is not strong enough for a research-scientist take-home.

Major problems:

- It treats the assignment as a generic essay rather than an implementable environment design.
- It over-centers TRL as if trainer code is the prototype, even though the no-training deliverable should be Harbor/environment-first.
- It does not define clean module boundaries.
- It does not specify rollout schemas.
- It does not handle benchmark contamination rigorously.
- It does not distinguish public progress tests from hidden final tests.
- It does not define integrity/tamper detection.
- It does not explain how to keep model improvements separate from scaffold improvements.
- It does not discuss current version drift in Terminal-Bench and tau2-bench.
- It does not provide enough metrics for diagnosing reward hacking.
- It does not identify what remains after the assignment: actual training.

The improved plan fixes those issues by making the environment/reward substrate the core deliverable.

## Final System Design

### Layered Architecture

```text
docs/
  learning and submission write-up

src/tb_rlvr/
  actions.py
  observations.py
  rewards.py
  rollout.py
  env.py
  safety.py
  trainers/
    grpo_config.py
  mock/
    fake_harbor.py

tests/
  test_actions.py
  test_rewards.py
  test_rollout.py
  test_safety.py
  test_mock_env.py

examples/
  toy_task/
  scripted_policy.py
  run_mock_rollout.py
```

The code repo does not need to be large. It needs to be coherent.

### Runtime Flow

```text
reset(task_id)
  -> load task instruction and environment metadata
  -> launch or mock container
  -> build initial observation

policy(observation)
  -> emits structured text action

parse_action(text)
  -> AgentAction(kind, payload, metadata)

env.step(action)
  -> safety precheck
  -> execute bash/patch/finish
  -> collect stdout/stderr/filesystem diff
  -> compute reward components
  -> build next observation
  -> append rollout record

termination
  -> success, timeout, safety violation, max steps, or finish
```

### Core Interfaces

```python
class TerminalBenchEnv:
    def reset(self, task_id: str, seed: int | None = None) -> Observation: ...
    def step(self, action: AgentAction) -> StepResult: ...
```

```python
class AgentAction:
    kind: Literal["bash", "patch", "finish"]
    payload: str
    path: str | None
```

```python
class Observation:
    task_id: str
    instruction: str
    cwd: str
    directory_summary: str
    recent_history: list[str]
    last_stdout: str
    last_stderr: str
    selected_files: dict[str, str]
    steps_remaining: int
```

```python
class RewardComponents:
    success: float
    progress: float
    integrity: float
    efficiency: float
    total: float
```

```python
class StepResult:
    observation: Observation
    reward: RewardComponents
    done: bool
    info: dict
```

### Action Protocol

Use simple XML-like tags because they are readable and easy to parse:

```text
<bash>
pytest -q
</bash>
```

```text
<patch path="/app/src/main.py">
diff or replacement block
</patch>
```

```text
<finish>
ready for grading
</finish>
```

Rules:

- Exactly one action per assistant turn.
- Invalid action receives an invalid-action penalty and observation explaining the parser error.
- `patch` must target files under allowed workspace paths.
- `finish` triggers verifier evaluation.
- `bash` commands are checked for forbidden patterns before execution.

### Observation Design

Include:

- task instruction,
- current directory,
- summarized file tree,
- last command output,
- last error output,
- relevant selected file snippets,
- recent actions,
- remaining steps,
- public diagnostic status.

Exclude:

- hidden tests,
- oracle solution,
- task metadata that reveals the answer,
- protected grader details,
- internet-fetched public solutions.

Observation should be bounded. Default caps:

- last stdout: 6,000 chars,
- last stderr: 6,000 chars,
- file snippet total: 12,000 chars,
- history: last 8 actions,
- max action output tokens: trainer-dependent, usually 512-1024.

### Reward Design

#### Reward 1: Final Success Reward

```text
R_success = 1.0 if hidden/final verifier passes else 0.0
```

Properties:

- Dominates all other rewards.
- Runs on `finish`, timeout, or periodic hidden-check trigger if allowed.
- Uses final container state.
- Does not expose hidden test output to the policy.

#### Reward 2: Progress Reward

```text
R_progress = min(0.20, sum(progress_events))
```

Potential progress events:

- public failing-test count decreases,
- required artifact exists,
- build succeeds after previously failing,
- linter/static check improves,
- target service starts,
- model creates a reasonable test/reproduction script.

Rules:

- Capped at 0.20 per episode.
- Never enough to beat a final successful trajectory.
- Public diagnostics only.
- Logged separately so we can audit correlation with final success.

#### Reward 3: Integrity Penalty

```text
R_integrity = -1.0 and done=True if tampering is detected
```

Tampering examples:

- editing `tests/`,
- editing `solution/`,
- editing task metadata,
- disabling grader scripts,
- deleting required verifier files,
- monkeypatching test runner,
- fetching public oracle answer,
- modifying PATH to fake expected tools,
- changing hidden-eval assumptions.

This is a hard constraint, not just a soft preference.

#### Reward 4: Efficiency Penalty

```text
R_efficiency = -0.01 per environment step
```

Optional token penalty:

```text
R_token = -0.00001 per generated token
```

Keep efficiency small; do not incentivize premature finishing.

### Reward Formula

```text
R_total =
    R_success
  + R_progress
  + R_integrity
  + R_efficiency
  + R_token
```

Examples:

- Solves task in 12 steps:
  - success 1.0, progress 0.12, efficiency -0.12, total about 1.0.
- Makes progress but fails:
  - success 0.0, progress 0.2, efficiency -0.3, total about -0.1.
- Edits tests:
  - integrity -1.0, done, total around -1.0.

## Future Training Plan

This is not executed in the assignment, but the write-up must specify it.

### Stage 0: Environment Validation

- Use toy tasks.
- Use scripted policies.
- Confirm reward and safety behavior.
- Confirm rollout logs are replayable.

### Stage 1: Baseline Rollouts

- Model: Qwen2.5-Coder-7B-Instruct.
- No weight updates.
- Run on synthetic or dev tasks.
- Collect pass/fail traces.
- Measure action-format compliance.

### Stage 2: Rejection Fine-Tuning

- Filter successful traces.
- Remove traces with suspicious shortcuts.
- Fine-tune on clean action trajectories.
- Goal: teach action protocol and basic terminal workflow.

### Stage 3: GRPO

- Sample 4-8 rollouts per task.
- Compute trajectory rewards.
- Normalize advantages within task group.
- Update policy with GRPO.
- Track KL to base/reference policy.

### Stage 4: Held-Out Evaluation

- Fixed scaffold.
- Same prompts and action protocol.
- Base model vs trained model.
- Pinned task versions.
- No internet unless explicitly configured.
- Report pass rate and cost.

## Training Configuration Defaults

### Model

Primary:

```text
Qwen/Qwen2.5-Coder-7B-Instruct
```

Justification:

- open weights,
- Apache-2.0 license,
- code-specialized,
- instruction-tuned,
- long-context support,
- feasible for LoRA pilot.

Scale-up alternatives:

- Qwen2.5-Coder-32B-Instruct,
- Qwen3-Coder variants,
- DeepSeek-Coder style models,
- model served through vLLM/SGLang.

### Algorithm

Primary:

```text
GRPO
```

Justification:

- natural for multiple rollouts per task,
- avoids separate value model,
- strong fit for verifiable rewards,
- popular in modern RLVR reasoning work.

Fallbacks:

- PPO if we want classic RLHF baseline,
- RLOO/REINFORCE++ if using OpenRLHF-style training,
- DPO for offline pairwise trace improvement.

### Hyperparameters

Initial pilot values:

```yaml
model: Qwen/Qwen2.5-Coder-7B-Instruct
fine_tuning: LoRA
lora_rank: 16
lora_alpha: 32
learning_rate: 1.0e-5
optimizer: AdamW
weight_decay: 0.01
rollouts_per_task: 4
max_env_steps: 30
max_action_tokens: 1024
temperature: 0.7
top_p: 0.95
kl_coefficient: 0.02
grpo_clip_range: 0.2
train_batch_tasks: 8
gradient_accumulation_steps: 4
max_grad_norm: 1.0
```

These are starting values, not guaranteed optimal. The key is to specify what would be tuned:

- rollouts per task,
- progress reward cap,
- KL coefficient,
- max steps,
- action token cap,
- learning rate,
- LoRA rank,
- task curriculum.

### Dataset Configuration

Do not train on official held-out Terminal-Bench 2 evaluation tasks if making benchmark claims.

Recommended data sources:

- synthetic Terminal-Bench-style tasks,
- toy tasks for smoke tests,
- internal held-out task authoring,
- prior public terminal tasks only if not used for final claims,
- official Terminal-Bench 2.0 only as pinned evaluation substrate unless split/decontamination is explicit.

Curriculum:

```text
level 0: toy file creation and command tasks
level 1: single-file code repair
level 2: multi-file build/test tasks
level 3: systems/security/data tasks
level 4: hard long-horizon tasks
```

## Metrics

### Primary Metrics

- Task pass rate.
- Pass@k or pass^k for repeated attempts.
- Average final reward.
- Hidden verifier success.

### Diagnostic Metrics

- Steps to success.
- Tokens to success.
- Cost per solved task.
- Invalid action rate.
- Parser failure rate.
- Safety/tamper violation rate.
- Public progress reward vs final success correlation.
- KL divergence to reference policy.
- Reward component distribution.
- Timeout rate.
- Environment crash rate.

### Failure Taxonomy

Classify failures into:

- execution errors: command failed, dependency issue, wrong file edited,
- coherence errors: plan drift, inconsistent state tracking,
- verification errors: did not run/check tests, false belief of success,
- safety errors: attempted forbidden shortcut,
- context errors: missed relevant file/output,
- reward errors: progress reward misleading.

This mirrors the spirit of Terminal-Bench trajectory analysis without requiring full benchmark runs.

## Private Repo Implementation Checklist

### Files

Current prototype files:

```text
docs/
  01_rlvr_framework_ecosystem.md
  02_rl_modeling_foundations.md
  03_terminalbench2_modeling_strategy_review.md
  04_take_home_solution_plan.md
src/tb_rlvr/
  __init__.py
  actions.py
  observations.py
  rewards.py
  safety.py
  rollout.py
  env.py
tests/
  test_actions.py
  test_rewards.py
  test_safety.py
  test_rollout.py
  test_mock_env.py
examples/
  run_mock_rollout.py
```

Optional final submission polish can add `README.md`, a concise
`assignment_writeup.md`, a Harbor adapter around `env.py`, and a future
training config module. These are packaging improvements, not required for the
no-training environment prototype.

### Minimal Behavior

- Parse valid `bash`, `patch`, `finish` actions.
- Reject invalid multi-action outputs.
- Build bounded observations.
- Compute reward components.
- Detect protected path edits.
- Serialize rollout records.
- Run a fake scripted task end to end.

### Acceptance Tests

- `test_action_parser_accepts_bash`
- `test_action_parser_rejects_two_actions`
- `test_patch_rejects_tests_directory`
- `test_success_reward_dominates_progress`
- `test_progress_reward_is_capped`
- `test_integrity_violation_terminates`
- `test_rollout_jsonl_roundtrip`
- `test_mock_env_scripted_success`
- `test_mock_env_timeout`

## Final Write-Up Outline

The submitted document should use this structure:

1. **Problem framing**
   - RLVR environment for Terminal-Bench 2.
   - No training run; design/prototype only.

2. **Framework selection**
   - Harbor as execution harness.
   - TRL/GRPO as future small-scale trainer.
   - SkyRL/OpenRLHF/verl as scale-up paths.

3. **Benchmark selection**
   - Choose Terminal-Bench 2.0.
   - Justify over tau2-bench.
   - Discuss versioning and contamination.

4. **Environment design**
   - State/observation.
   - Action space.
   - Transition/termination.
   - Safety.

5. **Reward design**
   - Final success.
   - Progress reward.
   - Integrity penalty.
   - Efficiency penalty.

6. **Model and algorithm**
   - Qwen2.5-Coder-7B-Instruct.
   - GRPO.
   - DPO/RFT as auxiliary baselines.

7. **Training plan**
   - Dataset/curriculum.
   - Hyperparameters.
   - Evaluation.

8. **Metrics and risks**
   - Pass rate, cost, KL, tamper rate.
   - Reward hacking and contamination.

9. **Stretch goals**
   - Synthetic task.
   - tau2 environment sketch.
   - scale-up model discussion.

10. **Repo link**
   - Private GitHub repo with code skeleton and docs.

## Defense Of The Plan

### Why This Is Better Than Training Immediately

Running RL before validating the environment is low-signal. A bad reward or leaky evaluation will produce misleading results. The highest-value work for a take-home is building the substrate:

- correct task execution,
- clear action/observation interface,
- safe reward functions,
- replayable rollouts,
- credible future trainer path.

This demonstrates research judgment.

### Why Harbor-First Is Correct

Harbor is the official Terminal-Bench execution substrate. Since the assignment is to create an RLVR environment, the environment/harness is more central than the optimizer. TRL/SkyRL/verl can only be useful after we can generate clean rollouts and rewards.

### Why GRPO Is Still The Right Future Algorithm

GRPO maps well to verifiable tasks:

- multiple attempts per same task,
- group-relative advantages,
- no learned value model,
- strong precedent in RLVR reasoning.

The plan does not pretend GRPO solves all credit assignment. It says GRPO is the first practical optimizer once the environment works.

### Why Qwen2.5-Coder-7B-Instruct

It is feasible, open, code-specialized, and instruction-tuned. Starting with a huge model would make the training plan unrealistic. Starting with a base non-instruct model would waste compute teaching basic action formatting.

### Why The Reward Mix Is Defensible

Final success matches the benchmark. Progress helps exploration but is capped. Integrity penalty prevents cheating. Efficiency penalty prevents loops. This reward structure is understandable, auditable, and testable.

## Expert Pressure-Test Questions

1. How do you know progress reward improves final success rather than teaching proxy behavior?
2. What prevents the agent from editing public tests or task metadata?
3. How do you prevent contamination from public Terminal-Bench tasks and oracle solutions?
4. Why choose Terminal-Bench over tau2 if tau2 has more explicit user/tool state?
5. Why choose GRPO over PPO, RLOO, REINFORCE++, or DAPO?
6. What happens if all GRPO rollouts for a hard task fail?
7. How do you assign credit across a 50-step terminal trajectory?
8. How do you separate scaffold gains from model-weight gains?
9. What hidden evaluation would convince you the trained model generalizes?
10. How do you handle flaky tests and nondeterministic package installs?
11. Should the policy see public test output? If yes, how do you avoid overfitting?
12. What is the right action granularity: raw bash, file patches, or keystrokes?
13. How do you track KL when the model generates multi-action trajectories?
14. What does the rollout format need so DPO, GRPO, and RFT can all use it later?
15. How would you scale from 7B LoRA to a 32B or MoE model?
16. How do you keep the environment reproducible across Docker/Daytona/Modal backends?
17. What failure modes would indicate reward hacking?
18. What would you monitor during the first 500 RL updates?
19. How would you add tau2-bench as a second environment without rewriting the trainer?
20. What claims are you explicitly not making because you did not train?

## Stretch Goals

### Synthetic Task Generation

Create one toy Terminal-Bench-style task:

- instruction: create a Python script that transforms a CSV and writes exact output,
- environment: small Docker image or local mock,
- tests: hidden expected output,
- solution: oracle script,
- reward: final test pass plus progress for creating the file.

This demonstrates the task-authoring path without touching official benchmark data.

### tau2-Bench Sketch

Define a tau2 environment variant:

- observation: dialogue history, agent tool results, policy text,
- action: message or tool call,
- transition: user simulator plus shared world state,
- reward: final DB/status assertions plus communication checks.

State clearly that tau2 is a future extension because the user simulator introduces nonstationarity and dialogue-coordination complexity.

### Scaling Discussion

Small model:

- LoRA,
- shorter context,
- more curriculum,
- more RFT before RL,
- smaller batch sizes.

Large model:

- vLLM/SGLang serving,
- async rollouts,
- SkyRL/OpenRLHF/verl,
- stronger KL/clip monitoring,
- more aggressive decontamination,
- cost-aware evaluation.

## What Is Needed Next

The immediate next step is to implement the repo skeleton and write the final assignment document from these docs.

After that, the only remaining real work should be:

1. author or select a decontaminated training task set,
2. collect rollouts,
3. run RFT/SFT,
4. run GRPO or a similar algorithm,
5. evaluate on held-out pinned tasks,
6. analyze failures and reward hacking.

That is beyond the assignment boundary.
