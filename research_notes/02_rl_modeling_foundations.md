# RL Modeling Foundations For RLVR And Coding Agents

This document explains the modeling ideas behind RLVR for coding and terminal agents. It is intentionally broader than the take-home implementation: the goal is to build enough conceptual fluency to defend the design in a research-scientist interview.

Primary papers and docs:

- PPO: https://arxiv.org/abs/1707.06347
- InstructGPT / RLHF: https://arxiv.org/abs/2203.02155
- DPO: https://arxiv.org/abs/2305.18290
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Process supervision: https://arxiv.org/abs/2305.20050
- CodeRL: https://arxiv.org/abs/2207.01780
- Agent Lightning: https://arxiv.org/abs/2508.03680
- ReTool: https://arxiv.org/abs/2504.11536
- Long-context multi-turn SWE RL: https://huggingface.co/papers/2508.03501

## Executive Summary

RLVR is reinforcement learning from verifiable or virtual rewards. Instead of asking humans to rank every output, we use objective checks: unit tests, exact answers, database state comparisons, compilers, linters, security checks, task-specific scripts, or environment assertions.

For Terminal-Bench 2:

- The **state** is the task instruction plus the current container-facing view: terminal output, file snippets, working directory, action history, and remaining budget.
- The **action** is a command, patch, or finish signal emitted by the model through an action protocol.
- The **transition** is the container changing after shell/file actions.
- The **reward** is final test success plus safe shaping rewards.
- The **episode** is one attempt to solve one benchmark task.
- The **policy** is a code-capable LLM.
- The **optimizer** is GRPO or PPO-style policy gradient.

The most important research idea is this:

> Coding agents turn language generation into sequential decision-making under delayed, executable rewards.

That means the hard parts are not only model size. The hard parts are reward design, action interfaces, sparse credit assignment, contamination control, environment reproducibility, and training stability.

## MDP And POMDP View

Classic RL models a task as a Markov Decision Process:

```text
state s_t
  -> policy pi(a_t | s_t)
  -> action a_t
  -> transition P(s_{t+1} | s_t, a_t)
  -> reward r_t
```

Terminal-Bench can be treated as an MDP if the full container state is available. In practice, the LLM does not receive the entire filesystem, process table, package cache, hidden tests, and environment state. It receives a text observation. Therefore the agent experiences a partially observable MDP:

```text
true container state
  -> observation builder
  -> text observation
  -> LLM action
  -> environment step
```

The observation builder is part of the agent. It decides what the model can see:

- latest command output,
- directory tree,
- selected file snippets,
- current task instruction,
- recent history,
- budget,
- diagnostic test output.

Bad observation design can make a solvable task impossible. Overly generous observation design can leak answers or hidden tests.

## Token Actions Versus Tool Actions

An LLM emits tokens. A terminal agent takes tool actions. This mismatch is central.

There are three common views:

1. **Token-level action view**
   - Every generated token is an action.
   - Reward arrives after a whole response or episode.
   - This is how language-model RL losses are usually implemented.

2. **Message-level action view**
   - A full assistant message is one action.
   - The message may contain a command or patch.
   - This is convenient for GRPO/PPO over model completions.

3. **Tool-level action view**
   - A parsed command, patch, or API call is one environment action.
   - The model text is serialized into a structured tool call.
   - This is best for environment design and logging.

For our take-home, use the tool-level view for the environment and the token-level view for optimization. That is:

- Environment sees structured actions.
- Trainer updates token probabilities that produced those actions.

## Reward Types

### Outcome Reward

Outcome reward scores the final result:

```text
R_success = 1 if hidden verifier passes else 0
```

For Terminal-Bench, this is the most faithful reward because the benchmark itself is outcome-driven.

Strengths:

- Hard to argue with.
- Directly matches evaluation.
- Avoids micromanaging agent behavior.

Weaknesses:

- Sparse.
- Expensive to evaluate.
- Provides little guidance for long trajectories.
- Can be hacked if the environment lets the agent tamper with tests.

### Progress Reward

Progress reward gives small intermediate credit:

```text
R_progress =
  +0.05 if build starts passing
  +0.05 if public failing tests decrease
  +0.03 if required artifact appears
  +0.02 if static checker improves
```

Strengths:

- Helps exploration.
- Improves credit assignment.
- Makes early training less all-or-nothing.

Weaknesses:

- Can teach proxy hacking.
- Can conflict with final success.
- Needs caps and audits.

Rule: progress reward must be bounded so it cannot dominate final success.

### Integrity Penalty

Integrity penalty prevents cheating:

```text
R_integrity = -1 and done=True if tests/oracle/task metadata are modified
```

Examples:

- editing `tests/`,
- deleting hidden checks,
- changing `task.toml`,
- reading oracle solution files,
- exploiting internet access to fetch the public answer,
- disabling the grader.

For Terminal-Bench, this is not optional. The benchmark is public, and public tasks create contamination and shortcut risk.

### Efficiency Penalty

Efficiency reward keeps agents from looping:

```text
R_step = -0.005 per environment step
R_token = -0.00001 per generated token
```

Keep this small. The goal is not to make the model rush; the goal is to avoid runaway behavior.

### Process Reward

Process reward scores intermediate reasoning or actions. It can be human-labeled, model-labeled, or inferred.

Examples:

- the plan names the right files,
- the agent writes a targeted test before editing,
- the agent localizes a bug correctly,
- the agent makes a small reversible patch,
- the agent checks its work.

Process rewards are powerful but risky. They can reward plausible-looking reasoning that does not cause success. For this assignment, mention process rewards as a future extension, not the core reward.

## PPO

PPO is the historical baseline for RLHF. It optimizes a clipped surrogate objective so each update cannot move the policy too far from the policy that generated the rollout.

Conceptually:

```text
collect rollouts
compute rewards
estimate advantages
update policy with clipped ratio
penalize KL drift
repeat
```

Why PPO mattered:

- It made RLHF practical at scale.
- InstructGPT used a PPO-style RLHF pipeline after SFT and reward-model training.
- Many LLM alignment systems inherited the same pattern.

Why PPO is hard for LLMs:

- Needs a value model or baseline.
- Requires stable reward normalization.
- Large models make rollouts and updates expensive.
- KL tuning is delicate.

For Terminal-Bench:

- PPO is a credible fallback.
- It is more machinery than needed for the take-home.
- Long-horizon terminal tasks make value estimation difficult.

## GRPO

GRPO, introduced in DeepSeekMath, is a PPO-style method that avoids a learned value function by sampling multiple completions for the same prompt and computing relative advantages within the group.

Intuition:

```text
same task prompt q
  -> sample G trajectories
  -> reward each trajectory
  -> advantage = reward - group mean
  -> increase probability of above-average trajectories
  -> decrease probability of below-average trajectories
```

Why GRPO fits RLVR:

- Verifiable tasks naturally allow multiple sampled attempts.
- Rewards vary by task difficulty; grouping normalizes within task.
- Avoiding a value model saves memory and implementation complexity.
- It is widely associated with modern open RLVR reasoning pipelines.

Failure modes:

- If every rollout fails, relative signal is weak.
- If reward shaping is noisy, group comparisons amplify noise.
- If group rewards are all nearly identical, gradients are small.
- If length normalization is wrong, the model can learn response-length artifacts.

For Terminal-Bench:

- Group by task id or task seed.
- Sample 4-8 attempts per task in the pilot plan.
- Compute final and shaped rewards per trajectory.
- Train only on model-produced action tokens, not environment output tokens.

## DPO And Preference Optimization

DPO is not an RL rollout algorithm. It trains directly from pairs:

```text
chosen trajectory > rejected trajectory
```

Why it matters:

- Simpler and more stable than PPO.
- Great for offline preference data.
- Useful after collecting successful and failed traces.

How to use for coding agents:

- Pair passing trajectory vs failing trajectory for same task.
- Pair shorter clean success vs long messy success.
- Pair test-preserving patch vs test-tampering patch.

Why it is not the main algorithm here:

- It does not explore.
- It cannot discover new strategies without sampled data.
- The assignment asks for RL algorithm design; GRPO is more directly responsive.

DPO should be described as an auxiliary offline stage, not the core RLVR phase.

## ReST, STaR, Self-Refine, And Reflexion

These methods are not the same as policy-gradient RL, but they are important because coding agents often improve through generate-check-retry loops.

### ReST

ReST repeats:

```text
generate many candidates
score with reward
filter good candidates
fine-tune on filtered candidates
repeat
```

For Terminal-Bench, ReST-like training could collect successful Harbor trajectories, then fine-tune the model to imitate them.

Risk: if the reward has loopholes, filtering amplifies them.

### STaR

STaR bootstraps reasoning traces by keeping rationales that lead to correct answers. For coding, an analogue is keeping action traces that lead to passing tests.

Risk: the model can learn explanations or command patterns that correlate with success without understanding.

### Self-Refine

Self-Refine uses the same model to critique and revise outputs. In coding, this resembles:

```text
write patch
run tests
read errors
revise patch
```

It is an agent behavior pattern, not necessarily weight training.

### Reflexion

Reflexion stores verbal lessons from failures in memory. In terminal tasks, the agent might store:

```text
The previous attempt failed because I edited the wrong config file.
Next time inspect service logs before changing nginx.conf.
```

This can improve test-time behavior and generate training data.

## Process Rewards And Credit Assignment

Sparse final rewards create a credit problem:

```text
step 1: ls
step 2: cat README
step 3: edit file A
step 4: run tests
step 5: edit file B
step 6: final tests pass
```

Which step deserves credit? Maybe step 3, maybe step 5, maybe both.

Process reward research tries to answer this by scoring steps. In coding agents, useful process signals include:

- failing-test count,
- compiler error class,
- static-analysis improvement,
- patch locality,
- file relevance,
- successful reproduction of bug,
- successful minimal test creation,
- successful rollback after bad edit.

But process rewards should remain subordinate to final hidden-test success.

## CodeRL And Execution Feedback

CodeRL used unit-test feedback for code generation. The key lesson is that executable checks are powerful rewards for code models.

Terminal-Bench generalizes this:

- not just write a function,
- operate a shell,
- inspect files,
- install/build/run,
- debug multi-step issues,
- produce final artifacts.

Execution-feedback RL extends this idea by letting the model react to compiler/test outputs across multiple repair steps. That is much closer to a terminal agent.

## Agent Lightning And Multi-Turn Agent RL

Agent Lightning is important because it frames real agents as MDPs and separates agent runtime from training. Its core insight is directly applicable:

```text
agent runtime produces traces
credit assignment turns traces into transitions
trainer consumes transitions
```

For Terminal-Bench, this means:

- Do not bake Harbor/container logic into trainer code.
- Do not bake TRL-specific assumptions into the environment.
- Keep trajectory logs replayable.
- Let future trainers consume the same logs.

This architecture is more robust than writing a one-off GRPO script.

## Long-Horizon Software Engineering RL

Recent long-context, multi-turn SWE RL work is one of the closest references for this assignment. It argues that most LLM RL focuses on single-turn math or code, while real software engineering requires stateful environment feedback.

Lessons for Terminal-Bench:

- Use a stable action protocol.
- Do not train on every token in the giant concatenated context indiscriminately.
- Manage context length aggressively.
- Track pass@1/pass@k and steps per trajectory.
- Expect rollouts to be expensive and noisy.
- Use filtered tasks and deterministic tests for training.

## Reward Hacking In Coding Environments

Reward hacking is not theoretical here. A terminal agent can:

- edit tests,
- delete assertions,
- monkeypatch imports,
- skip failing checks,
- install a fake binary earlier in PATH,
- read oracle solutions,
- use internet search to find public answers,
- create files that fool fragile tests,
- exploit test-order dependence.

Mitigations:

- run hidden tests outside the writable workspace,
- copy tests after agent actions,
- hash and protect test directories,
- deny access to oracle solution files,
- pin task images and dependency versions,
- disable or control internet access,
- inspect filesystem diffs,
- terminate on forbidden mutations,
- keep final reward separate from public progress checks.

## KL Control

RL can damage a model if updates are too aggressive. KL control keeps the trained policy near the reference policy.

Intuition:

```text
maximize reward
but penalize moving too far from base model
```

For code agents, this matters because the base model already knows syntax, shell idioms, and instruction following. RL should improve task-solving behavior without destroying general capability.

In the take-home:

- Use a frozen reference model.
- Track KL per update.
- Start with a small KL coefficient or GRPO clipping.
- Treat KL spikes as a training failure.

## Curriculum Learning

Terminal-Bench tasks vary wildly. A curriculum can help:

```text
stage 0: toy tasks, deterministic smoke tests
stage 1: short public synthetic tasks
stage 2: medium Terminal-Bench-style tasks
stage 3: hard held-out tasks
stage 4: official pinned evaluation
```

Curriculum variables:

- step budget,
- context budget,
- task category,
- number of files,
- dependency complexity,
- public-test availability,
- historical solve rate.

Do not overfit curriculum to official tasks. The goal is general terminal competence.

## Paper Map

### PPO

Contribution:

- Clipped policy optimization for stable RL updates.
- Foundation for many RLHF pipelines.

Where it fits:

- Baseline optimizer.
- Useful fallback if GRPO is unavailable.

Takeaway:

- Stable but heavy; value estimation is painful for long-horizon LLM agents.

### InstructGPT

Contribution:

- SFT plus reward model plus PPO for instruction following.

Where it fits:

- Canonical post-training pipeline.
- Shows why starting from an instruction-tuned model matters.

Takeaway:

- RL is usually not the first step. Start with a capable SFT/instruct policy.

### DPO

Contribution:

- Direct preference training without online RL.

Where it fits:

- Offline trace preference baseline.

Takeaway:

- Useful after collecting passing/failing trajectories; not enough alone for exploration.

### DeepSeekMath

Contribution:

- Introduces GRPO for math reasoning.

Where it fits:

- Main algorithmic template for RLVR.

Takeaway:

- Group-relative rewards are natural when multiple attempts can be sampled per task.

### DeepSeek-R1

Contribution:

- Shows large-scale RL can induce reasoning behaviors on verifiable tasks.

Where it fits:

- Justifies RLVR with objective rewards.

Takeaway:

- Pure RL can work, but cold-start data and readability fixes matter.

### Process Supervision

Contribution:

- Rewards reasoning steps, not only final answers.

Where it fits:

- Future extension for terminal credit assignment.

Takeaway:

- Powerful but expensive and potentially subjective.

### PRIME

Contribution:

- Attempts process-like reinforcement from implicit rewards.

Where it fits:

- Relevant for sparse long-horizon rewards.

Takeaway:

- Helpful direction, but for this assignment keep reward design simpler and auditable.

### CodeRL

Contribution:

- Uses unit-test execution feedback for code RL.

Where it fits:

- Direct predecessor to coding RLVR.

Takeaway:

- Executable rewards are strong but incomplete tests create overfitting risk.

### SWE-agent

Contribution:

- Shows agent-computer interface design matters for SWE tasks.

Where it fits:

- Justifies structured action space.

Takeaway:

- The environment interface is part of the model.

### Agent Lightning

Contribution:

- Decouples agent execution from RL training.

Where it fits:

- Architectural principle for our repo.

Takeaway:

- Keep the environment, trace format, rewards, and trainer separate.

## Modeling Defaults For The Take-Home

Use these defaults in the final plan:

- Policy: `Qwen2.5-Coder-7B-Instruct`.
- Reference: frozen copy of the same model.
- Optimizer: GRPO in TRL.
- Rollouts per task group: 4-8.
- Max environment steps: 30 for pilot, higher for hard tasks.
- Max action tokens: 512-1024.
- Learning rate: `1e-6` to `5e-6` for full fine-tune, `5e-6` to `2e-5` for LoRA.
- Batch unit: grouped task trajectories, not individual unrelated prompts.
- Reward: final success dominates, progress capped, integrity terminating, step penalty small.
- Evaluation: held-out tasks, pass rate, cost, steps, invalid actions, KL, reward-success correlation.

## What Remaining Training Would Need

The assignment stops before training. The remaining work after the docs/prototype would be:

1. Create a contamination-safe training task set.
2. Run scripted smoke tests for the environment.
3. Collect baseline rollouts from the base policy.
4. Verify reward functions correlate with final success.
5. Run small LoRA GRPO on synthetic/dev tasks.
6. Evaluate on held-out pinned Terminal-Bench tasks.
7. Compare to the base model and same scaffold.
8. Audit failures and reward hacks.

This is the correct boundary. The take-home should design and prototype everything needed up to that point, not spend compute on a low-quality training run.

