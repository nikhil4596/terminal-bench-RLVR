# Terminal-Bench 2 And Modeling Strategy Review

This document makes Terminal-Bench 2 concrete and connects it to modeling strategies for RLVR-style coding agents. It is deliberately different from the framework ecosystem doc: this file is about the benchmark, task structure, reward surface, modeling choices, and related research.

Primary sources:

- Terminal-Bench paper: https://arxiv.org/abs/2601.11868
- Terminal-Bench site: https://www.tbench.ai/
- Terminal-Bench 2 leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.0
- Harbor: https://github.com/harbor-framework/harbor
- Harbor Terminal-Bench docs: https://www.harborframework.com/docs/tutorials/running-terminal-bench
- tau2-bench paper: https://arxiv.org/abs/2506.07982
- tau2-bench repo: https://github.com/sierra-research/tau2-bench

## What Terminal-Bench 2 Is

Terminal-Bench is a benchmark for AI agents operating in terminal environments. Terminal-Bench 2.0 contains 89 hard, realistic tasks. Each task is built around:

- an English instruction,
- a containerized environment,
- tests/verifiers,
- an oracle or reference solution,
- a time limit.

The task is outcome-driven. The tests check whether the final container state satisfies the instruction; they do not require the agent to use a particular sequence of commands.

That design is important for RLVR:

- It gives us a verifiable reward.
- It allows many solution strategies.
- It forces the model to interact with a stateful environment.
- It makes shortcut/cheating prevention essential.

## Current Version Caveat

Use precise language:

- The assignment says `terminalbench2`.
- The paper describes **Terminal-Bench 2.0**.
- The public site also lists Terminal-Bench 2.1 and Terminal-Bench 3.0-in-development.
- Harbor and repo names have changed over time.

For this take-home, pin the target as:

```text
Target benchmark: terminal-bench@2.0
Execution substrate: Harbor
Training status: no RL training run in the take-home
Future training: contamination-safe synthetic/dev tasks, then held-out evaluation
```

Also distinguish historical and current performance:

- The paper reports frontier agents below 65% at publication time.
- Current leaderboard rows are live and have moved above that.
- Any performance number in the write-up should be dated or treated as time-sensitive.

The take-home should not claim a performance improvement unless a pinned evaluation was actually run.

## Task Anatomy

A Terminal-Bench 2 task is naturally decomposed as:

```text
task/
  instruction.md
  task metadata
  environment/
    Dockerfile or image config
  tests/
    verifier scripts
  solution/
    oracle/reference solution
```

Different source pages expose slightly different filenames, but the conceptual structure is stable: instruction, environment, tests, solution, timeout/resources.

The agent receives the instruction and an environment. The verifier/test artifacts are not supposed to be part of the policy observation in a way that leaks the answer.

## Example Task Families

The official site and paper show tasks across many domains. Examples include:

- Build a Linux kernel from source and run it in QEMU.
- Configure a git server so pushed content appears on a webserver.
- Crack or recover a secret from an encrypted archive.
- Generate a self-signed OpenSSL certificate with exact file outputs.
- Reshard a C4-like dataset and implement compression/decompression scripts.
- Train a fastText model under size and accuracy constraints.
- Rewrite a COBOL program in Python.
- Recover a corrupted SQLite database.
- Implement tensor parallelism in PyTorch.
- Design a nucleotide sequence for a fusion protein.
- Implement path tracing or graphics/data-processing workflows.

Why this matters:

- Tasks are not only software patching.
- The benchmark covers system administration, security, data processing, scientific computing, ML, and legacy systems.
- A successful agent must plan, inspect, edit, run commands, test, and recover from failures.

## Evaluation Mechanics

The canonical score is task success rate:

```text
score = tasks solved / tasks attempted
```

Runs are repeated for stochastic agents. Leaderboards often use multiple attempts, such as `k=5`, and report confidence intervals.

For RLVR design, define three levels of evaluation:

1. **Training reward**
   - May include public progress checks and shaping.
   - Must not expose hidden tests.

2. **Development evaluation**
   - Uses held-out synthetic/dev tasks.
   - Used for iteration and debugging.

3. **Final evaluation**
   - Uses pinned Terminal-Bench 2.0 tasks or a private compatible set.
   - Reports pass rate, cost, steps, and tamper rate.

## Why Terminal-Bench Over tau2-Bench

The assignment allows either Terminal-Bench 2 or tau2-bench. Choose Terminal-Bench 2 for this take-home.

### Terminal-Bench Strengths

- Directly terminal-native.
- Natural match for coding/SWE/research-compute agents.
- Docker/container state creates a rich environment.
- Tests provide verifiable rewards.
- Harbor is the official runtime harness.
- Long-horizon tasks stress planning and tool use.

### tau2-Bench Strengths

- Strong for conversational tool-use agents.
- Models dual-control user-agent coordination.
- Verifiable final state and communication checks.
- Has user simulation and customer-service domains.

### Why tau2 Is Not The Main Choice Here

tau2 adds a simulated user and Dec-POMDP-style dual control. That is valuable research, but it shifts the assignment from terminal autonomy to dialogue coordination. It also introduces simulator stochasticity and version drift because the repo now includes tau3-style voice/knowledge extensions.

For a post-training/RLVR role focused on coding agents and terminal environments, Terminal-Bench is cleaner:

```text
terminal state + shell/file actions + verifier tests
```

That maps directly to RLVR.

## RLVR Mapping For Terminal-Bench

### Observation

Observation should be a bounded text view:

- task instruction,
- current working directory,
- recent command history,
- latest stdout/stderr,
- selected file contents or snippets,
- visible public test output,
- remaining budget,
- safety notices.

Do not include:

- hidden tests,
- oracle solution,
- benchmark answer keys,
- grader internals,
- canary-sensitive content.

### Action

Use structured actions:

```text
ActionKind = bash | patch | finish
```

Examples:

```text
<bash>
pytest -q
</bash>
```

```text
<patch path="/app/src/main.py">
...
</patch>
```

```text
<finish>
The requested artifact has been created and verified.
</finish>
```

Why structured actions:

- easier to parse,
- easier to safety-filter,
- easier to log,
- easier to train on action tokens only,
- still expressive enough for terminal tasks.

### Transition

The transition is the container state change after running the action. For example:

- files are edited,
- commands create artifacts,
- processes start or exit,
- packages are installed,
- logs are produced,
- tests pass or fail.

### Termination

Episode ends when:

- hidden verifier succeeds,
- model emits `finish` and verifier runs,
- max steps reached,
- timeout reached,
- safety violation occurs,
- environment crashes irrecoverably.

### Reward

Use a composite reward:

```text
R_total =
  1.00 * R_success
+ 0.20 * capped(R_progress)
- 1.00 * R_integrity_violation
- 0.01 * steps
- 0.00001 * generated_tokens
```

The exact weights are tuning defaults, not laws. The principle is:

- final success dominates,
- progress helps but cannot win alone,
- integrity violations terminate,
- efficiency is a small nudge.

## Contamination And Anti-Cheating

Terminal-Bench is public. Therefore:

- Do not train on official evaluation tasks and claim unbiased improvement.
- Do not allow internet access during evaluation unless the benchmark configuration explicitly allows it.
- Pin Harbor version and task revision.
- Hash task files and test files.
- Run hidden tests outside the writable workspace when possible.
- Copy tests into the container after agent actions when possible.
- Fail fast if tests, oracle, task metadata, or grader files are modified.
- Log task ids and prompt hashes for reproducibility.

The final take-home should explicitly say:

> I use Terminal-Bench 2.0 as the target benchmark and task format. For actual model training, I would use decontaminated synthetic or held-out Terminal-Bench-style tasks, then evaluate on a pinned held-out set. This submission stops before training and benchmark claims.

## Modeling Strategy

### Strategy 1: Harbor-First RLVR Substrate

Because the deliverable is no-training, the primary implementation should be Harbor-first:

```text
Harbor task execution
  -> observations
  -> structured actions
  -> reward extraction
  -> rollout records
  -> future trainer handoff
```

This is more accurate than making TRL the runnable dependency. TRL is for future policy updates; Harbor is for the actual benchmark environment.

### Strategy 2: Scripted Smoke Tests Before Any Model

Before training or even model inference, validate with scripted policies:

- a policy that intentionally fails,
- a policy that solves a toy task,
- a policy that tries to modify tests,
- a policy that times out.

Expected results:

- solve path gets success reward,
- fail path gets zero success,
- tamper path terminates with integrity penalty,
- timeout path terminates cleanly.

### Strategy 3: Baseline Agent Before RL

The first model baseline should be non-RL:

- Qwen2.5-Coder-7B-Instruct or similar,
- fixed action format,
- temperature 0.2-0.7,
- no weight updates,
- run on toy/synthetic tasks only for local testing.

The goal is to test environment and logging, not train.

### Strategy 4: Rejection Fine-Tuning Before RL

If training were allowed later:

1. Collect successful traces from oracle/scripted or model rollouts.
2. Filter traces for integrity and generality.
3. Fine-tune the model on successful action traces.
4. Use that model as the RL starting point.

This is more compute-efficient than pure RL from a weak base policy.

### Strategy 5: GRPO For Future RL

For future RL:

- sample multiple attempts per task,
- compute rewards per trajectory,
- normalize within task group,
- update action-token probabilities,
- monitor KL and invalid actions.

GRPO is attractive because it avoids a learned value function, which is hard in long-horizon terminal tasks.

### Strategy 6: Separate Scaffold Improvements From Model Improvements

Agent scaffolds can change performance independently of model weights. Any evaluation must compare:

```text
same scaffold + base model
same scaffold + trained model
```

If the scaffold changes, the comparison is confounded.

## Nine Deep Paper Reviews For This Task

### 1. Terminal-Bench: Benchmarking Agents On Hard, Realistic Tasks In Command Line Interfaces

Why it matters:

Terminal-Bench is the target benchmark. Its core contribution is not merely "some shell tasks"; it formalizes a realistic terminal-agent evaluation protocol: containerized task, natural-language instruction, tests over final state, oracle solution, and time/resource constraints.

Key contributions:

- Defines terminal-native tasks as a benchmark substrate for autonomous agents.
- Uses final-state tests rather than command-output matching.
- Covers diverse domains: SWE, systems, security, data science, ML, scientific computing.
- Includes human verification and anti-cheating review.
- Evaluates both model and scaffold combinations, showing the scaffold matters.

How it shapes our design:

- The reward should be final verifier success.
- The action space should allow general terminal work, not narrow function synthesis.
- The environment must prevent test/oracle tampering.
- Evaluation should report task pass rate, steps, cost, and failure modes.

Open issues:

- Public benchmark contamination.
- External dependencies and environment drift.
- Long-horizon expensive rollouts.
- Leaderboard numbers change over time.

### 2. DeepSeekMath And GRPO

Why it matters:

DeepSeekMath introduced Group Relative Policy Optimization, a key RLVR method. It showed that verifiable rewards and relative advantage estimation can improve reasoning without a separate value model.

Key contributions:

- Introduces GRPO as a PPO variant.
- Samples groups of completions for the same prompt.
- Computes relative rewards within each group.
- Reduces memory by avoiding a value model.
- Demonstrates gains on math reasoning.

How it shapes our design:

- For each Terminal-Bench task, sample multiple attempts.
- Compare attempts within the same task to reduce task-difficulty variance.
- Avoid value-model complexity in a take-home design.

Open issues:

- Terminal tasks are multi-turn and longer than math completions.
- If all attempts fail, the group signal is weak.
- Reward shaping quality matters more.

### 3. DeepSeek-R1

Why it matters:

DeepSeek-R1 popularized large-scale RL on verifiable tasks, including math, coding competitions, and STEM. It supports the idea that rule-based rewards can induce stronger reasoning behaviors.

Key contributions:

- Demonstrates RL can produce self-reflection and verification behaviors in verifiable domains.
- Shows cold-start and staged training can improve readability and stability.
- Uses verifiable rewards rather than dense human annotations.
- Distills reasoning into smaller models.

How it shapes our design:

- Use executable rewards wherever possible.
- Do not rely on LLM judges for correctness.
- Consider SFT/RFT before RL for readability and action-format compliance.
- Track behavior quality, not only reward.

Open issues:

- Scale is much larger than this take-home.
- Results on math/competition coding do not automatically transfer to open-ended terminal work.

### 4. CodeRL

Why it matters:

CodeRL is an early bridge between code generation and reinforcement learning from unit tests. It treats functional correctness as a reward signal.

Key contributions:

- Uses execution results and unit tests as feedback.
- Applies actor-critic methods to code generation.
- Shows code LMs can improve from verifier-based rewards.
- Highlights functional correctness over text similarity.

How it shapes our design:

- Unit and integration tests are natural rewards.
- Public tests can help with progress shaping.
- Final hidden tests should remain the main objective.

Open issues:

- Unit-test suites can be incomplete.
- One-shot code generation is simpler than terminal-agent tasks.
- Test overfitting is a constant risk.

### 5. RLTF: Reinforcement Learning From Unit Test Feedback

Why it matters:

RLTF focuses on multi-granularity unit-test feedback. It supports the idea that tests can produce richer rewards than binary pass/fail.

Key contributions:

- Uses unit tests as online reward signals.
- Explores feedback at different granularities.
- Strengthens the case for execution-grounded code RL.

How it shapes our design:

- Progress reward can use public test deltas.
- Reward logs should separate test categories.
- Hidden tests remain separate from training reward.

Open issues:

- More granular test feedback can be gamed.
- Terminal tasks include non-code artifacts, services, and system state.

### 6. RLEF: Reinforcement Learning From Execution Feedback

Why it matters:

RLEF moves closer to agentic debugging: the model uses execution feedback across iterative steps. This is more relevant to Terminal-Bench than one-shot code synthesis.

Key contributions:

- Models execution feedback as a training signal.
- Encourages iterative repair.
- Shows sample-efficiency gains from feedback-grounded learning.
- Connects code generation to multi-step environment interaction.

How it shapes our design:

- Include stdout/stderr and test failures in observations.
- Train only on model action tokens, not environment outputs.
- Reward the final fix, but preserve intermediate feedback for credit analysis.

Open issues:

- Feedback can be misleading.
- Long terminal trajectories create context and credit-assignment pressure.

### 7. SWE-agent

Why it matters:

SWE-agent shows that agent-computer interface design materially affects software-engineering performance. This is critical: the environment is not neutral.

Key contributions:

- Introduces agent-computer interfaces for SWE tasks.
- Shows command/edit affordances can improve benchmark performance.
- Highlights repository navigation and file editing as first-class actions.
- Makes scaffold design part of the modeling problem.

How it shapes our design:

- Use structured `bash`, `patch`, `finish` actions.
- Keep file edits auditable.
- Log all observations and commands.
- Do not evaluate model quality without fixing the scaffold.

Open issues:

- Scaffold engineering can mask policy weaknesses.
- Different models may prefer different action formats.

### 8. Agent Lightning

Why it matters:

Agent Lightning proposes separating agent execution from RL training and decomposing trajectories into transitions. That maps naturally to Terminal-Bench.

Key contributions:

- Treats agent execution as an MDP.
- Decouples runtime from training.
- Introduces a credit-assignment module for complex trajectories.
- Supports existing agent frameworks with minimal instrumentation.

How it shapes our design:

- Harbor execution should produce generic rollout records.
- Trainer-specific logic should be downstream.
- The no-training prototype should focus on the rollout substrate.

Open issues:

- "Any agent" claims should be treated cautiously.
- It does not solve reward design or contamination.

### 9. Long-Context, Multi-Turn Software Engineering Agents With RL

Why it matters:

This is one of the closest references for future work after the take-home. It addresses stateful SWE environments, long contexts, delayed rewards, and tool feedback.

Key contributions:

- Applies RL to multi-turn SWE tasks, not only single-turn prompts.
- Uses a DAPO-style approach and rejection fine-tuning.
- Reports large gains on SWE-bench Verified under a fixed scaffold.
- Discusses data quality, sparse rewards, expensive evaluation, and long contexts.

How it shapes our design:

- Start with RFT/SFT before RL.
- Keep the scaffold fixed for model comparisons.
- Track pass@1/pass@k, steps, and submit behavior.
- Treat rollouts as expensive resources.

Open issues:

- Needs independent reproduction.
- SWE-bench repository repair differs from broad Terminal-Bench tasks.

### 10. ReTool

Why it matters:

ReTool trains models to use tools strategically during reasoning. Terminal-Bench agents must learn when to inspect, run tests, edit, and verify.

Key contributions:

- Interleaves natural-language reasoning with code execution.
- Uses RL to teach when and how to invoke tools.
- Shows tool-use behaviors can improve under outcome rewards.

How it shapes our design:

- Tool-use policy is part of the learning objective.
- Reward should not only favor final answers, but trajectories that use tools effectively.
- Observations must include tool results in a clean format.

Open issues:

- Tool use in math/code interpreter settings is simpler than arbitrary terminal control.
- Tool misuse can increase cost without improving success.

### 11. PRIME And Process Rewards

Why it matters:

PRIME tackles implicit process rewards, a central issue in long-horizon tasks where final reward is sparse.

Key contributions:

- Attempts process-level credit without dense human labels.
- Uses rollout/outcome information to infer useful intermediate signals.
- Addresses sparse-reward RLVR.

How it shapes our design:

- Keep intermediate reward logs.
- Preserve trajectories for future process-reward experiments.
- Do not overcommit to hand-crafted shaping.

Open issues:

- Process rewards can encode spurious correlations.
- Terminal tasks have many irrelevant-but-benign actions.

### 12. ReST, STaR, Reflexion, And Self-Refine

Why they matter:

These methods create improvement loops using feedback without necessarily running online policy-gradient RL.

Key contributions:

- ReST: generate, score, filter, fine-tune.
- STaR: bootstrap reasoning traces from successful outputs.
- Reflexion: store verbal feedback in memory.
- Self-Refine: generate critique and revise.

How they shape our design:

- Use successful traces as future SFT/RFT data.
- Let agents use execution feedback to revise.
- Keep final selection verifier-gated.

Open issues:

- Self-generated feedback can be wrong.
- Verbal reflection is not a substitute for gradient RL.
- Filtering can collapse diversity.

## Recommended Final Strategy

The best take-home strategy is:

1. Choose Terminal-Bench 2.0.
2. Use Harbor as the environment/evaluation substrate.
3. Define a structured RLVR environment wrapper.
4. Implement no training in the deliverable.
5. Provide reward functions and rollout schema.
6. Define a future GRPO training path through TRL or SkyRL.
7. Defend contamination-safe train/eval splits.
8. Include a synthetic task-generation stretch path.

This is stronger than pretending to train. It shows research judgment: the bottleneck is building a correct environment and reward substrate before spending GPU time.

