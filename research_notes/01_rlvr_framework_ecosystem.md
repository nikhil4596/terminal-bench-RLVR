# RLVR Framework Ecosystem For Terminal Agents

This document teaches the open-source ecosystem around RLVR-style post-training for language-model agents. It is written for this take-home assignment: design an RLVR environment for improving an LLM on Terminal-Bench 2, without actually running expensive RL training.

Primary sources to keep open while reading:

- TRL docs: https://huggingface.co/docs/trl
- TRL GRPO trainer: https://huggingface.co/docs/trl/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
- verl: https://github.com/verl-project/verl
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- SkyRL: https://github.com/NovaSky-AI/SkyRL
- Ray RLlib: https://docs.ray.io/en/latest/rllib/index.html
- Agent Lightning: https://arxiv.org/abs/2508.03680
- Terminal-Bench: https://arxiv.org/abs/2601.11868 and https://www.tbench.ai/

## Executive Summary

For this assignment, the best answer is not "use a classic RL library and make a Gym wrapper." That would satisfy the words of the prompt, but it would miss what modern LLM post-training systems actually need: high-throughput generation, token-level logprobs, reference-policy KL, reward functions over text/tool traces, and multi-turn trajectory logging.

The practical framework choice for a no-training prototype is:

**Use Harbor as the primary environment and rollout substrate.**

Reason:

- Harbor is the official Terminal-Bench execution harness.
- The assignment asks for an RLVR environment design, not an expensive training run.
- A correct container/task/reward/rollout substrate is the work that must exist before any trainer is useful.
- Harbor can run Terminal-Bench tasks and produce the traces a future trainer will need.

The scale-up answer is:

**Use SkyRL, TRL, verl, or OpenRLHF once the work becomes real model training.**

Reason:

- Terminal-Bench rollouts are long and expensive.
- Real training needs async environment execution, GPU/CPU separation, high-throughput serving, checkpoint recovery, and careful observability.
- SkyRL is especially relevant for Harbor/OpenEnv-style agent environments.
- TRL is the cleanest educational GRPO path for a small research pilot.
- OpenRLHF and verl are stronger when distributed training infrastructure matters.

The framework decision should be presented as a ladder:

```text
4-8 hour design + no-training prototype:
  Harbor + TerminalBenchEnv adapter + rewards + rollout schema

Single-machine research pilot:
  Harbor/OpenEnv rollouts + TRL GRPO or SkyRL, LoRA, small code model

Multi-GPU agentic RL:
  SkyRL or OpenRLHF, async multi-turn environments, vLLM/SGLang rollouts

Large lab-scale training:
  verl or OpenRLHF, Ray/Megatron/FSDP, strict evaluation pipeline
```

## The Pieces Of An RLVR System

An RLVR setup for coding agents has six separate pieces. Good framework selection means choosing where each piece lives.

```text
Task source
  -> environment runtime
  -> agent scaffold/action parser
  -> reward functions
  -> rollout storage
  -> policy optimizer
  -> evaluation harness
```

### Task Source

For this take-home, the task source is Terminal-Bench 2.0. A task contains an instruction, container image/environment, tests, reference solution, and time limit. The benchmark is outcome-driven: tests verify final container state, not the exact commands the agent ran.

Important implication: training should not contaminate the official evaluation task set. If we claim evaluation improvement on Terminal-Bench 2.0, we must either train on synthetic/dev tasks or use a clearly separated training split and disclose it.

### Environment Runtime

The runtime must launch isolated containers, expose a terminal/filesystem interface, execute commands/patches, collect stdout/stderr, enforce timeouts, and run tests. Terminal-Bench 2.0 is distributed through Harbor, so the natural implementation target is a Harbor-backed adapter.

For the take-home, we do not need to run all 89 tasks. We need to design the adapter and provide a smoke-testable skeleton.

### Agent Scaffold

The agent scaffold turns model text into environment actions. For Terminal-Bench, actions can be raw shell commands, file edits, or a finish signal.

A bad scaffold silently lets the model do arbitrary things and makes training data hard to parse. A good scaffold uses a structured action protocol:

```text
<bash>
pytest -q
</bash>

<patch path="src/foo.py">
...
</patch>

<finish>
ready for grading
</finish>
```

This still leaves the model enough freedom to solve the task, but makes the environment safer and more inspectable.

### Reward Functions

RLVR stands on executable or virtual reward signals. For Terminal-Bench, the cleanest reward is hidden-test success. That reward is sparse, so we add capped progress rewards and hard integrity penalties.

Minimum reward set for the assignment:

- Final success reward: hidden/verifier tests pass.
- Progress reward: public or sandboxed diagnostic tests improve, build succeeds, or failing-test count decreases.
- Integrity penalty: modifying tests, oracle files, benchmark metadata, or using forbidden shortcuts terminates the episode.
- Efficiency penalty: small step/token penalty to discourage infinite loops.

### Rollout Storage

For actual training, every step must be logged:

```json
{
  "task_id": "openssl-selfsigned-cert",
  "prompt_hash": "...",
  "step": 7,
  "action_kind": "bash",
  "action_text_hash": "...",
  "stdout_hash": "...",
  "reward_components": {
    "success": 0.0,
    "progress": 0.1,
    "integrity": 0.0,
    "step": -0.01
  },
  "done": false
}
```

Do not store official benchmark private tests or canary-sensitive content in public artifacts. Store hashes and metadata when possible.

### Policy Optimizer

This is where TRL, verl, OpenRLHF, SkyRL, RLlib, or Agent Lightning enters. A modern LLM RL optimizer needs access to:

- token logprobs from the active policy,
- token logprobs from a reference policy or old policy,
- rewards for generated completions or trajectories,
- grouping for GRPO-style relative advantages,
- batching and gradient updates.

For the take-home, define the optimizer interface and training plan. Do not run training.

### Evaluation Harness

The evaluation harness should run the trained or candidate agent against held-out tasks and report:

- pass rate,
- pass@k or pass^k if multiple stochastic attempts are used,
- cost and tokens per solved task,
- average steps to success,
- invalid action/tamper rate,
- reward-success correlation.

## Framework Comparison

### Harbor

Harbor is the official execution and evaluation harness for Terminal-Bench 2. It is not an RL trainer. It is the environment substrate: it launches tasks, manages sandboxes, runs agents, and records results. That makes it the right primary implementation target for a no-training take-home.

Why Harbor fits this assignment:

- It is the native way to run `terminal-bench@2.0`.
- It already understands Terminal-Bench task metadata, timeouts, resources, and sandbox backends.
- It can produce the rollouts that future RL optimization needs.
- It keeps the deliverable focused on environment design rather than GPU training infrastructure.

Weaknesses:

- Harbor does not update model weights.
- Future training still needs token/logprob capture and an optimizer such as TRL, SkyRL, OpenRLHF, or verl.
- Exact reproducibility requires pinning Harbor version, task version, and sandbox backend.

Best use cases:

- No-training prototype.
- Terminal-Bench evaluation and rollout collection.
- Smoke tests for reward functions and safety checks.

Use in our design:

```text
Harbor task runner -> TerminalBenchEnv adapter -> rewards -> rollout records
```

### OpenEnv

OpenEnv is an environment API layer for agentic RL environments. It can bridge environment execution to trainer libraries and has documented integrations with TRL and other stacks. It should be treated as useful but experimental.

Why OpenEnv matters:

- It gives a standard environment interface.
- It can make Terminal-Bench-like environments easier to plug into RL trainers.
- It aligns with multi-turn agent training concepts.

Weaknesses:

- It is not the canonical Terminal-Bench harness; Harbor is.
- Experimental APIs may change.
- Local/lightweight modes may not reproduce full Docker-based Terminal-Bench fidelity.

Best use cases:

- Optional bridge between Harbor-style environments and TRL/SkyRL training.
- Educational demos of environment-based GRPO.
- Future integration once the Harbor-first environment is stable.

Use in our design:

OpenEnv is optional. The no-training artifact should not depend on it, but the write-up can mention it as a bridge if the training stack expects an OpenEnv/Gym-like interface.

### Hugging Face TRL

TRL is a full-stack post-training library for transformer language models. Current docs list SFT, GRPO, DPO, Reward, RLOO, PPO, and other methods. TRL is also easy to explain: the model is a Hugging Face model, rewards are Python functions or reward models, and trainers implement known post-training algorithms.

Why TRL fits this assignment:

- It is popular and actively maintained.
- It supports GRPO, which maps well to verifiable rewards.
- It supports custom reward functions.
- It integrates with vLLM for faster generation.
- It is realistic to prototype in a small repo.

Weaknesses:

- Stateful multi-turn terminal environments are more complex than one-shot math/completion examples.
- You may need an adapter layer to turn Terminal-Bench trajectories into trainer-compatible prompts/completions.
- For large-scale asynchronous container rollouts, TRL is less naturally systems-oriented than SkyRL, verl, or OpenRLHF.

Best use cases:

- Research prototype.
- Small model LoRA training.
- Single-turn or lightly multi-turn verifiable tasks.
- Clear pedagogical write-up for a take-home.

Use in our future training design:

```text
TerminalBenchEnv -> rollout collector -> grouped trajectories -> TRL GRPOTrainer
```

The no-training prototype should not depend on TRL. The environment produces trajectory-level rewards and rollout records. Later, a training script can sample multiple trajectories for the same task, compute group-relative advantages, and train the policy on action-generating turns.

### verl

verl is a production-oriented RL training library for LLMs and the open-source version of the HybridFlow work. It is built around efficient RLHF/RLVR dataflows and integrates with FSDP, Megatron-LM, vLLM, SGLang, and Hugging Face models. It supports PPO, GRPO, RLOO, REINFORCE++, DAPO, PRIME, DrGRPO, and other recent variants.

Why verl matters:

- It is closer to how serious RLVR training is run at scale.
- It supports function-based rewards, which are central to verifiable tasks.
- It is designed around throughput and model placement.
- It has strong relevance to math/coding RL pipelines.

Weaknesses:

- It is heavier than TRL.
- It introduces distributed systems complexity that distracts from this 4-8 hour assignment.
- The agent-loop parts may require careful integration and are not as simple to demonstrate as a TRL reward function.

Best use cases:

- Multi-GPU or cluster-scale post-training.
- Large models, long contexts, high-throughput rollouts.
- Research labs building sustained RLVR infrastructure.

How to discuss it in the take-home:

Use verl as the scale-up path: "If this environment moved from prototype to real training, I would port the reward and rollout interfaces to verl for async generation, vLLM/SGLang serving, and distributed optimization."

### OpenRLHF

OpenRLHF is a Ray/vLLM/DeepSpeed-oriented framework for RLHF and RLVR. It supports PPO, GRPO, RLOO, REINFORCE++ variants, DPO/KTO-style methods, custom rewards, and agentic/multi-turn training modes.

Why it matters:

- Strong distributed training story.
- Good support for custom reward servers.
- Explicitly relevant to agentic multi-turn RL.
- More production-minded than a minimal TRL prototype.

Weaknesses:

- Heavy configuration surface.
- More infrastructure than needed for a take-home.
- Ray/vLLM/DeepSpeed debugging can dominate the project if used too early.

Best use cases:

- When rollouts and rewards can be served asynchronously.
- When GPU utilization and distributed scheduling matter.
- When a team already uses Ray.

How to discuss it:

OpenRLHF is a credible alternative. If the interviewer cares about scale, explain that the same `TerminalBenchEnv.step` and reward APIs can be exported behind an OpenRLHF custom reward/agent function.

### SkyRL

SkyRL is a newer agent-oriented RL framework from NovaSky. It is especially relevant because it focuses on multi-turn RL and includes agentic environments such as coding, search, SQL, and terminal-use integrations. It is closer to the shape of Terminal-Bench than classic RLHF libraries.

Why it matters:

- Agentic RL is a first-class concern.
- Multi-turn tasks and environment feedback are central.
- Its ecosystem points toward Harbor/terminal-use style tasks.

Weaknesses:

- Fast-moving APIs.
- Less universally recognized than TRL.
- For a take-home, reviewers may know TRL better.

Best use cases:

- Long-horizon agentic RL with real environment feedback.
- Terminal/SWE/search/SQL agents where actions and observations are multi-turn.

How to discuss it:

SkyRL is arguably the most semantically aligned future training stack for Terminal-Bench-style RLVR. For this no-training assignment, keep the runnable artifact Harbor-first. For a small training pilot, compare TRL and SkyRL: TRL is easier to explain and validate, while SkyRL is closer to long-horizon agentic RL.

### Ray RLlib

RLlib is a general-purpose distributed RL library. It supports algorithms such as PPO, APPO, IMPALA, DQN/Rainbow, SAC, DreamerV3, offline RL methods, custom environments, and multi-agent setups.

Why it matters:

- Strong classic RL infrastructure.
- Good for custom Gym-like environments.
- Multi-agent and external simulator support.

Weaknesses for LLM RLVR:

- Not LLM-native.
- Does not directly solve token logprob, reference KL, transformer model loading, LoRA, chat templates, or vLLM rollout serving.
- You would spend much of the project rebuilding LLM post-training machinery.

Best use cases:

- Robotics/control/game RL.
- Multi-agent simulation where the policy is not a giant language model.
- Baseline environment formalization if an interviewer asks for "classic RL."

How to discuss it:

RLlib is useful as a conceptual reference for MDP structure, but it is not the best primary choice for LLM post-training in 2026. Picking RLlib as the main answer would be defensible only if the assignment specifically prioritized classic RL engineering over LLM post-training realism.

### Agent Lightning

Agent Lightning proposes decoupling agent execution from RL training. It treats agent runs as MDP transitions and adds a credit-assignment layer, letting existing agents be trained without deeply rewriting them.

Why it matters:

- This is exactly the systems problem in coding agents: existing agents use tools, memory, retries, dynamic workflows, and nontrivial observations.
- It frames trajectories as training transitions rather than one giant concatenated sequence.
- It helps reason about credit assignment across steps.

Weaknesses:

- It is younger than TRL/verl/OpenRLHF.
- It is more of an agent optimization architecture than a standalone mature trainer.
- For this take-home, it is better as research support than the primary implementation choice.

Best use cases:

- Retrofitting RL onto existing agents.
- Multi-agent and workflow-heavy systems.
- Production agent observability plus training.

How to discuss it:

Use Agent Lightning as the conceptual architecture: decouple Terminal-Bench execution from optimizer code. Use TRL as the concrete trainer.

## Framework Decision Matrix

| Framework | LLM-native | GRPO/RLVR fit | Multi-turn agents | Scale | Take-home fit | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Harbor | N/A | High as env substrate | High | High | Very high | Choose for no-training prototype |
| OpenEnv | N/A | High as API bridge | High | Medium | Medium | Optional bridge, experimental |
| TRL | High | High | Medium | Medium | High for future pilot | Future trainer path |
| verl | High | High | Medium-High | Very high | Medium | Scale-up path |
| OpenRLHF | High | High | High | High | Medium | Scale-up path |
| SkyRL | High | High | High | High | Medium-High | Agentic RL alternative |
| RLlib | Low for LLMs | Medium | High for classic RL | High | Low-Medium | Conceptual baseline only |
| Agent Lightning | Medium | Medium-High | High | Medium | Medium | Architecture inspiration |

## Recommended Architecture

```text
                       +----------------------+
                       | Terminal-Bench task  |
                       | instruction + image  |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Harbor/container env |
                       | reset / step / test  |
                       +----------+-----------+
                                  |
                                  v
+----------------+      +----------------------+      +------------------+
| LLM policy     | ---> | action parser         | ---> | shell/patch exec |
| Qwen coder     |      | bash/patch/finish     |      | timeout/safety   |
+----------------+      +----------+-----------+      +---------+--------+
                                  |                            |
                                  v                            v
                       +----------------------+      +------------------+
                       | observation builder  | <--- | stdout/files     |
                       | prompt + state       |      | process status   |
                       +----------+-----------+      +------------------+
                                  |
                                  v
                       +----------------------+
                       | reward functions     |
                       | success/progress/etc |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | rollout dataset      |
                       | jsonl/parquet        |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | future trainer       |
                       | TRL/SkyRL/OpenRLHF   |
                       +----------------------+
```

The take-home repo should not train the policy. It should show that the interfaces are coherent:

- A mock environment can return observations and rewards.
- A scripted policy can solve a toy task.
- The reward functions are composable and inspectable.
- A future `train_grpo.py` can consume the rollout format.

## Case Studies To Understand The Ecosystem

### Case Study 1: Math RLVR With GRPO

Math reasoning is the cleanest RLVR domain because the reward is often an exact answer check. DeepSeekMath introduced GRPO as a PPO variant that samples multiple outputs per prompt and estimates relative advantages within the group instead of learning a value model.

Transfer to Terminal-Bench:

- Replace "math problem" with "terminal task."
- Replace "final answer exact match" with "hidden tests pass."
- Replace "single completion" with "multi-turn tool trajectory."
- Keep the idea of multiple rollouts per task and group-relative advantage.

The harder part is that Terminal-Bench rewards arrive after environment interaction, not after one text response.

### Case Study 2: Code Generation With Unit-Test Rewards

CodeRL and later execution-feedback work showed that unit tests can be used as rewards for code generation and repair. The model samples code, executes tests, and learns from pass/fail or partial execution signals.

Transfer to Terminal-Bench:

- Unit tests are still useful, but now they are part of a broader terminal workflow.
- The agent may need to inspect files, install packages, debug, write scripts, and run commands.
- Public tests should be treated as progress signals, not as the final objective.

The main risk is overfitting visible tests. The environment must protect hidden tests and disallow test tampering.

### Case Study 3: Software Engineering Agents

SWE-agent and long-context multi-turn SWE RL work show that interface design matters. The model's action language, file-edit affordances, and feedback loop can change performance as much as the model itself.

Transfer to Terminal-Bench:

- A raw shell-only interface is general but hard to learn.
- A structured action interface is easier to train but must not oversimplify the benchmark.
- Environment traces need to separate model decisions from tool outputs.

For this take-home, structured `bash`, `patch`, and `finish` actions are the right compromise.

### Case Study 4: Agent Execution Decoupled From Training

Agent Lightning argues that training should not be tightly coupled to the runtime agent. Instead, agent execution produces traces; a separate trainer consumes those traces.

Transfer to Terminal-Bench:

- Harbor/container execution should be isolated from GRPO code.
- Reward computation should be deterministic and replayable.
- The same environment should support scripted policies, local model policies, and future RL policies.

This is why the proposed repo should have separate modules for environment, actions, rewards, rollouts, and training config.

## What To Avoid

Do not claim we trained a model. The assignment does not require it, and running real RL would be expensive.

Do not claim benchmark improvement without a contamination-safe train/eval setup.

Do not make the reward just "all tests pass" and stop there. Sparse reward is correct but insufficient as a training design.

Do not choose RLlib as the main framework unless the argument is specifically about classic RL. It is not the most realistic LLM post-training stack.

Do not build an environment that lets the policy edit the tests, oracle solution, task metadata, or grader.

Do not expose hidden tests to the policy as observations.

## Recommended Take-Home Claim

The final submission should say:

> I choose Terminal-Bench 2.0 and Harbor for the no-training RLVR environment prototype. Terminal-Bench gives executable, outcome-based rewards in realistic terminal environments, and Harbor is the official substrate for running those tasks and collecting rollouts. The prototype defines a Harbor-backed environment adapter, structured action space, observation builder, reward functions, safety checks, rollout logging, and a future GRPO training configuration. I do not run expensive RL; the remaining work after submission is to execute the training plan on a non-contaminating task split and evaluate on held-out Terminal-Bench tasks.

That statement is accurate, scoped, and defensible.
