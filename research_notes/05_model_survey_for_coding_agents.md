# Model Survey For Coding Agents And RLVR Post-Training

This note surveys the current model landscape for coding agents, with emphasis
on models that matter for Terminal-Bench-style RLVR, SWE agents, tool use, and
post-training research. It is not part of the official take-home submission.
Its job is to help defend model choices in an interview.

The take-home uses `Qwen2.5-Coder-7B-Instruct` as the pilot model. That is a
pragmatic training choice, not a claim that it is the strongest available
coding agent. The real landscape spans small local coders, medium open SWE
agents, large open MoE models, and closed frontier systems.

## Executive Summary

The coding-agent model landscape has split into four tiers:

```text
small local code models
  -> cheap experimentation, SFT, toy RL, action grammar tests

medium open coding/SWE models
  -> practical RLVR pilots and reproducible research

large open MoE and agentic models
  -> strong baselines, expensive training, good distillation teachers

closed frontier models
  -> strongest agent performance, teacher traces, evaluation references,
     but not directly trainable by us
```

For this project:

- Best pilot model: `Qwen2.5-Coder-7B-Instruct`.
- Best stronger open baseline: `Qwen2.5-Coder-32B-Instruct` or
  `Devstral-Small-2507`.
- Best open agentic code frontier reference: `Qwen3-Coder-480B-A35B-Instruct`.
- Best open RL-trained SWE-agent reference: `DeepSWE-Preview`.
- Best closed frontier references: GPT-5.1-Codex / GPT-5.1-Codex-Max,
  Claude Sonnet 4.5, and Gemini 3 Pro.

The key interview point:

> Model selection for RLVR is not only about the leaderboard. It is about
> trainability, context length, license, scaffold compatibility, inference
> cost, rollout throughput, action-format reliability, and whether the model is
> a policy, a teacher, a critic, or an evaluation baseline.

## The Main Axes For Model Choice

### 1. Trainability

A model can be great for inference but impractical for RL:

- closed API models cannot be weight-updated by us;
- very large MoEs may be too expensive to train;
- custom architectures may be hard to serve in vLLM/SGLang;
- licenses may restrict commercial or redistribution use;
- models with unusual chat templates may need careful action-format tuning.

For RLVR, the model must produce many samples per prompt. GRPO usually wants
multiple completions for the same state. That makes rollout cost central.

### 2. Code Prior

Terminal agents need more than general reasoning. They need:

- shell fluency,
- repository navigation,
- test interpretation,
- multi-file editing,
- build/debug loops,
- patch discipline,
- patience across long horizons.

General frontier models often do this well because they have broad post-training
and agent tuning. Open models usually need either code pretraining or SWE-agent
trajectory tuning.

### 3. Agent Scaffold Compatibility

A model's measured performance depends heavily on the scaffold:

- SWE-agent,
- OpenHands,
- R2E-Gym,
- Codex CLI,
- Claude Code,
- Gemini CLI / Antigravity,
- custom Terminal-Bench harnesses.

The same model can look weak or strong depending on tool schema, prompt,
context packing, file editor, search tool, max steps, and verifier strategy.
Interview defense should always mention this.

### 4. Context Length

Coding agents need long context for:

- task instruction,
- repository structure,
- file contents,
- command history,
- stdout/stderr,
- previous patches,
- traceback logs,
- test output.

Long context matters, but it is not sufficient. A 128K model with poor tool
discipline can lose to a shorter-context model with better agent training.

### 5. Reward Robustness

Some models are especially prone to:

- verbose action outputs that fail parsing,
- hidden reasoning that does not map to executable actions,
- overuse of shell commands,
- test tampering,
- finishing too early,
- getting stuck in inspect-only loops.

The best RLVR base policy is not necessarily the smartest model. It is the
model whose behavior can be improved reliably by rewards.

## Spectrum Of Models

### Small Models: 0.5B To 7B

Use these for:

- action grammar experiments,
- toy Terminal-Bench tasks,
- SFT over successful traces,
- cheap rollout debugging,
- LoRA or QLoRA pilot training,
- classroom-scale RL.

Examples:

- `Qwen2.5-Coder-0.5B/1.5B/3B`
- `Qwen2.5-Coder-7B-Instruct`
- `StarCoder2-3B/7B`
- `CodeGemma-2B/7B`
- small general instruct models like Llama or Phi variants

Main limitation: they often lack robust long-horizon planning. They may know
syntax but fail at repository-level debugging.

### Medium Models: 14B To 34B

Use these for:

- serious open RLVR pilots,
- SWE-agent baselines,
- post-training research,
- local or small-cluster inference,
- tool-use SFT/RL.

Examples:

- `Qwen2.5-Coder-14B-Instruct`
- `Qwen2.5-Coder-32B-Instruct`
- `Qwen3-32B`
- `DeepSWE-Preview`
- `Skywork-SWE-32B`
- `SWE-agent-LM-32B`
- `Devstral-Small-2507`

This is the most important range for reproducible coding-agent research. It is
large enough to solve real SWE tasks and small enough that labs can train or
serve it with manageable infrastructure.

### Large Open MoE Models

Use these for:

- strong open baselines,
- teacher traces,
- distillation,
- large-scale inference,
- evaluation reference models,
- potentially large RL if the lab has serious compute.

Examples:

- `Qwen3-Coder-480B-A35B-Instruct`
- `Kimi-K2-Instruct`
- `DeepSeek-Coder-V2-Instruct`
- `DeepSeek-V3`
- `gpt-oss-120b`

These models can be excellent agents, but direct RL training is expensive.
They are more realistic as teachers, rollouts generators, or frontier open
baselines than as first-pass trainable policies.

### Closed Frontier Models

Use these for:

- teacher traces,
- evaluation reference,
- scaffolding benchmarks,
- synthetic task generation,
- judge/critic roles,
- qualitative comparison.

Examples:

- GPT-5.1 and GPT-5.1-Codex family,
- Claude Sonnet 4.5,
- Gemini 3 Pro.

These are not weight-trainable by us, but they define the frontier behavior we
want open RLVR systems to approach.

## Recommended Roles For This Project

| Role | Recommended Model | Reason |
| --- | --- | --- |
| Pilot trainable policy | Qwen2.5-Coder-7B-Instruct | Good code prior, feasible GRPO/LoRA, easy serving. |
| Stronger open baseline | Qwen2.5-Coder-32B-Instruct | Same family, stronger code ability, still understandable. |
| Open SWE-agent baseline | Devstral-Small-2507 | Agentic SWE-focused, strong OpenHands result. |
| RL-trained SWE reference | DeepSWE-Preview | Shows what RL on executable SWE environments can do. |
| Large open frontier reference | Qwen3-Coder-480B-A35B-Instruct | Agentic coding, long context, strong open-model reference. |
| Open general agent reference | Kimi-K2-Instruct | Huge MoE with tool-use and agentic training emphasis. |
| Closed frontier reference | GPT-5.1-Codex / Claude Sonnet 4.5 / Gemini 3 Pro | Best available coding-agent behavior, but not trainable. |
| Small debugging model | Qwen2.5-Coder-1.5B/3B | Cheap tests for action grammar and reward plumbing. |

## Deep Profiles

### 1. Qwen2.5-Coder Series

Primary source:

- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

Why it matters:

Qwen2.5-Coder is one of the most practical open code model families for
research. The series spans 0.5B, 1.5B, 3B, 7B, 14B, and 32B sizes, which makes
it unusually useful for scaling studies. The 7B instruct model is a strong
pilot candidate because it is code-specialized, instruction-tuned, and feasible
to serve and fine-tune.

Key details from the model card:

- 7B instruct checkpoint has roughly 7.6B parameters.
- The series is trained on a large mixture of source code, text-code grounding,
  synthetic data, and general tokens.
- It supports long-context usage, with 128K context described for the family.
- The model card reports very high Hugging Face usage, making it a popular
  practical baseline.

Contributions:

- It gives a clean small-to-large model ladder.
- It has enough code prior that RL can focus on agent behavior rather than
  basic syntax.
- It is common enough that tooling, quantizations, adapters, and examples are
  easy to find.
- It is Apache 2.0, which simplifies research and commercial discussion.

Why we chose 7B:

- It is feasible for a take-home training plan.
- It can be adapted with LoRA/QLoRA.
- It should understand shell/code enough to make Terminal-Bench rollouts useful.
- It is a better pilot than a 32B model if compute is limited.

Weaknesses:

- It is not the strongest coding agent.
- It may need action-format SFT before stable RL.
- Smaller variants may fail long-horizon tasks badly.
- It is not specifically trained on Terminal-Bench or SWE-agent trajectories.

Interview defense:

> Qwen2.5-Coder-7B is the policy I would train first, not the model I expect to
> win the benchmark. It is the right pilot because the bottleneck in this
> assignment is environment and reward design, not demonstrating frontier model
> inference.

### 2. Qwen3-Coder-480B-A35B-Instruct

Primary source:

- https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct

Why it matters:

Qwen3-Coder-480B-A35B-Instruct is a frontier open agentic coding model. The
model card describes it as Qwen's most agentic code model, with 480B total
parameters and 35B activated parameters. It has native 256K context and can be
extended further with YaRN-style context extension.

Key details:

- MoE architecture.
- 480B total parameters, 35B active.
- Native 256K context.
- Apache 2.0.
- Designed for agentic coding and tool-call style workflows.
- The model card reports Terminal-Bench 2 and SWE-bench Pro evaluation rows.

Contributions:

- Shows the open frontier is moving from code completion to agentic coding.
- Makes long-context repository-scale work central.
- Provides a strong open reference for Terminal-Bench-like work.
- Uses tool-call formatting as a first-class capability.

Why not choose it as the pilot:

- It is far too expensive for a take-home training plan.
- It is a better teacher, baseline, or evaluation reference than a first
  trainable policy.
- GRPO on this model would require serious distributed infrastructure.

How to use it in this project:

- As a high-end open baseline.
- As a teacher for synthetic rollouts.
- As a model to compare after the Harbor adapter works.
- As evidence that agentic coding is now a core model-design objective.

Interview defense:

> Qwen3-Coder is the open frontier reference; Qwen2.5-Coder-7B is the practical
> trainable policy. Those roles are different.

### 3. gpt-oss-20b And gpt-oss-120b

Primary sources:

- https://openai.com/index/gpt-oss-model-card/
- https://openai.com/index/introducing-gpt-oss

Why they matter:

gpt-oss models are open-weight reasoning models designed for agentic workflows,
tool use, and adjustable reasoning effort. They matter because they bring
frontier-style post-training ideas into an open-weight format.

Key details from OpenAI's release:

- `gpt-oss-20b`: 21B total parameters, 3.6B active per token, 128K context.
- `gpt-oss-120b`: 117B total parameters, 5.1B active per token, 128K context.
- Apache 2.0.
- Text-only.
- Designed for reasoning, tool use, and agentic workflows.
- Post-trained with supervised fine-tuning and high-compute RL-style stages.

Contributions:

- Strong example of open-weight models tuned for reasoning and tools.
- Adjustable reasoning effort is directly relevant to agent budgets.
- 20B is interesting for local or lab-scale experimentation.
- 120B is a strong teacher/reference model if hardware is available.

Risks and caveats:

- These are general reasoning models, not code-only models.
- Terminal-agent reliability still depends on scaffold and action grammar.
- Safety posture differs when open weights can be modified.

How to use them:

- `gpt-oss-20b` as a medium open reasoning-agent baseline.
- `gpt-oss-120b` as a teacher or high-end open baseline.
- Compare against code-specialized models to see whether general reasoning or
  code pretraining matters more for Terminal-Bench.

Interview defense:

> gpt-oss is important because it makes reasoning and tool-use post-training
> visible in open weights, but a code-specialized model may still be the better
> initial RLVR policy for Terminal-Bench.

### 4. DeepSeek-Coder-V2 And DeepSeek-V3/R1 Lineage

Primary sources:

- https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct
- https://huggingface.co/deepseek-ai/DeepSeek-V3

Why it matters:

DeepSeek is central to the modern open model landscape because it pushed MoE
architectures, code specialization, and RL reasoning into widely used open
models. DeepSeek-Coder-V2 in particular is a major code model family, while
DeepSeek-R1 influenced the community's understanding of RLVR-style reasoning.

Key details:

- DeepSeek-Coder-V2 has Lite and full variants.
- Lite is 16B total / 2.4B active.
- Full is 236B total / 21B active.
- Context length is 128K.
- It supports many programming languages and emphasizes code plus math.
- DeepSeek-V3 is a very large MoE foundation/chat model.

Contributions:

- Shows the value of code continued pretraining on top of a strong base.
- Demonstrates sparse MoE as a way to scale capability without dense inference
  cost.
- Provides strong code/math baselines for open research.
- DeepSeek-R1 made rule-based/verifiable reward training a mainstream topic.

Weaknesses:

- Some checkpoints require `trust_remote_code` or custom serving care.
- Licenses are not always as straightforward as Apache 2.0.
- Full models are expensive to serve and train.

How to use them:

- DeepSeek-Coder-V2-Lite as a medium code baseline.
- DeepSeek-Coder-V2 full as a large open code baseline.
- DeepSeek-R1/GRPO lineage as conceptual support for RLVR.

Interview defense:

> DeepSeek is part of the intellectual foundation for this project, especially
> because it connects code reasoning, MoE scale, and verifiable-reward RL.

### 5. Kimi-K2-Instruct

Primary source:

- https://huggingface.co/moonshotai/Kimi-K2-Instruct

Why it matters:

Kimi K2 is a very large open MoE model explicitly optimized for agentic
capabilities, tool use, reasoning, and coding. It is important because it sits
between open research models and proprietary frontier systems.

Key details from the model card:

- 1T total parameters.
- 32B activated parameters.
- 128K context.
- Modified MIT license.
- Trained with Muon optimizer.
- Designed for tool use and autonomous problem solving.
- Model card reports strong SWE-bench, TerminalBench, Tau2, and tool-use
  results.

Contributions:

- Strong example of open agentic post-training at very large scale.
- Treats tool calling as a core capability rather than a wrapper.
- Demonstrates that open models can compete on agentic coding evaluations.
- Provides useful benchmark comparisons against Claude, GPT, DeepSeek, and
  Qwen-style models.

Weaknesses:

- Very expensive to train directly.
- Modified license needs review.
- Operationally more complex than Qwen2.5-Coder or Devstral.

How to use it:

- Teacher model.
- High-end baseline.
- Reference for tool-use prompt design.
- Comparison point for whether code-specialized versus general agentic
  post-training wins on Terminal-Bench.

Interview defense:

> Kimi K2 is a serious open agent model, but for this take-home it is a
> reference model, not a practical first policy to train.

### 6. Devstral-Small-2507

Primary source:

- https://huggingface.co/mistralai/Devstral-Small-2507

Why it matters:

Devstral is an agentic software-engineering model from Mistral AI and All Hands
AI. It is designed for using tools, exploring codebases, editing multiple files,
and working in OpenHands-style SWE workflows.

Key details:

- 24B parameters.
- Apache 2.0.
- 128K context.
- Finetuned from Mistral-Small-3.1.
- Intended to work well with OpenHands.
- Model card reports 53.6% on SWE-bench Verified with the OpenHands scaffold.

Contributions:

- Strong example of a medium-sized model specialized for SWE agents.
- Shows how much scaffold-model co-design matters.
- Practical enough to run compared with huge MoEs.
- Good open baseline for real software-engineering tasks.

Weaknesses:

- Its best results are scaffold-specific.
- It is optimized around OpenHands, not necessarily Terminal-Bench/Harbor.
- It may need action-protocol adaptation for our `<bash>/<patch>/<finish>`
  schema.

How to use it:

- Strong medium open baseline.
- Candidate model after the Qwen2.5-Coder pilot.
- Useful comparison for agent-specialized tuning versus general code pretraining.

Interview defense:

> Devstral is exactly the kind of model I would benchmark after the environment
> works. It may outperform a generic coder in SWE workflows, but its scaffold
> coupling must be measured.

### 7. Skywork-SWE-32B

Primary source:

- https://huggingface.co/Skywork/Skywork-SWE-32B

Why it matters:

Skywork-SWE is a 32B software-engineering agent model based on
Qwen2.5-Coder-32B-Instruct. It is valuable because it studies data scaling laws
for SWE capabilities and trains on large numbers of executable trajectories.

Key details:

- 32B/33B class model.
- Apache 2.0.
- Based on Qwen2.5-Coder-32B-Instruct.
- Reports 38.0% pass@1 on SWE-bench Verified.
- Reports improved performance with test-time scaling.
- Uses OpenHands in the evaluation setup.

Contributions:

- Shows that trajectory data scale matters for SWE agents.
- Provides a clear example of turning a strong base coder into an agent model.
- Connects data curation, executable environments, and agent performance.
- Useful empirical support for our rollout-collection emphasis.

Weaknesses:

- Performance depends on OpenHands and evaluation setup.
- It is a result/model to learn from, not necessarily the easiest first policy
  to train.
- Monthly downloads are relatively small compared with Qwen2.5-Coder, so
  ecosystem maturity may be lower.

How to use it:

- Medium/large open SWE-agent baseline.
- Evidence for scaling trajectory data.
- Comparison point against our future Terminal-Bench rollouts.

Interview defense:

> Skywork-SWE supports the claim that environment traces and data scaling are
> central to software-engineering agents, not just raw model size.

### 8. SWE-agent-LM-32B And SWE-smith

Primary source:

- https://huggingface.co/SWE-bench/SWE-agent-LM-32B

Why it matters:

SWE-agent-LM-32B is a model trained with the SWE-smith toolkit. The model card
states that it fine-tunes Qwen2.5-Coder-Instruct on thousands of trajectories
generated by SWE-agent plus a stronger teacher model.

Key details:

- 32B/33B class model.
- Apache 2.0.
- Based on Qwen2.5-Coder-32B-Instruct.
- Trained on SWE-agent trajectories.
- Designed to be compatible with SWE-agent.

Contributions:

- Direct evidence that agent trajectory data can specialize code models.
- Strong example of distilling frontier-agent behavior into an open model.
- Shows the importance of scaffold-specific formatting and action traces.
- Gives a concrete recipe: collect trajectories, filter, fine-tune.

Weaknesses:

- Scaffold-specific.
- SFT trajectory imitation may reproduce teacher habits and errors.
- It may not generalize automatically to Terminal-Bench action formats.

How to use it:

- Baseline for trajectory-imitation methods.
- Comparison to RL-trained models like DeepSWE.
- Inspiration for an offline SFT warm start before GRPO.

Interview defense:

> SWE-agent-LM is important because it shows that agent traces are a real data
> asset. For our project, Harbor traces would be the analogous asset.

### 9. DeepSWE-Preview And R2E-Gym

Primary source:

- https://huggingface.co/agentica-org/DeepSWE-Preview
- https://github.com/R2E-Gym/R2E-Gym

Why it matters:

DeepSWE-Preview is one of the most relevant models for this take-home because
it is explicitly an RL-trained SWE agent. Its model card states that it is
trained on top of Qwen3-32B using only RL on executable SWE environments.

Key details:

- 32B class model based on Qwen3-32B.
- MIT license.
- Trained with RL using rLLM.
- Uses R2E-Gym environments and tools.
- Model card reports large SWE-bench Verified gains after around 200 RL steps.
- Uses tool set: bash execution, search, file editor, finish/submit.

Contributions:

- Very close to our target idea: RL over executable software tasks.
- Demonstrates that sparse outcome rewards can work for SWE agents.
- Highlights modern GRPO variants: DAPO, Dr.GRPO, RLOO/LOOP-style changes,
  length normalization, compact filtering, and entropy choices.
- Shows the value of test-time scaling with execution-based and execution-free
  verifiers.

Weaknesses:

- Trained/evaluated in R2E-Gym, not Terminal-Bench.
- Uses its own scaffold and tools.
- Reported results should be interpreted with scaffold and verifier details in
  mind.

How to use it:

- The closest "what this project could become" reference.
- Model/paper to study before any interview on coding-agent RL.
- Strong support for our Harbor-first environment design.

Interview defense:

> DeepSWE is the clearest evidence that RLVR-style training can improve
> software-engineering agents, but it also proves why environment fidelity,
> tool schema, verifier design, and train/eval contamination control matter.

### 10. StarCoder2, CodeGemma, And Code Llama

Primary sources:

- https://huggingface.co/bigcode/starcoder2-15b
- https://huggingface.co/google/codegemma-7b
- https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf

Why they matter:

These models are less likely to be the best current Terminal-Bench policies,
but they define the evolution of open code models.

StarCoder2:

- 3B, 7B, and 15B family.
- Trained on The Stack v2.
- Strong transparency focus around code data.
- Useful for code completion and lower-resource language coverage.

CodeGemma:

- 2B and 7B code models built on Gemma.
- Includes code completion, generation, and instruction-tuned variants.
- Useful lightweight baseline for code generation and editor-style tasks.

Code Llama:

- Historically important family from Meta.
- 7B, 13B, 34B, and 70B variants.
- Code completion, infilling, Python specialist, and instruct variants.

Contributions:

- Established open code model pretraining as a major category.
- Popularized FIM/infilling, repository-oriented data, and multilingual code
  coverage.
- Provide historical baselines for how far agentic models have moved beyond
  one-shot code generation.

Weaknesses:

- Many are weaker than modern Qwen/DeepSeek/Devstral-style models on agentic
  SWE tasks.
- Older chat templates and shorter context windows can be limiting.
- They are better baselines than primary choices for this take-home.

Interview defense:

> These are important historical baselines. The field moved from code
> completion toward repository-level agent behavior, which is why Terminal-Bench
> RLVR needs environment design rather than just HumanEval-style generation.

### 11. Closed Frontier Coding-Agent Models

Primary sources:

- https://platform.openai.com/docs/models/gpt-5.1/
- https://platform.openai.com/docs/models/gpt-5.1-codex
- https://openai.com/index/gpt-5-1-codex-max/
- https://docs.claude.com/en/docs/about-claude/models/whats-new-claude-4-5
- https://ai.google.dev/models/gemini
- https://blog.google/products/gemini/gemini-3/

Why they matter:

Closed frontier models define the practical ceiling for coding-agent behavior.
They are often the models used in tools like Codex, Claude Code, Cursor,
Gemini CLI, Antigravity, and other agentic developer products.

GPT-5.1 / GPT-5.1-Codex:

- OpenAI docs describe GPT-5.1 as the flagship model for coding and agentic
  tasks.
- GPT-5.1-Codex is optimized for agentic coding in Codex-style environments.
- GPT-5.1-Codex-Max emphasizes long-running coding work and compaction across
  context windows.

Claude Sonnet 4.5:

- Anthropic docs describe it as their best model for complex agents and coding.
- It is strongly associated with Claude Code and long-horizon software tasks.

Gemini 3 Pro:

- Google describes Gemini 3 Pro as its most powerful agentic and vibe-coding
  model.
- Google reports strong Terminal-Bench 2.0 and SWE-bench Verified numbers in
  its Gemini 3 announcement.

Contributions:

- They set expectations for long-running coding agents.
- They provide high-quality teacher traces.
- They can generate synthetic tasks, critiques, and rubrics.
- They are useful for comparing open models under a fixed scaffold.

Weaknesses:

- We cannot directly train their weights.
- Benchmark numbers can depend heavily on private scaffolds.
- Cost and rate limits matter for large rollout generation.
- They may have hidden system-level behavior that is hard to reproduce.

How to use them:

- Teacher traces for SFT.
- Critic/judge models for failure review.
- Upper-bound references.
- Synthetic task generation and rubric generation.

Interview defense:

> Closed frontier models are not candidates for our trainable policy, but they
> are essential references for what good agent behavior looks like.

## Model Ranking And Popularity Caveats

Do not overinterpret raw leaderboards.

Reasons:

- SWE-bench scores depend on scaffold, max iterations, tool design, and
  patch-selection strategy.
- Terminal-Bench results depend on harness version, task version, time limits,
  and whether the model gets a terminal-oriented scaffold.
- Multi-attempt and best-of-N numbers are not comparable to pass@1.
- Proprietary model results can include private agent infrastructure.
- Hugging Face downloads measure usage, not quality.

Useful signals:

- Official model cards and technical reports.
- SWE-bench Verified under a fixed scaffold.
- Terminal-Bench 2.0 under Harbor.
- Long-context tool-use benchmarks.
- Whether the model has real users, quantizations, adapters, and serving
  support.

For interview answers, say:

> I would compare models under the same scaffold and environment, not by mixing
> leaderboard rows collected with different agents.

## What To Learn From The Research Landscape

### Lesson 1: Code Completion Is Not Coding Agency

HumanEval-style code generation is not enough. Terminal agents need:

- inspect,
- search,
- edit,
- run,
- debug,
- retry,
- finish.

This is why models like Devstral, DeepSWE, Skywork-SWE, SWE-agent-LM, and
Qwen3-Coder matter more than older pure completion models.

### Lesson 2: Data Is Becoming Executable

Modern coding-agent data is no longer just files and docstrings. It is:

- issue descriptions,
- repository snapshots,
- shell traces,
- failed tests,
- patches,
- final verifier outcomes,
- Docker environments.

Skywork-SWE, SWE-smith, and R2E-Gym all point in this direction.

### Lesson 3: RL Is Returning Because Rewards Are Verifiable

RLHF for chat needed subjective preference models. Coding agents can use:

- test pass/fail,
- build success,
- verifier success,
- environment outcome,
- tamper detection.

That makes RLVR much more natural for coding than for open-ended chat.

### Lesson 4: Scaffold And Model Co-Evolve

The action format is not neutral. A model trained for OpenHands may not be
optimal in SWE-agent. A model trained with R2E-Gym tools may need adaptation for
Terminal-Bench/Harbor.

Our repo's action grammar is therefore a real modeling decision, not just a
parser detail.

### Lesson 5: Test-Time Scaling Is A Major Lever

DeepSWE, Skywork-SWE, Kimi K2, and frontier closed models all show that multiple
attempts, critics, verifiers, and selection strategies can greatly change
reported performance.

For RL training, this matters because:

- train-time sampling gives GRPO its group comparisons;
- test-time sampling can improve benchmark performance without weight updates;
- evaluation must separate policy quality from sampling/selection tricks.

## How This Changes The Take-Home Model Story

The submission currently says:

```text
Pilot model: Qwen2.5-Coder-7B-Instruct
Follow-ups: larger Qwen, DeepSeek-Coder, CodeLlama, general instruct models
```

That is directionally correct but underspecified. A stronger spoken defense is:

```text
I chose Qwen2.5-Coder-7B-Instruct because this take-home is about the
environment and reward substrate. For a first trainable policy, I want a model
that is open, code-specialized, instruction-tuned, long-context enough, and
cheap enough for GRPO exploration.

If I were benchmarking today, I would also compare against Devstral,
Qwen2.5-Coder-32B, DeepSWE-Preview, Skywork-SWE, and SWE-agent-LM. If I wanted
a frontier open baseline or teacher, I would use Qwen3-Coder-480B-A35B, Kimi K2,
DeepSeek-Coder-V2, or gpt-oss. If I wanted a closed frontier reference, I would
use GPT-5.1-Codex, Claude Sonnet 4.5, or Gemini 3 Pro.
```

## Recommended Interview Talking Points

### Why Not Use The Biggest Open Model?

Because the assignment is not leaderboard chasing. The goal is an RLVR
environment that can later train. A 480B MoE is great for evaluation, but bad
for a pilot training plan.

### Why Not Use A Closed Frontier Model?

Closed models are excellent teachers and references, but we cannot update their
weights. RLVR post-training requires control over the policy.

### Why Not Use A Tiny Model?

Tiny models are useful for plumbing and toy RL, but Terminal-Bench tasks require
repository-level reasoning and long-horizon debugging. A 7B code model is a
reasonable minimum serious pilot.

### Why Not Use A SWE-Specialized Model Like Devstral Immediately?

It is a strong candidate, but it may be scaffold-coupled. For a clean research
take-home, starting with Qwen2.5-Coder-7B keeps the base policy simple and
trainable. Devstral is an excellent baseline after the Harbor adapter works.

### What Would You Evaluate?

Use a fixed scaffold and compare:

- Qwen2.5-Coder-7B-Instruct,
- Qwen2.5-Coder-32B-Instruct,
- Devstral-Small-2507,
- DeepSWE-Preview,
- Skywork-SWE-32B,
- SWE-agent-LM-32B,
- one large open MoE,
- one closed frontier reference.

Track:

- pass rate,
- invalid action rate,
- timeout rate,
- average steps,
- cost per solved task,
- token budget,
- safety violations,
- success by task family.

## Short Model Cheat Sheet

| Model | Size | Open? | Best role | Main caveat |
| --- | --- | --- | --- | --- |
| Qwen2.5-Coder-1.5B/3B | small dense | yes | cheap experiments | weak long-horizon agent. |
| Qwen2.5-Coder-7B-Instruct | 7B dense | yes | pilot RL policy | not frontier. |
| Qwen2.5-Coder-32B-Instruct | 32B dense | yes | stronger open baseline | heavier training. |
| Qwen3-Coder-480B-A35B | 480B/35B MoE | yes | frontier open baseline | expensive. |
| gpt-oss-20b | 21B/3.6B MoE | yes | reasoning/tool baseline | not code-specific. |
| gpt-oss-120b | 117B/5.1B MoE | yes | teacher/reference | large serving footprint. |
| DeepSeek-Coder-V2-Lite | 16B/2.4B MoE | yes | medium code baseline | custom serving/license review. |
| DeepSeek-Coder-V2 | 236B/21B MoE | yes | large code baseline | expensive. |
| Kimi-K2-Instruct | 1T/32B MoE | yes | open agentic frontier | very large, modified license. |
| Devstral-Small-2507 | 24B dense | yes | SWE-agent baseline | OpenHands coupling. |
| Skywork-SWE-32B | 32B dense | yes | trajectory-scaling reference | lower ecosystem usage. |
| SWE-agent-LM-32B | 32B dense | yes | trace imitation reference | SWE-agent coupling. |
| DeepSWE-Preview | 32B dense | yes | RL-trained SWE reference | R2E-Gym coupling. |
| StarCoder2-15B | 15B dense | yes | historical code baseline | less agentic. |
| CodeGemma | 2B/7B dense | yes | lightweight baseline | less terminal-agent focused. |
| Code Llama | 7B-70B dense | yes-ish | historical baseline | older, license/context caveats. |
| GPT-5.1-Codex | undisclosed | no | frontier reference | not trainable. |
| Claude Sonnet 4.5 | undisclosed | no | frontier reference | not trainable. |
| Gemini 3 Pro | undisclosed | no | frontier reference | not trainable. |

## Suggested Reading Order

1. Qwen2.5-Coder model card.
2. Qwen3-Coder model card.
3. DeepSWE-Preview model card.
4. R2E-Gym paper/project.
5. Skywork-SWE model card/technical report.
6. SWE-agent-LM / SWE-smith.
7. Devstral model card.
8. Kimi K2 model card.
9. gpt-oss model card.
10. GPT-5.1-Codex, Claude Sonnet 4.5, and Gemini 3 Pro docs.

## Sources

- Qwen2.5-Coder-7B-Instruct:
  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Qwen2.5-Coder-1.5B:
  https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B
- Qwen3-Coder-480B-A35B-Instruct:
  https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct
- gpt-oss model card:
  https://openai.com/index/gpt-oss-model-card/
- Introducing gpt-oss:
  https://openai.com/index/introducing-gpt-oss
- DeepSeek-Coder-V2-Lite-Instruct:
  https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- DeepSeek-Coder-V2-Instruct:
  https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct
- DeepSeek-V3:
  https://huggingface.co/deepseek-ai/DeepSeek-V3
- Kimi-K2-Instruct:
  https://huggingface.co/moonshotai/Kimi-K2-Instruct
- Devstral-Small-2507:
  https://huggingface.co/mistralai/Devstral-Small-2507
- Skywork-SWE-32B:
  https://huggingface.co/Skywork/Skywork-SWE-32B
- SWE-agent-LM-32B:
  https://huggingface.co/SWE-bench/SWE-agent-LM-32B
- DeepSWE-Preview:
  https://huggingface.co/agentica-org/DeepSWE-Preview
- R2E-Gym:
  https://github.com/R2E-Gym/R2E-Gym
- StarCoder2-15B:
  https://huggingface.co/bigcode/starcoder2-15b
- CodeGemma-7B:
  https://huggingface.co/google/codegemma-7b
- Code Llama 34B Instruct:
  https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf
- GPT-5.1 model docs:
  https://platform.openai.com/docs/models/gpt-5.1/
- GPT-5.1-Codex model docs:
  https://platform.openai.com/docs/models/gpt-5.1-codex
- GPT-5.1-Codex-Max announcement:
  https://openai.com/index/gpt-5-1-codex-max/
- Claude 4.5 docs:
  https://docs.claude.com/en/docs/about-claude/models/whats-new-claude-4-5
- Gemini model docs:
  https://ai.google.dev/models/gemini
- Gemini 3 announcement:
  https://blog.google/products/gemini/gemini-3/
