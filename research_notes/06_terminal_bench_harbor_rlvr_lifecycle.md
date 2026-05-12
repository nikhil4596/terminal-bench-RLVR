# Terminal-Bench, Harbor, And RLVR Data Lifecycle

This note fills the conceptual gap in the earlier research notes: what a
Terminal-Bench task actually is, what Harbor does, what our RLVR wrapper does,
what a "prompt" means in a stateful terminal task, and what gets handed to a
trainer.

The short version:

```text
Terminal-Bench = task suite
Harbor = runtime/evaluation harness for those task environments
tb_rlvr = RLVR contract around the runtime: observation, action, reward,
          safety, rollout records, and trainer handoff
TRL/verl = optimizers that update model weights after rollouts exist
```

Using Harbor does not mean we are no longer creating an RLVR environment. It
means we are using the correct benchmark runtime instead of reimplementing
Docker task execution ourselves. The RLVR environment is the interface around
that runtime:

```text
state observation -> policy action -> Harbor execution -> reward -> next state
```

## What The Assignment Actually Asked

The AfterQuery PDF asks us to:

- select a popular open-source RL framework,
- choose Terminal-Bench 2 or tau2-bench,
- design/create an RLVR environment for improving model performance,
- define observation space,
- define action space,
- define at least two virtual reward functions,
- choose a base model,
- choose and justify an RL algorithm,
- specify dataset size/configuration, curriculum, hyperparameters, and metrics,
- submit a detailed write-up plus a private code repo link.

It does not ask us to run expensive RL training.

But it also does not ask for only an essay. The strongest deliverable is:

```text
real benchmark runtime choice
  + clear environment contract
  + code implementing the contract
  + realistic training handoff
  + honest execution boundary
```

For Terminal-Bench 2, Harbor is the real runtime choice.

## What Terminal-Bench Is

Terminal-Bench is not a CSV of one-shot prompts.

It is a benchmark suite of task directories. Terminal-Bench 2.0 currently lists
89 tasks in the Harbor registry. Each task is closer to a mini software/project
environment than to a single prompt.

A task directory typically contains:

```text
task-id/
  instruction.md
  task.toml
  environment/
    Dockerfile
  solution/
    solve.sh
  tests/
    test.sh
    test_outputs.py
```

The task instruction is the initial user-facing problem statement. The Docker
environment contains the filesystem and dependencies. The tests/verifier define
success. The solution is an oracle/reference implementation used to validate the
task, not something the policy should see or train on.

Concrete task examples from Terminal-Bench 2.0 include:

- `openssl-selfsigned-cert`
- `fix-code-vulnerability`
- `sqlite-db-truncate`
- `train-fasttext`
- `nginx-request-logging`
- `build-cython-ext`
- `log-summary-date-ranges`

## Is It Just 89 Prompts?

No.

There are 89 public benchmark tasks in the version we inspected. Each task has
one initial instruction, but a rollout creates many prompts because the
environment changes after every action.

For one task:

```text
initial task instruction
  -> observation prompt at step 1
  -> action 1
  -> stdout/stderr/filesystem changes
  -> observation prompt at step 2
  -> action 2
  -> stdout/stderr/filesystem changes
  -> ...
  -> final verifier
```

If a task takes 12 steps, it produces 12 decision prompts for that trajectory.
If we sample 8 trajectories for the same task, it can produce around 96
step-level prompts, plus 8 trajectory-level rewards.

For training scale:

```text
num_task_envs * rollouts_per_task * steps_per_rollout
```

Example:

```text
89 tasks * 8 rollouts/task * 20 steps/rollout = 14,240 step records
```

That is still small for serious post-training. So for real RLVR, Terminal-Bench
public tasks are mainly:

- evaluation environments,
- smoke-test environments,
- seed examples for task design,
- held-out benchmark tasks.

For training at scale, we would need:

- synthetic Terminal-Bench-style tasks,
- private/internal tasks,
- generated variants,
- task-family splits,
- decontaminated held-out eval tasks.

Do not train directly on the official eval set and then claim benchmark
improvement.

## What Harbor Does

Harbor is the benchmark runtime/harness. For Terminal-Bench, Harbor can:

- load `terminal-bench@2.0`,
- select a task,
- build or pull its Docker environment,
- run an agent inside/against that environment,
- run the verifier,
- write logs and reward files,
- run the oracle/reference solution for task validation.

Harbor is not the policy optimizer. It does not update model weights. It is
also not "the whole RL environment" by itself, because an RL environment needs a
specific observation/action/reward interface for a policy and trainer.

Think of Harbor as:

```text
Docker/task runtime + benchmark evaluator
```

Think of our wrapper as:

```text
RL state/action/reward/trajectory contract around that runtime
```

## What The Oracle Agent Means

The Harbor oracle agent is not an LLM. It is the task reference solution. It is
used to check that the task is solvable and that the verifier can produce a
passing reward.

Laptop-feasible command shape:

```bash
uvx harbor run -d terminal-bench@2.0 -t openssl-selfsigned-cert -a oracle
```

This needs Docker and Harbor, not a GPU and not an API model.

The oracle path is useful for:

- verifying Harbor can run locally,
- verifying the Docker task environment works,
- verifying the task's tests can pass,
- confirming where reward/log artifacts appear.

It is not a training rollout from our policy.

## What Our Code Does If Harbor Exists

Our code is still useful because Harbor by itself does not define our training
contract.

Our package defines:

- `Observation`: what the policy sees.
- `AgentAction`: what the policy is allowed to emit.
- `parse_action`: how text becomes an executable action.
- `check_action_safety`: what gets blocked before execution.
- `RewardComponents`: how final/progress/safety/efficiency rewards are logged.
- `RolloutRecord`: what gets saved for audit/training handoff.
- trainer export helpers: how records become prompt/action/reward samples.

Without this layer, we only have an eval harness. With this layer, we have an
RLVR environment interface.

## Where TRL Fits

TRL is not the environment.

TRL is the optimizer/trainer. In GRPO, it needs:

- prompts,
- sampled completions from the current policy,
- token logprobs for those completions,
- reward for each completion or trajectory,
- grouping of multiple samples per prompt,
- reference-policy/KL handling or equivalent regularization.

For terminal agents, the hard part is connecting:

```text
sampled completion/action -> Harbor execution -> verifier/progress reward
```

The JSONL rollout records are not the whole GRPO loop. They are:

- audit logs,
- replay data,
- debugging data,
- possible SFT/DPO data,
- an intermediate artifact for trainer integration.

Online GRPO must sample from the current model during training so it has fresh
logprobs. A completed JSONL file alone does not contain enough information for
policy-gradient training unless it also stores the exact policy/logprobs used
at sampling time.

## Two Ways To Train From Terminal Agents

### Option A: Step-Level RL

Each decision step is one sample:

```text
prompt = current observation
completion = one action
reward = immediate/progress reward, plus maybe terminal reward if episode ends
```

Pros:

- easier to fit into text-generation trainers,
- clean logprob accounting for one action,
- many samples per trajectory.

Cons:

- credit assignment is weak,
- a good early action may not get final success credit,
- progress rewards can distort behavior.

### Option B: Trajectory-Level RL

One sampled completion represents a whole action trajectory:

```text
prompt = initial task observation
completion = serialized sequence of actions
reward = final verifier reward plus penalties
```

Pros:

- reward matches benchmark outcome,
- easier to compare full solutions,
- less risk of overvaluing local progress.

Cons:

- long completions,
- difficult environment interleaving,
- harder to implement with ordinary chat trainers,
- expensive because every candidate trajectory must run in Harbor.

For this project, the practical design is:

```text
log step-level records for audit
score final trajectory success strongly
use progress rewards cautiously
plan GRPO around grouped candidate trajectories or grouped next actions
```

## Example 1: `openssl-selfsigned-cert`

### Raw Task Shape

Files:

```text
openssl-selfsigned-cert/
  instruction.md
  task.toml
  environment/Dockerfile
  solution/solve.sh
  tests/test.sh
  tests/test_outputs.py
```

Metadata summary from `task.toml`:

```text
difficulty: medium
category: security
agent timeout: 900 seconds
verifier timeout: 900 seconds
docker image: alexgshaw/openssl-selfsigned-cert:20251031
gpus: 0
```

Paraphrased instruction:

```text
Create a self-signed TLS certificate setup under /app/ssl. Generate a 2048-bit
RSA key, create a certificate for dev-internal.company.local under organization
DevOps Team, create a combined PEM file, write certificate verification details,
and create /app/check_cert.py that verifies and prints certificate information.
```

Verifier checks include:

- `/app/ssl` exists,
- private key exists and has restrictive permissions,
- key is 2048-bit RSA,
- certificate contains expected common name and organization,
- certificate validity is exactly 365 days,
- PEM contains key and certificate,
- verification text contains subject, dates, and SHA-256 fingerprint,
- Python verification script runs and prints success.

### Harbor Oracle Run

```bash
uvx harbor run -d terminal-bench@2.0 -t openssl-selfsigned-cert -a oracle
```

Lifecycle:

```text
Harbor loads task
  -> starts Docker environment
  -> oracle runs solution/solve.sh
  -> tests/test.sh runs pytest over tests/test_outputs.py
  -> verifier writes reward.txt as 1 or 0
```

No LLM is involved.

### RLVR Policy Rollout

Initial observation prompt, simplified:

```text
Task: openssl-selfsigned-cert

Instruction:
Create a self-signed TLS certificate setup under /app/ssl...

CWD: /app

Directory summary:
(initial task files and directories)

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

Action 1:

```xml
<bash>mkdir -p /app/ssl && openssl genrsa -out /app/ssl/server.key 2048 && chmod 600 /app/ssl/server.key</bash>
```

Environment effect:

```text
/app/ssl/server.key appears
stdout/stderr from openssl captured
```

Step reward:

```json
{"success":0.0,"progress":0.10,"integrity":0.0,"step":-0.01,"token":-0.0002,"total":0.0898}
```

Later action:

```xml
<finish>ready for grading</finish>
```

Final verifier reward:

```json
{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.06,"token":-0.0001,"total":0.9399}
```

Final rollout record, compact:

```json
{"task_id":"openssl-selfsigned-cert","episode_id":"openssl-selfsigned-cert:run-001","backend":"harbor","step":6,"observation_prompt":"Task: openssl-selfsigned-cert\n\nInstruction:\nCreate a self-signed TLS certificate setup under /app/ssl...\n\nCWD: /app\n\nDirectory summary:\n/app/ssl/server.key\n/app/ssl/server.crt\n/app/ssl/server.pem\n/app/ssl/verification.txt\n/app/check_cert.py\n\nRecent actions:\n...\n\nLast stdout:\n...\n\nLast stderr:\n\nSteps remaining: 24","model_output":"<finish>ready for grading</finish>","action":{"kind":"finish","path":null,"payload":"ready for grading"},"reward":{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.06,"token":-0.0001,"total":0.9399},"done":true,"terminal_reason":"success","observation_hash":"hash_before_final","next_observation_hash":"hash_after_final","info":{"dataset":"terminal-bench@2.0","verifier_ran":true,"verifier_reward_raw":"1","safety_ok":true}}
```

## Example 2: `fix-code-vulnerability`

### Raw Task Shape

Paraphrased instruction:

```text
Inspect a Bottle web framework repository in /app. Identify a CWE-class
vulnerability in /app/bottle.py, write /app/report.jsonl with the vulnerable
file and exact CWE id, modify the vulnerable code, and make pytest pass.
```

This task is more like realistic code repair:

- read repo,
- understand CWE categories,
- locate vulnerable logic,
- write structured vulnerability report,
- patch source code,
- run tests.

### Initial Observation

```text
Task: fix-code-vulnerability

Instruction:
Inspect the Bottle repository in /app, identify and fix a CWE vulnerability in
/app/bottle.py, write /app/report.jsonl, and make tests pass.

CWD: /app

Directory summary:
bottle.py
test/
setup.py
...

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

### Step 1: Inspect

```xml
<bash>grep -n "except\\|raise\\|HTTPError\\|ValueError" /app/bottle.py | head -80</bash>
```

Environment returns stdout lines. The next prompt includes that stdout.

Possible reward:

```json
{"success":0.0,"progress":0.02,"integrity":0.0,"step":-0.01,"token":-0.0001,"total":0.0099}
```

Small progress because inspection happened, but it should not dominate.

### Step 2: Patch

```xml
<patch path="/app/bottle.py">...modified code...</patch>
```

Safety check:

```text
allowed: /app/bottle.py
blocked: /app/tests/... or /tests/...
```

Possible progress reward:

```json
{"success":0.0,"progress":0.10,"integrity":0.0,"step":-0.02,"token":-0.0020,"total":0.078}
```

### Step 3: Write Report

```xml
<bash>printf '%s\n' '{"file_path":"/app/bottle.py","cwe_id":["cwe-20"]}' > /app/report.jsonl</bash>
```

### Step 4: Test

```xml
<bash>pytest -rA</bash>
```

If tests pass, this is progress. It is not necessarily final benchmark success
until the hidden verifier runs.

### Final JSONL Record

```json
{"task_id":"fix-code-vulnerability","episode_id":"fix-code-vulnerability:run-004","backend":"harbor","step":8,"observation_prompt":"Task: fix-code-vulnerability\n\nInstruction:\nInspect the Bottle repository...\n\nRecent actions:\nbash: grep -n ...\npatch: /app/bottle.py\nbash: printf ... > /app/report.jsonl\nbash: pytest -rA\n\nLast stdout:\n... tests passed ...\n\nLast stderr:\n\nSteps remaining: 22","model_output":"<finish>ready for grading</finish>","action":{"kind":"finish","path":null,"payload":"ready for grading"},"reward":{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.08,"token":-0.0001,"total":0.9199},"done":true,"terminal_reason":"success","observation_hash":"hash_before_final","next_observation_hash":"hash_after_final","info":{"dataset":"terminal-bench@2.0","verifier_ran":true,"verifier_reward_raw":"1","safety_ok":true}}
```

## Example 3: `sqlite-db-truncate`

### Raw Task Shape

Paraphrased instruction:

```text
A SQLite database at /app/trunc.db was corrupted by binary truncation. Recover
as many rows as possible and write them to /app/recover.json in a list of
objects with word/value fields.
```

This is a forensic/data recovery task. It may require:

- inspecting file size and magic bytes,
- trying `sqlite3`,
- reading raw pages,
- writing a recovery script,
- validating JSON format.

### Initial Observation

```text
Task: sqlite-db-truncate

Instruction:
Recover rows from /app/trunc.db and write /app/recover.json as a JSON list of
{"word": ..., "value": ...} records.

CWD: /app

Directory summary:
trunc.db

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

### Step 1: Inspect File

```xml
<bash>file /app/trunc.db && ls -lh /app/trunc.db && sqlite3 /app/trunc.db '.tables'</bash>
```

The command might partially fail because the database is corrupt. That failure
is useful observation, not terminal failure.

### Step 2: Write Recovery Script

```xml
<patch path="/app/recover_sqlite.py">...script that scans pages and extracts rows...</patch>
```

### Step 3: Run Recovery

```xml
<bash>python /app/recover_sqlite.py > /app/recover.json && python -m json.tool /app/recover.json | head</bash>
```

Progress reward can reflect:

- `/app/recover.json` exists,
- JSON parses,
- list has plausible records,
- word/value keys exist.

But final verifier decides if enough rows were recovered.

### Final Record

```json
{"task_id":"sqlite-db-truncate","episode_id":"sqlite-db-truncate:run-002","backend":"harbor","step":10,"observation_prompt":"Task: sqlite-db-truncate\n\nInstruction:\nRecover rows from /app/trunc.db...\n\nDirectory summary:\n/app/trunc.db\n/app/recover_sqlite.py\n/app/recover.json\n\nRecent actions:\n...\n\nLast stdout:\nJSON valid, 184 rows recovered\n\nLast stderr:\n\nSteps remaining: 20","model_output":"<finish>ready for grading</finish>","action":{"kind":"finish","path":null,"payload":"ready for grading"},"reward":{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.10,"token":-0.0001,"total":0.8999},"done":true,"terminal_reason":"success","observation_hash":"hash_before_final","next_observation_hash":"hash_after_final","info":{"dataset":"terminal-bench@2.0","verifier_ran":true,"verifier_reward_raw":"1","safety_ok":true}}
```

## Example 4: `train-fasttext`

### Raw Task Shape

Paraphrased instruction:

```text
Train a fastText model on Yelp data in /app/data. Save /app/model.bin. The final
model must be under 150MB and reach at least 0.62 accuracy on a private test set
from the same distribution.
```

This is a machine-learning systems task, not a software patch task.

### Initial Observation

```text
Task: train-fasttext

Instruction:
Train a fastText model on Yelp data in /app/data. Save /app/model.bin under
150MB with at least 0.62 private-test accuracy.

CWD: /app

Directory summary:
data/

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

### Step 1: Inspect Data

```xml
<bash>find /app/data -maxdepth 2 -type f -print && head -3 /app/data/* 2>/dev/null</bash>
```

### Step 2: Train Candidate

```xml
<bash>python - <<'PY'
import fasttext
model = fasttext.train_supervised('/app/data/train.txt', epoch=20, lr=0.5, wordNgrams=2, dim=100)
model.save_model('/app/model.bin')
PY</bash>
```

### Step 3: Check Size And Public Accuracy If Available

```xml
<bash>ls -lh /app/model.bin && python /app/eval.py /app/model.bin || true</bash>
```

Progress reward can reflect:

- model file exists,
- size under 150MB,
- public/dev accuracy above threshold if a dev set exists.

Final verifier uses private test distribution.

### Final Record

```json
{"task_id":"train-fasttext","episode_id":"train-fasttext:run-003","backend":"harbor","step":5,"observation_prompt":"Task: train-fasttext\n\nInstruction:\nTrain a fastText model...\n\nDirectory summary:\n/app/data\n/app/model.bin\n\nRecent actions:\n...\n\nLast stdout:\nmodel.bin 83M, dev accuracy 0.64\n\nLast stderr:\n\nSteps remaining: 25","model_output":"<finish>ready for grading</finish>","action":{"kind":"finish","path":null,"payload":"ready for grading"},"reward":{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.05,"token":-0.0001,"total":0.9499},"done":true,"terminal_reason":"success","observation_hash":"hash_before_final","next_observation_hash":"hash_after_final","info":{"dataset":"terminal-bench@2.0","verifier_ran":true,"verifier_reward_raw":"1","safety_ok":true}}
```

## Example 5: `nginx-request-logging`

### Raw Task Shape

Paraphrased instruction:

```text
Install/configure Nginx to listen on port 8080, serve /var/www/html, write
custom access/error logs, configure rate limiting, create index and 404 pages,
test config, and verify localhost:8080 works.
```

This is a systems administration task.

### Initial Observation

```text
Task: nginx-request-logging

Instruction:
Set up Nginx on port 8080 with custom request logging, error logging, rate
limiting, static files, custom 404 page, and working localhost access.

CWD: /app

Directory summary:
(base system files)

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

### Candidate Actions

Install:

```xml
<bash>apt-get update && apt-get install -y nginx</bash>
```

Configure:

```xml
<patch path="/etc/nginx/conf.d/benchmark-site.conf">server { listen 8080; ... }</patch>
```

Validate:

```xml
<bash>nginx -t && service nginx restart && curl -i localhost:8080</bash>
```

Final:

```xml
<finish>ready for grading</finish>
```

Reward shaping:

- progress if config file exists,
- progress if `nginx -t` passes,
- progress if `curl localhost:8080` returns expected content,
- final success only if verifier checks all required log/rate-limit behavior.

## The Exact Handoff Problem

There are three different "handoffs" that are easy to confuse.

### Handoff 1: Harbor To Our RLVR Wrapper

Input:

```text
task id + task instruction + container state + command outputs
```

Our wrapper produces:

```text
Observation object
```

Example:

```json
{"task_id":"nginx-request-logging","instruction":"Set up Nginx...","cwd":"/app","directory_summary":"...","recent_history":[],"last_stdout":"","last_stderr":"","selected_files":{},"steps_remaining":30}
```

### Handoff 2: Policy To Harbor

Input:

```text
model text
```

Our parser produces:

```json
{"kind":"bash","path":null,"payload":"nginx -t && service nginx restart"}
```

Harbor/runtime executes the action in the task environment.

### Handoff 3: Rollout To Trainer

For audit/SFT/DPO:

```json
{"prompt":"Task: ...","completion":"<bash>...</bash>","reward":0.089,"metadata":{"task_id":"...","step":1}}
```

For true online GRPO, this is not enough by itself. The trainer needs to sample
from the current policy and compute logprobs during training.

Online GRPO data flow:

```text
prompt/state
  -> current policy samples G completions/actions
  -> each candidate action or trajectory runs in Harbor
  -> reward function returns one scalar per candidate
  -> trainer computes logprobs for sampled tokens
  -> group-relative advantages are computed
  -> policy update changes token probabilities
```

If `G = 4`, one state might produce:

```text
prompt: current observation for openssl-selfsigned-cert

completion 1: <bash>mkdir -p /app/ssl ...</bash>     reward 0.09
completion 2: <bash>openssl genrsa ...</bash>        reward 0.04
completion 3: <patch path="/app/tests/x">...</patch> reward -1.01
completion 4: <finish>done</finish>                  reward -0.01
```

Group mean reward:

```text
mean = -0.2225
```

Completions 1 and 2 are above the group mean, so GRPO increases their
probability. Completion 3 is far below the mean, so GRPO decreases its
probability. Completion 4 is also below the mean because it finished too early.

At token level, the trainer is not updating "the action" as a symbolic object.
It is updating probabilities of the generated tokens:

```text
< bash > mkdir -p / app / ssl ...
```

The action parser and safety layer decide whether the token sequence is valid
and executable. The reward tells the optimizer whether to make similar token
sequences more or less likely.

## Why JSONL Is Still Useful If GRPO Needs Fresh Logprobs

Rollout JSONL is useful for:

- debugging reward hacking,
- replaying trajectories,
- SFT on successful traces,
- DPO/preference construction,
- offline analysis,
- dataset conversion,
- monitoring training data quality,
- comparing scaffolds.

It is not a complete substitute for an online RL trainer unless it also includes
policy logprobs and was sampled from the policy being updated.

So the honest statement is:

```text
Our repo defines and logs the data contract. A future online trainer must call
the same environment/reward code while sampling from the current policy.
```

## What We Can Do On A Laptop With No GPU Or API Keys

Feasible:

- run unit tests,
- run mock environment,
- install Harbor,
- run Harbor oracle on selected tasks,
- inspect task files,
- validate verifier/reward flow,
- document exact RLVR data contract,
- write scripts that dry-run configuration.

Not feasible without more resources:

- run a real LLM policy rollout,
- compute policy logprobs for a large model,
- run GRPO updates,
- benchmark trained performance,
- run distributed verl training.

This is the clean deliverable boundary:

```text
We validate the real task runtime with Harbor oracle.
We validate the RLVR data contract with unit tests and mock rollouts.
We specify exactly how online GRPO would connect.
We do not pretend to train or sample from an LLM without model access.
```

## How To Explain This In An Interview

If asked, "Why are you using Harbor if the assignment asks you to create an
environment?", answer:

> Terminal-Bench tasks are Dockerized terminal environments with verifiers.
> Harbor is the official runtime for launching and scoring those tasks. I am not
> replacing environment design with Harbor; I am using Harbor as the environment
> backend and defining the RLVR interface around it: observations, actions,
> safety checks, reward shaping, rollout logs, and trainer handoff.

If asked, "Is Terminal-Bench just 89 prompts?", answer:

> No. It is 89 stateful task environments. Each task has one initial instruction
> but produces many decision prompts during an agent rollout. Training scale
> comes from multiple rollouts per task, multiple steps per rollout, and
> eventually synthetic/private Terminal-Bench-style tasks.

If asked, "What does TRL consume?", answer:

> For online GRPO, TRL consumes prompts and samples completions from the current
> policy during training so it can compute token logprobs. The environment
> scores those completions by executing them in Harbor. Our JSONL records are
> the audit and handoff schema, and they can also support SFT/DPO, but online
> GRPO needs live policy sampling.

If asked, "Why keep the mock?", answer:

> The mock is not the proposed benchmark environment. It is a fast unit-test
> fixture for action parsing, reward composition, safety logic, rollout
> serialization, and trainer export. Harbor is the real task runtime.

## Research Note Gaps To Fix

The earlier notes need these additions:

1. `01_rlvr_framework_ecosystem.md`
   - Add a clearer separation between benchmark runtime, RLVR wrapper, and
     trainer.
   - Add a diagram showing Terminal-Bench -> Harbor -> tb_rlvr -> TRL/verl.
   - Clarify that Harbor oracle requires no LLM and is for task validation.

2. `02_rl_modeling_foundations.md`
   - Add a section on step-level vs trajectory-level RL for terminal agents.
   - Add a warning that offline JSONL without policy logprobs is not enough for
     online policy-gradient training.
   - Explain group sampling with actions/trajectories, not just one-shot text.

3. `03_terminalbench2_modeling_strategy_review.md`
   - Add real task anatomy examples like the five above.
   - Add the answer to "is it just 89 prompts?"
   - Add public benchmark contamination guidance.

4. `04_take_home_solution_plan.md`
   - Mark it as superseded by official docs plus this lifecycle note.
   - Replace any language that makes mock env sound like the main environment.
   - Tighten the execution boundary: Harbor oracle and mock are feasible;
     policy rollouts and GRPO need model access/GPU.

5. Official docs later
   - `docs/submission_writeup.md` and `docs/implementation_plan.md` should be
     updated after we agree on this story.
   - The known limitations section should become an execution-boundary section.

## Sources Consulted

- AfterQuery take-home PDF in this repo.
- Terminal-Bench 2.0 Harbor registry:
  https://www.harborframework.com/registry/terminal-bench/2.0
- Harbor task docs:
  https://harborframework.com/docs/tasks
- Harbor Terminal-Bench running docs:
  https://harborframework.com/docs/running-tbench
- Public Terminal-Bench 2 task repository:
  https://github.com/harbor-framework/terminal-bench-2
