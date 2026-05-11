# Terminal-Bench To RLVR Data Lifecycle

This document explains the concrete data flow behind the submission. It is the
bridge between the high-level environment design and the code in `src/tb_rlvr`.

## Short Answer

Terminal-Bench is not a table of prompts. It is a suite of stateful terminal
tasks. Harbor runs those tasks. The RLVR environment is the contract around
Harbor:

```text
Terminal-Bench task
  -> Harbor Docker/runtime execution
  -> tb_rlvr observation/action/reward/safety/rollout schema
  -> future online GRPO trainer
```

Using Harbor does not remove the need to design an RLVR environment. Harbor
launches and scores terminal tasks. This repo defines what the policy sees, what
it can do, how rewards are decomposed, and what trajectory data is logged.

## What A Task Looks Like

A Terminal-Bench task is a directory with files like:

```text
task-id/
  instruction.md
  task.toml
  environment/Dockerfile
  solution/solve.sh
  tests/test.sh
  tests/test_outputs.py
```

The instruction is the initial problem statement. The Docker environment is the
world the agent acts in. The tests/verifier define success. The solution is an
oracle/reference path used for task validation; a policy should not see it.

## Is Terminal-Bench Just 89 Prompts?

No. Terminal-Bench 2.0 contains 89 task environments in the public registry,
but each task can produce many model prompts during a rollout.

```text
one task instruction
  -> observation prompt at step 1
  -> action 1
  -> stdout/stderr/filesystem changes
  -> observation prompt at step 2
  -> action 2
  -> ...
  -> final verifier
```

If one task takes 20 steps and we sample 8 attempts, that one task can produce
160 step-level records. For real post-training, the public tasks are not enough
as a training set. They are best used for evaluation, smoke tests, task-design
examples, and held-out validation. Scale would come from private or synthetic
Terminal-Bench-style tasks.

## Harbor Oracle Smoke

The laptop-feasible Harbor check is an oracle run. Oracle means Harbor runs the
task's reference solution, not an LLM.

Dry-run the command from this repo:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert
```

If Docker and Harbor/uvx are available, execute it:

```bash
python3 scripts/harbor_oracle_smoke.py --task openssl-selfsigned-cert --execute
```

The equivalent Harbor command is:

```bash
uvx harbor run -d terminal-bench@2.0 -t openssl-selfsigned-cert -a oracle
```

This validates the benchmark runtime and verifier. It does not validate an LLM
policy.

## Example: `openssl-selfsigned-cert`

The task asks the agent to create a self-signed TLS certificate setup under
`/app/ssl`, including a private key, certificate, PEM file, verification text,
and a Python checker.

Initial observation prompt:

```text
Task: openssl-selfsigned-cert

Instruction:
Create a self-signed TLS certificate setup under /app/ssl...

CWD: /app

Directory summary:
(initial task files)

Recent actions:

Last stdout:

Last stderr:

Steps remaining: 30
```

Candidate action:

```xml
<bash>mkdir -p /app/ssl && openssl genrsa -out /app/ssl/server.key 2048 && chmod 600 /app/ssl/server.key</bash>
```

The runtime executes the command, captures outputs, updates the filesystem, and
builds the next observation. Our reward layer may give bounded progress for
creating the expected key file, but final success comes only from the verifier.

Final record shape:

```json
{"task_id":"openssl-selfsigned-cert","episode_id":"openssl-selfsigned-cert:run-001","backend":"harbor","step":6,"observation_prompt":"Task: openssl-selfsigned-cert\n\nInstruction:\nCreate a self-signed TLS certificate setup under /app/ssl...\n\n...","model_output":"<finish>ready for grading</finish>","action":{"kind":"finish","path":null,"payload":"ready for grading"},"reward":{"success":1.0,"progress":0.0,"integrity":0.0,"step":-0.06,"token":-0.0001,"total":0.9399},"done":true,"terminal_reason":"success","observation_hash":"hash_before_final","next_observation_hash":"hash_after_final","info":{"dataset":"terminal-bench@2.0","verifier_ran":true,"verifier_reward_raw":"1","safety_ok":true}}
```

## Example: `fix-code-vulnerability`

The task gives a Bottle web framework repository in `/app`, asks the agent to
find a CWE-class vulnerability in `/app/bottle.py`, write `/app/report.jsonl`,
patch the vulnerability, and make tests pass.

Typical trajectory:

```xml
<bash>grep -n "except\|raise\|ValueError" /app/bottle.py | head -80</bash>
```

```xml
<patch path="/app/bottle.py">...modified vulnerable function...</patch>
```

```xml
<bash>printf '%s\n' '{"file_path":"/app/bottle.py","cwe_id":["cwe-20"]}' > /app/report.jsonl && pytest -rA</bash>
```

```xml
<finish>ready for grading</finish>
```

The verifier checks both the report and the code behavior. Public test output
can be used as observation, but hidden verifier details must not be exposed.

## Example: `sqlite-db-truncate`

The task gives a corrupted SQLite database and asks the agent to recover rows
into `/app/recover.json`.

Typical trajectory:

```xml
<bash>file /app/trunc.db && ls -lh /app/trunc.db && sqlite3 /app/trunc.db '.tables'</bash>
```

```xml
<patch path="/app/recover_sqlite.py">...script that scans pages and extracts records...</patch>
```

```xml
<bash>python /app/recover_sqlite.py > /app/recover.json && python -m json.tool /app/recover.json | head</bash>
```

```xml
<finish>ready for grading</finish>
```

Progress reward can check that JSON exists and parses. Final reward depends on
the hidden verifier's row-recovery criteria.

## What Gets Handed To A Trainer?

For audit, SFT, DPO, or offline analysis, each rollout record can become:

```json
{"prompt":"Task: ...","completion":"<bash>...</bash>","reward":0.0898,"metadata":{"task_id":"openssl-selfsigned-cert","step":1,"terminal_reason":"","reward_components":{"success":0.0,"progress":0.1,"integrity":0.0,"step":-0.01,"token":-0.0002,"total":0.0898}}}
```

For true online GRPO, this JSONL is not enough by itself. The trainer must
sample from the current policy during training so it has current token logprobs.

Online GRPO flow:

```text
observation prompt
  -> current policy samples G candidate actions/trajectories
  -> Harbor executes/scored each candidate
  -> reward function returns one scalar per candidate
  -> trainer computes token logprobs
  -> group-relative advantages are computed
  -> policy update changes token probabilities
```

If `G = 4`, one state may produce:

```text
candidate 1: <bash>mkdir -p /app/ssl ...</bash>       reward 0.09
candidate 2: <bash>openssl genrsa ...</bash>          reward 0.04
candidate 3: <patch path="/app/tests/x">...</patch>   reward -1.01
candidate 4: <finish>done</finish>                    reward -0.01
```

GRPO increases probability of above-average completions and decreases
below-average completions. The update happens at the token level, even though
we reason about actions symbolically.

## Laptop Boundary

Feasible without GPU or API keys:

- run unit tests,
- run mock rollout,
- dry-run Harbor commands,
- run Harbor oracle if Docker/uvx/Harbor are available,
- inspect reward/verifier artifacts,
- export toy JSONL samples,
- validate the data contract.

Not feasible without additional resources:

- real LLM policy rollouts,
- token logprob collection for a large model,
- GRPO updates,
- benchmark improvement claims,
- distributed verl training.

This is the honest deliverable boundary for the current machine.
