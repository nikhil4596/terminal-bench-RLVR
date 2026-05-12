# Internal Learning Notes And Plan Critique

This file is not part of the submission-facing deliverable. It keeps the
learning material and earlier-plan critique separate from the official README
and assignment write-up.

## Main Learning Points

- The take-home asks for an RLVR environment design and repo, not a completed RL
  training run.
- The strongest answer is environment-first: Terminal-Bench task execution,
  observation/action design, reward design, rollout logging, and trainer
  handoff.
- Harbor is the natural task-execution layer for Terminal-Bench 2.0.
- TRL and verl should be treated as trainer backends, not as replacements for
  the benchmark environment.
- A no-training repo can still be large-scale ready if it defines stable
  rollout records, reward components, config boundaries, and adapter seams.
- GRPO is a strong default algorithm because verifiable rewards can compare
  groups of sampled actions/trajectories without requiring a separate value
  model.
- Reward decomposition is critical. The scalar reward feeds the optimizer, but
  the components are needed for diagnosing hacking, overfitting, and safety
  failures.

## Critique Of The Earlier Naive Plan

The earlier plan had useful instincts but was not yet submission-grade.

Main issues:

- It over-centered trainer choice instead of environment design.
- It did not cleanly separate Harbor execution from TRL/verl training.
- It did not define a rollout schema that could be audited or exported.
- It did not specify action grammar enough for stable training.
- It did not address hidden tests, oracle protection, or benchmark tampering
  rigorously.
- It did not clearly state what is and is not implemented.
- It did not provide a readiness path from local prototype to large-scale RL.
- It did not explain what capabilities are lost by not wiring TRL/verl directly
  into core code.

## Improved Direction

The final submission should present:

- Terminal-Bench 2.0 as the selected benchmark.
- Harbor as the environment substrate.
- TRL as the pilot GRPO trainer.
- verl as the large-scale distributed trainer.
- Qwen2.5-Coder-7B-Instruct as the pilot base model.
- A constrained terminal-agent action protocol.
- Verifiable success reward plus bounded progress and safety rewards.
- Rollout records with prompt, action, reward, metadata, and hashes.
- A clear statement that no training was run.

## Where The Learning Docs Live

The in-depth learning docs are preserved in this directory. They are useful for
interview prep and deeper study, but they should not be mixed into the final
take-home submission docs.
