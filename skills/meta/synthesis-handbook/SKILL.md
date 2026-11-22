---
name: synthesis-handbook
description: Consult synthesis for undersupplied capabilities and use accepted data without changing the fixed training protocol.
---

# Synthesis handbook

The meta agent may use the synthesis subagent when `synthesis` is present in the run's component list. This selection is independent of the executor and analysis components. Consult synthesis each round; generate examples only for evidence-backed supply gaps. Keep generated data separate from the staged source pool until publication.

Use `python -m autosft.synthesis.job --help` for decision, request, launch, status and abandonment commands. Targets contain capability requirements and verifiers. The worker generates candidates with the recorded batch inference runner and delivers `data.jsonl`, `report.md` and `experience_delta.md`. It cannot alter the run protocol or read private benchmark answers.

A delivery must contain the requested number of valid unique conversations and pass capability verification. The supervisor checks selected-benchmark n-gram overlap and obtains an independent semantic review of a deterministic sample before publication. Inspection reports are bound to the delivery hash and retained under `synthesis_inspections/`. Failed or unknown checks do not admit the batch.

Accepted batches live under `syn_data_source/`; the meta agent decides whether they enter training. The fixed token cap applies to source and synthetic supervision together. After the first round, record the synthesis decision for each upcoming round, including a decision not to launch. Recover terminal workers and publication receipts before retrying. If every attempt failed without a published batch, record an attributable abandonment before continuing.

Publication and training use are distinct. Preparation previews usage from the proposed data; successful training confirms usage from frozen submission inputs. Status recovery repairs interrupted accounting. Keep experience and publication history durable; use the run's own evidence to decide later targets.
