# EvalPlus benchmark data

These public releases are used for evaluation only. Keep benchmark solutions and tests out of training data and synthesis inputs.

| Benchmark | Release | Problems | Upstream asset |
| --- | --- | --- | --- |
| HumanEval+ | v0.1.10 | 164 | [HumanEvalPlus.jsonl.gz](https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz) |
| MBPP+ | v0.2.0 | 378 | [MbppPlus.jsonl.gz](https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz) |

The JSONL files preserve the release data. [manifest.json](manifest.json) records the evaluator version, source URLs, row counts, and content hashes. The loader verifies SHA-256 and the complete problem ID set before scoring. MBPP+ uses the sanitized split, whose task numbers are not consecutive.

From the repository root, validate both datasets without loading a model:

```bash
python -m autosft.benchmarks.catalog check humaneval_plus mbpp_plus
```

See the [runtime guide](../../../docs/local-runtime.md#code-benchmark-data) for dependency installation, alternate data directories, and evaluation commands.
