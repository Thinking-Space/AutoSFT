# Third-party code

The instruction-following evaluation path includes checkers from [Google IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [AllenAI IFBench](https://github.com/allenai/IFBench) and [OpenCompass](https://github.com/open-compass/opencompass). Their licenses and notices remain with the corresponding source directories. Bundled verifier dependencies retain their package license files.

Megatron-LM is obtained from [NVIDIA's public repository](https://github.com/NVIDIA/Megatron-LM) at an explicit full commit through `scripts/setup/fetch_megatron.py`. Its checkout is a local ignored dependency; the generated lock records the chosen upstream identity. AutoSFT does not distribute its internal Megatron fork.

Public benchmark input files live under `assets/benchmarks`. Code benchmarks use the checked dataset manifest and public EvalPlus package. Training data are downloaded separately and retain their publishers' licenses.
