"""Is the diffusion-LLM speedup a function of script?"""

__version__ = "0.1.0"

# Shared by Dream-7B, so fertility is identical for the AR and diffusion arms.
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
AR_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DLLM_MODELS = ("Dream-org/Dream-v0-Instruct-7B", "GSAI-ML/LLaDA-8B-Instruct")

PIVOT_LANG = "en"
# docs/prereg-g0.md: below this the x-axis is too compressed to resolve anything.
MIN_FERTILITY_SPREAD = 1.5
# Relative spread of parallel_factor required to call the effect real.
MIN_PARALLEL_SPREAD = 0.20
MIN_LANGS = 6
MIN_ITEMS = 90
