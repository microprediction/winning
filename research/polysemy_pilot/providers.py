"""Provider registry and capability probe for exact-probability elicitation.

The batteries in this directory need the top-k alternatives at a single token
position, not merely the log probability of the token the model happened to
choose. Providers differ, and vendor documentation is not a reliable guide, so
nothing here is assumed: `probe_providers.py` sends one real call per provider
and records what actually comes back.

Three capability levels matter:

  topk    top_logprobs returns k alternatives at one position. One call per
          cell, exactly like the OpenAI batteries already in this directory.
  chosen  only the sampled token's log probability is returned. Still usable
          by scoring each candidate item in a separate call (see
          `SCORING_NOTE`), at one call per item rather than per cell.
  none    no log probabilities at any granularity. Cannot participate in the
          exact-probability designs; sampling designs only.

Keys are read from winning/.env. Absent keys are reported as skipped rather
than failing, so the probe is safe to run with any subset configured.
"""

SCORING_NOTE = """\
Where only chosen-token log probabilities are available, an item's probability
can still be measured exactly by continuation scoring: send the prompt plus
the candidate item as a forced continuation and read the log probability
assigned to the item's first token. This costs one call per candidate instead
of one per cell, and needs either an echo/prefill option on a completions
endpoint or a provider that scores assistant prefills."""

# env_var: the key name expected in winning/.env
# base_url: OpenAI-compatible endpoint, or None for a bespoke client
# models: cheap probe targets, ideally including a base (non-instruct) model
PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "base_url": None,  # native client
        "models": ["gpt-4.1-nano"],
        "expected": "topk",
        "note": "verified topk=20; the gpt-5 tier returns 403 for logprobs",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["meta-llama/llama-3.3-70b-instruct",
                   "qwen/qwen-2.5-7b-instruct",
                   "mistralai/mistral-7b-instruct"],
        "expected": "topk",
        "note": "one key, many labs; passthrough depends on the upstream",
    },
    "fireworks": {
        "env_var": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "models": ["accounts/fireworks/models/llama-v3p1-8b-instruct"],
        "expected": "topk",
        "note": "has a text-completions endpoint, which suits cloze prompts",
    },
    "together": {
        "env_var": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"],
        "expected": "chosen",
        "note": "top-k historically limited on chat; try completions",
    },
    "hyperbolic": {
        "env_var": "HYPERBOLIC_API_KEY",
        "base_url": "https://api.hyperbolic.xyz/v1",
        "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
        "expected": "topk",
        "note": "serves base checkpoints, the venue for base-vs-instruct",
    },
    "deepinfra": {
        "env_var": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "models": ["meta-llama/Meta-Llama-3.1-8B-Instruct"],
        "expected": "topk",
        "note": "",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat"],
        "expected": "topk",
        "note": "",
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ["gemini-2.0-flash"],
        "expected": "unknown",
        "note": "native API uses responseLogprobs; OpenAI-compat layer varies",
    },
    "xai": {
        "env_var": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "models": ["grok-3-mini"],
        "expected": "unknown",
        "note": "",
    },
    "moonshot": {
        "env_var": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.ai/v1",
        "models": ["kimi-k2-0711-preview", "moonshot-v1-8k"],
        "expected": "unknown",
        "note": "OpenAI-compatible; logprob support unverified",
    },
    "zai": {
        "env_var": "Z_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "models": ["glm-4.5-air", "glm-4-flash"],
        "expected": "unknown",
        "note": "Zhipu GLM family; OpenAI-compatible path",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile"],
        "expected": "none",
        "note": "expected to ignore or reject logprobs",
    },
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest"],
        "expected": "none",
        "note": "",
    },
}

# Providers that cannot participate in exact designs at all, recorded so the
# panel's coverage gaps are explicit rather than forgotten.
NO_LOGPROBS = {
    "anthropic": "exposes no log probabilities; sampling designs only, which "
                 "is why the Claude replication is the one sampled design",
    "perplexity": "no log probabilities",
    "bedrock": "most served models expose none",
}

# Full exact distributions over the entire vocabulary, no key required.
LOCAL = {
    "mlx_lm": "fast path on Apple silicon; exact full-vocabulary logits",
    "transformers": "portable; exact full-vocabulary logits",
    "vllm": "OpenAI-compatible server with top_logprobs",
    "tgi": "top_n_tokens returns top tokens per position",
}
