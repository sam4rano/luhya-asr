# AGENTS.md

## Rules

### 1. Verify API surface against official docs before making changes

When modifying any call to a third-party library API (especially Hugging Face Transformers, PyTorch, datasets, etc.), **always** consult the official documentation for the installed version before writing code:

- Hugging Face Transformers: `https://huggingface.co/docs/transformers/v<major>.<minor>.<patch>/en/<page>`
- PyTorch: `https://pytorch.org/docs/stable/`
- Hugging Face Datasets: `https://huggingface.co/docs/datasets/`

**Never** rely on memory for API signatures, parameter names, or class constructors — they change between versions and what was valid in Transformers 4.x may be deprecated or removed in 5.x.

Before using a class, method, or parameter, either:
- Fetch the relevant doc page with `WebFetch` to verify the signature
- Grep the installed package source for the parameter to confirm it exists

This applies especially to:
- `Trainer.__init__()` parameters (e.g., `group_by_length` removed in 5.x)
- `TrainingArguments` fields (e.g., `eval_strategy` replaced `evaluation_strategy`)
- Model factory calls (e.g., `attn_implementation` support varies by architecture)
- Tokenizer/processor APIs
