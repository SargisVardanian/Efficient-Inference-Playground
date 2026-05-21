# Project Notes

## Confirmed Local Facts

- Project folder: `/Users/sargisvardanyan/Efficient Inference Playground`
- The current comparison matrix is mixed-backend:
  - baseline: Hugging Face Transformers
  - quantized: Ollama GGUF fallback
  - KV-cache: Hugging Face Transformers
- Current benchmark scope is Needle short, medium, and long with 10 prompts per bucket.
- `hf_baseline_all_clean.csv` is the canonical baseline file.
- `ollama_quantized_short.csv`, `medium`, and `long` are complete.
- The original Gemma4 attention-sink KV attempt was replaced with a hybrid recency-window policy to match Gemma4's hybrid attention structure.
- `hf_kv_short.csv` is partially complete under the rerun, and the Gemma4 KV path is much heavier than baseline.

## Assumptions

- The team wants one schema even if one row must fall back to another backend.
- Apple Silicon is the active local target, so Metal/MPS behavior and unified-memory pressure are central constraints.
- Gemma4 remains the primary KV target for fairness unless it becomes impossible to finish.

## Open Checks

- Add real team member names to `team_members.txt`.
- Confirm whether Gemma4 KV can complete short/medium/long in acceptable time on this machine.
- If Gemma4 KV proves infeasible, document `Qwen2.5-1.5B-Instruct` only as a fallback candidate, not as the primary report row.
