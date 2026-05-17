# Project Notes

## Confirmed Local Facts

- Project folder: `/Users/sargisvardanyan/Efficient Inference Playground`
- Ollama is installed and reachable if `ollama list` works.
- Local model detected at scaffold time: `gemma4:e4b`.

## Assumptions

- The team will use Gemma 4 in Ollama for baseline and local quantization experiments.
- Quantized experiments require additional installed Ollama tags or custom Modelfiles.
- Speculative decoding will be easier to demonstrate with Hugging Face model pairs than with Ollama.

## Open Checks

- Add real team member names to `team_members.txt`.
- Confirm whether a CUDA machine, Apple Silicon only, or Colab will be used for Hugging Face experiments.
- Install quantized Ollama tags and enable them in `configs/ollama_gemma4.json`.

