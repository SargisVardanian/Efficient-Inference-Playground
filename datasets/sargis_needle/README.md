# Sargis Needle Benchmark Dataset

This folder contains the exact 30 prompts used for Sargis's Needle-in-a-haystack benchmark results.

## Files

- `prompts.jsonl` - canonical 30-prompt dataset used for the final Sargis benchmark workbook.

## Composition

| Bucket | Prompts | Source file mirrored |
| --- | ---: | --- |
| short | 10 | `prompts/needle_short_prompts.jsonl` |
| medium | 10 | `prompts/needle_medium_prompts.jsonl` |
| long | 10 | `prompts/needle_long_prompts.jsonl` |

Each JSONL row has:

- `id`
- `length_bucket`
- `task`
- `dataset`
- `reference`
- `prompt`

## Result Linkage

The final result workbooks generated from this dataset are:

- `results/sargis_needle_benchmark_results.xlsx`
- `results/team_final_benchmark_results.xlsx`

Hardware for these Sargis runs: Apple M3 Pro.

Silvi's comparison rows in the team workbook were run on Apple M4 Max.
