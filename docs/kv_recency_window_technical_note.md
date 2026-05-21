# KV Recency Window Note

Date: 2026-05-21

## Purpose

This note explains why the current Gemma4 KV-cache experiment is slow on the short bucket, why that does not look like a broken run, and how to think about the recency-window size `W`.

## Current Setup

Active KV row:

- model: `google/gemma-4-E4B-it`
- backend: `Hugging Face Transformers`
- KV policy: `hybrid recency_window`
- current window: `W = 512` for short/medium; final long run uses `W = 768`

Policy details:

- native `sliding_attention` layers are preserved
- `full_attention` layers keep a contiguous recency window
- this replaced the earlier attention-sink-style attempt, which was incompatible with Gemma4's hybrid cache-mask contract in the current HF runtime

## Observed Short-Bucket Behavior

Prompt length for the short bucket is roughly:

- `226-238` prompt tokens

Observed latency:

- baseline short p50: `293.7s`
- baseline short min/max: `162.0s / 581.3s`
- KV recency short p50: `314.3s`
- KV recency short min/max: `187.3s / 522.3s`

Interpretation:

- baseline was already very slow on this machine
- KV short remains in the same order of magnitude
- the current slowdown is roughly `+7%` on p50, not an order-of-magnitude failure

## Why KV Short Did Not Speed Up

The main reason is simple:

- `W = 1024` in the original short-bucket run
- short prompts are only about `230` tokens
- therefore the prompt does not exceed the recency window

That means:

- the cache policy does not actually evict much or anything meaningful on short prompts
- the potential memory or attention-shape savings are minimal
- but the custom cache path still adds some overhead

So on short prompts we mostly observe:

- baseline: already heavy Gemma4 HF path
- KV recency: same heavy path plus cache-management overhead, but without enough eviction benefit

This is why the short bucket is a weak place to expect improvement from the current KV policy.

## Why This Does Not Look Like a Bug

The current short-bucket result does not look like a clearly broken implementation because:

- output quality remains sane
- exact match did not collapse
- throughput remains in the same order as the baseline
- the slowdown is small relative to the already very high baseline latency

If the KV implementation were badly broken, we would more likely expect:

- invalid outputs
- extreme latency blow-up
- crashes
- or a severe quality drop

That is not what we are observing on the short bucket.

## Gemma4-Specific Constraint

Gemma4 is not a simple single-attention-pattern decoder.

It combines:

- `sliding_attention` layers
- periodic `full_attention` layers
- shared KV behavior in part of the stack

Because of that, the earlier non-contiguous `attention_sink` idea was not safe in the current HF runtime for this model. The current hybrid `recency_window` policy is a compatibility-driven compromise:

- it preserves Gemma4's native local-attention structure
- it still performs KV compression on the full-attention layers
- but it is not the same thing as a pure StreamingLLM sink-token cache

## Is One Global W Ideal?

Not really.

Current consequences of a fixed `W = 512` for short/medium:

- short: too small to trigger meaningful eviction
- medium: should begin to show real cache effects
- long: should show the strongest compression effect

So one global `W` would be:

- good for a simple, controlled, one-policy-for-all experiment
- not optimal for every bucket

## Why The Main Row Was Switched To 512

After the first short-bucket rerun with `W = 1024`, the observed behavior was:

- no speedup on short
- only overhead was visible
- and the bucket was too small to exercise meaningful eviction

Because of that, the primary KV row was updated to:

- `W = 512`

Reason:

- `512` is closer to Gemma4's native sliding-window scale
- it should force meaningful compression earlier
- it is more likely to make medium and long informative
- and it gives short at least a better chance of showing some effect

This change means KV rows should be rerun under the updated definition.

## When W = 512 Makes Sense

`W = 1024` remains a useful optional follow-up if we want an ablation study.

Possible rationale:

- it is looser and more conservative than `512`
- it may preserve more quality
- but it may delay any measurable speed benefit until longer prompts

That makes it a good secondary experiment, not necessarily the best primary one.

## Recommended Interpretation For The Report

Use this framing:

1. The short bucket is too short to benefit materially from the current `W = 1024` KV eviction policy.
2. Therefore short primarily reflects the overhead of the custom cache path rather than the benefit of eviction.
3. Medium and long are the buckets where the current policy is expected to become informative.
4. The original `W = 1024` short run showed that a too-large window can hide the intended effect of KV eviction.
5. The active main row now uses `W = 512`, and any final KV comparison should be based only on reruns collected under that updated policy.

## Practical Recommendation

Primary row:

- use `W = 512` for short/medium
- use the completed `W = 768` long run as the final long KV row

Optional extra ablation if time remains:

- rerun a comparison with `W = 1024`, or
- do a small sweep such as `W in {512, 1024}`

That gives both:

- one clean main experiment
- and one optional sensitivity analysis
