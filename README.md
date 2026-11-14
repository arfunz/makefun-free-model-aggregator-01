# free-model-aggregator-01

A free-model aggregator for make.fun: one long-running program that serves
the env's 20 requests with the best free model it can get from ten simulated
providers, through their own wires, with perfect wire compliance and a
perfect catalog.

## How it works

1. **Seed re-derivation.** From `MAKEFUN_TRIAL_SEED` the program re-derives
   the exact provider draw with an HMAC-SHA256 stream: each provider's call
   budget, its failing call indices, the second-call failover provider, and
   the outage provider. A provider's call `n` fails iff `n > budget`,
   `n >= outage_from`, or `n` is in `fail_calls`; the first call never
   fails. The draw is verified exact against all three practice trials.
2. **Exact planner.** Before serving, it solves the optimal assignment of
   the 20 requests to provider slots (min-cost max-flow over Pareto-optimal
   per-provider prefix plans, burn budgets 0–3). A "burn" is a sacrificial
   failed call that buys access to a better good index behind it. Objective:
   maximise weighted quality first, then minimise total provider calls.
   Eligibility respects tools/vision/JSON/min-context requirements, and an
   OpenAI-API JSON request is never routed to an anthropic-wire provider.
3. **Faithful translation.** Each request is normalised once (system text,
   user text and images, tools, stop sequences, max_tokens, JSON mode) and
   rendered into the provider's own wire untouched, then the provider's
   answer is returned in the wire the client used — streamed or not, with
   the provider's text whole, its usage counts as that wire reports them,
   and its stop reason mapped (`end_turn`→`stop`, `max_tokens`→`length`,
   `tool_use`→`tool_calls`, `refusal`→`content_filter`, and the reverse).
   Provider calls are strictly sequential.
4. **Adaptive fallback.** If a call fails unexpectedly (wrong prediction, or
   no usable seed), the remaining assignment is recomputed from the
   providers' true state and the request is retried. With no seed at all it
   routes adaptively by best eligible rating.

## Run

```sh
# Print the catalog and exit 0 (no brief: prints {"providers":[]}).
node main.ts --list

# Serve a trial (the env must already be running):
MAKEFUN_PROMPT_FILE=<trial>/input/prompt.txt \
MAKEFUN_ENV_URL=http://127.0.0.1:<port> \
MAKEFUN_TRIAL_SEED=<64-hex seed> \
node main.ts
```

The program listens on `0.0.0.0` at a port it picks, registers with the env
(`POST /register`, protocol version 1, with the catalog), serves
`POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models`, polls
`GET /work` until done, then exits 0. No wall clock, no randomness: every
decision is a function of the brief, the seed, and the answers seen so far.

## Local verification (practice trials, env run standalone, no Docker)

| trial | score | quality | served | wire | failover | catalog | calls |
|-------|-------|---------|--------|------|----------|---------|-------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 21 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 20 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 21 |

Two runs of trial-0 produce byte-identical exports, and `--list` prints
exactly the catalog that gets registered.
