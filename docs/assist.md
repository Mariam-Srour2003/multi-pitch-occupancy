# Second opinion from a free vision model

The detector counts people inside each camera's pitch boundary. Its main mistake, measured on
848 labelled frames, is a real game read as "a few people" when players at the far side are
too small to find. A vision-language model reads the whole scene instead of counting boxes, so
the playground runner can ask one about the doubtful checks. Only free tiers are used, and
each provider is held to half of its free daily limit.

## What it changes

| the detector found | the model says | result |
|---|---|---|
| 1 to 4 people (fewer than a game needs) | playing, confidence ≥ 0.7 | the check becomes **playing**; `playing_source = 'assist'`, `detector_playing` keeps the original |
| 1 to 4 people | anything else | unchanged |
| a game, or nobody | anything | unchanged; the opinion is stored for comparison |

It never turns a game off and never makes a pitch empty. The detector made no such mistake on
the labelled frames, and a model that can be wrong should not decide a NOTUSED on its own.
Every question and answer is stored in the `assist_opinions` table.

## How often

These are the defaults in `configs/assist.json`:

- At most **3 questions per playground every 30 minutes**, and never two within 5 minutes.
- Doubtful checks are asked about first. When nothing doubtful happens, one ordinary check per
  window is asked about (`audit`), so the stored opinions show how often the two disagree.
- All cameras of a playground go in **one** request (up to 3 pictures).

One playground checked all day uses at most 144 requests a day. The budget refuses any request
that would pass half of a provider's free limit. It then moves on to the next provider in the
list. If none is left, the check simply keeps the detector's answer.

## Providers (free tiers, checked 2026-10-20)

| name | model | key variable | free limit (budget allows half) |
|---|---|---|---|
| `gemini` | `gemini-3.5-flash-lite` | `GEMINI_API_KEY` | about 500 requests/day, 15/min |
| `groq` | `qwen/qwen3.8-27b` | `GROQ_API_KEY` | 1,000 requests/day, 200,000 tokens/day (about 80 pictures) |
| `openrouter` | `google/gemma-4-31b-it:free` | `OPENROUTER_API_KEY` | 50 requests/day |

Free lists change. When a provider drops a model, set another one under `models` in
`configs/assist.json`, for example `"models": {"groq": "<new model id>"}`, without a code
change. A quota refusal (HTTP 429) rests that provider until its quota resets. Nothing is
retried in a loop.

**Use keys from accounts with no billing attached:**

- **Google:** a Google AI Studio project without a billing account.
- **Groq:** the free plan, with no payment method.
- **OpenRouter:** an account without credits.

Such a key cannot produce a bill, whatever the software does.

## Turning it on

1. Get a key from at least one provider and put it in `.env` in the project folder (gitignored):
   ```
   GEMINI_API_KEY=...
   GROQ_API_KEY=...
   ```
2. Copy `configs/assist.example.json` to `configs/assist.json` and set `"enabled": true`.
3. Check that it answers: `uv run pitch assist-test path/to/a/picture.jpg`.
4. Restart the server. Doubtful checks are now asked about. The `/results` page shows how many
   checks were upgraded.

## Privacy

Pictures of people are sent to the provider. Free tiers may use what they receive to improve
their models, so read the provider's terms before turning this on. Heads are pixelated before
sending (`blur_heads`), and pictures are shrunk to at most 1024 pixels on the long side. Keys
are read from the environment only, never written to a file in the repository, and never
logged.
