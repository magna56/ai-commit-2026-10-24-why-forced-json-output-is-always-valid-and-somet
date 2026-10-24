# Why Forced JSON Output Is Always Valid and Sometimes Wrong

**TL;DR:** Forced JSON mode removes the tokens your schema forbids and rescales the rest, so the output always parses. The model still chooses among legal tokens. When it wanted something illegal, it guesses, and the guess looks as sure as a real answer.

Published from [The AI Commit](https://theaicommit.com/#2026-10-24/code) — Hands-on Techniques, 2026-10-24.

## Run

```bash
python3 code_example.py
```

## Output

```
1. A forced pick: the 'confidence' field, one of high / medium / low
   clear invoice: kept 0.95 of the model's probability -> high 0.84, medium 0.13, low 0.03
                  most likely output: "high", and the JSON looks the same either way
   blurry scan  : kept 0.15 of the model's probability -> high 0.40, medium 0.33, low 0.27
                  most likely output: "high", and the JSON looks the same either way
   add 'unknown' to the enum: blurry scan -> unknown 0.77

2. A whitespace loop to max_tokens = 2048
   whitespace share 0.900 each step: average run   10 tokens, P(all 2048) = 1.9e-94
   whitespace share 0.990 each step: average run  100 tokens, P(all 2048) = 1.2e-09
   whitespace share 0.999 each step: average run 1000 tokens, P(all 2048) = 1.3e-01
   a 1% chance of filling 2048 tokens needs a share of 0.9978 at every step:
   the model has to be nearly certain, which is repetition feeding itself, not leftover probability

3. Lint a schema before you ship it
   WARN confidence: enum ['high', 'medium', 'low'] has no way to say 'unknown', so a doubtful model must pick one
   WARN confidence: written before 'reasoning', so the answer cannot depend on the reasoning
   WARN vendor: free string with no maxLength; bound it or use an enum
   WARN vendor: written before 'reasoning', so the answer cannot depend on the reasoning
   fixed schema: 0 warnings
   ["reasoning", "vendor", "confidence"]

```

## Code

See [`code_example.py`](code_example.py).
