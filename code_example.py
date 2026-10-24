"""Why forced JSON output is always valid and sometimes wrong.

Structured-output modes (OpenAI and Gemini response schemas, vLLM and SGLang via XGrammar,
llama.cpp grammars) all run the same loop at every token:

  1. the model scores every token in its vocabulary,
  2. the engine deletes the tokens the grammar forbids at this position,
  3. it renormalizes what is left and samples from it.

Step 3 is the whole story. The grammar decides what is legal, never what is right. This file
computes the three things that follow from it, and gives you two helpers to lift:

  kept_mass()          -- how much of the model's probability survived the mask. Low means forced.
  check_schema()       -- lint a JSON schema for the three shapes that invite forced output.

Run: python3 code_example.py   (pure Python, no model, no API key)
"""
import json


# ---- the mask -------------------------------------------------------------------------
def mask_and_renormalize(probs, allowed):
    """Step 2 and 3 of the loop: drop forbidden tokens, rescale the rest to sum to 1."""
    kept = {t: p for t, p in probs.items() if t in allowed}
    total = sum(kept.values())
    return {t: p / total for t, p in kept.items()}, total


def kept_mass(probs, allowed):
    """The share of the model's probability the grammar left standing at one position."""
    return sum(p for t, p in probs.items() if t in allowed)


# ---- 1. a forced pick ---------------------------------------------------------------------
# The schema says "confidence" is one of high / medium / low. These are the model's own
# next-token probabilities at that value, for a clear invoice and for a blurry scan.
# "unknown" and "I can't" stand for every token outside the enum.
ENUM = {"high", "medium", "low"}
CLEAR = {"high": 0.80, "medium": 0.12, "low": 0.03, "unknown": 0.03, "I can't": 0.02}
BLURRY = {"high": 0.06, "medium": 0.05, "low": 0.04, "unknown": 0.50, "I can't": 0.35}

# ---- 2. a whitespace loop -----------------------------------------------------------------
def p_run_reaches(w, n_tokens):
    """If whitespace keeps share w at every step, independently, P(n whitespace tokens in a row)."""
    return w ** n_tokens


def w_needed(n_tokens, p=0.01):
    """The per-step whitespace share needed for a p chance of running all n tokens."""
    return p ** (1 / n_tokens)


# ---- 3. the schema lint -------------------------------------------------------------------
REASON_WORDS = ("reason", "rationale", "explanation", "thinking", "evidence")


def check_schema(schema, answer_fields):
    """Return warnings for the three shapes that turn the mask into a guess.

    answer_fields: the keys your code acts on (a label, a score, a decision).
    """
    warns, props = [], schema.get("properties", {})
    keys = list(props)
    for name in answer_fields:
        spec = props.get(name, {})
        enum = spec.get("enum")
        if enum and not any(v in (None, "unknown", "none", "unclear") for v in enum):
            warns.append(f"{name}: enum {enum} has no way to say 'unknown', so a doubtful "
                         "model must pick one")
        if "enum" not in spec and spec.get("type") == "string" and "maxLength" not in spec:
            warns.append(f"{name}: free string with no maxLength; bound it or use an enum")
        reasons = [k for k in keys if any(w in k.lower() for w in REASON_WORDS)]
        if reasons and keys.index(reasons[0]) > keys.index(name):
            warns.append(f"{name}: written before '{reasons[0]}', so the answer cannot "
                         "depend on the reasoning")
    return warns


def main():
    print("1. A forced pick: the 'confidence' field, one of high / medium / low")
    for label, probs in (("clear invoice", CLEAR), ("blurry scan", BLURRY)):
        dist, kept = mask_and_renormalize(probs, ENUM)
        top = max(dist, key=dist.get)
        shares = ", ".join(f"{t} {dist[t]:.2f}" for t in ("high", "medium", "low"))
        print(f"   {label:13s}: kept {kept:.2f} of the model's probability -> {shares}")
        print(f"   {'':13s}  most likely output: \"{top}\", and the JSON looks the same either way")
    with_unknown = mask_and_renormalize(BLURRY, ENUM | {"unknown"})[0]
    print(f"   add 'unknown' to the enum: blurry scan -> unknown {with_unknown['unknown']:.2f}")

    print("\n2. A whitespace loop to max_tokens = 2048")
    for w in (0.9, 0.99, 0.999):
        print(f"   whitespace share {w:5.3f} each step: average run {1 / (1 - w):4.0f} tokens, "
              f"P(all 2048) = {p_run_reaches(w, 2048):.1e}")
    print(f"   a 1% chance of filling 2048 tokens needs a share of {w_needed(2048):.4f} at every "
          "step:\n   the model has to be nearly certain, which is repetition feeding itself, not "
          "leftover probability")

    print("\n3. Lint a schema before you ship it")
    schema = {"type": "object", "properties": {
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "vendor": {"type": "string"},
        "reasoning": {"type": "string"}}}
    for w in check_schema(schema, ["confidence", "vendor"]):
        print("   WARN", w)
    fixed = {"type": "object", "properties": {
        "reasoning": {"type": "string", "maxLength": 600},
        "vendor": {"type": "string", "maxLength": 120},
        "confidence": {"type": "string", "enum": ["high", "medium", "low", "unknown"]}}}
    print(f"   fixed schema: {len(check_schema(fixed, ['confidence', 'vendor']))} warnings")
    print("   " + json.dumps(list(fixed["properties"])))


if __name__ == "__main__":
    main()
