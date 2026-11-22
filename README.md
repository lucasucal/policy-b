# policy-b

A small vision-conditioned joint-space controller: a frozen CLIP image encoder plus a
tiny learned MLP head. Weights for the head live in a separate Hugging Face model repo.

Run: `python serve.py` (HTTP policy server).
