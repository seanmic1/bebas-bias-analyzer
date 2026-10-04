"""
Counterfactual bias audit for the Bebas Bias scorer.

Motivated by IndoBias (arXiv:2606.01260) — a benchmark showing that decoder
LLMs carry strong stereotype bias when prompted in Indonesian. Our scorer *is*
such a model, and it reads articles saturated with ethnicity, religion, party
and institution mentions. This package answers the question that actually
matters for the product: **does the demographic group named in an article move
its bias scores, holding everything else constant?**

IndoBias' own Pairs track compares perplexity of prototypical vs
counter-stereotypical sentences. That is not computable over the OpenAI Chat
Completions API (no `echo`, logprobs cover generated tokens only), so we
transplant the contrastive *logic* onto the production task instead: take a
real article, swap the group it names for a same-category counterpart, re-score
it through the unmodified pipeline, and measure the delta on each axis.

A fair scorer yields a delta indistinguishable from its own sampling noise.
Nothing here imports into the serving path — the audit is offline and
read-only.
"""
