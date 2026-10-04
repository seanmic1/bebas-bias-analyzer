# Contributing

This repository mirrors the engine in the private Bebas Bias backend. Each
deploy overwrites it, so nothing merged here directly would survive the next
sync. That changes how to contribute, not whether you can.

## Disputing a verdict

Open an issue with:

- the article link from the site (or its id);
- which score you think is wrong, and why;
- the output of `python -m scripts.backtest show <article>`. If you ran
  `python -m scripts.backtest run`, attach the Markdown report too.

Quote only as much of the article as the point needs. The text belongs to its
publisher.

A backtest result helps sort the dispute. A score that the re-scorings don't
reproduce points to scorer noise. A score that reproduces consistently points
to the rubric or the prompt, which is the more interesting case.

## Changing the rubric, the prompt, or the code

Pull requests are welcome as proposals. If a change is accepted, the
maintainer ports it to the backend, where it's tested and deployed alongside
the rest of the system. It comes back here with the next sync, credited to you
with a `Co-authored-by` trailer, and the pull request is then closed.

Rubric changes are judged on evidence. Use backtests or audits that show the
current wording misreads a class of Indonesian reporting, and show what your
wording does on the same articles.

## Security

Report security problems privately through GitHub's "Report a vulnerability"
button on this repository, not in a public issue.
