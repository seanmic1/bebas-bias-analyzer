"""Download the nltk resources used by app.nlp. Run once after `pip install`.

Pinned to ``$NLTK_DATA`` when set: the Docker build sets it so the data lands
in a path the non-root runtime user can also read (``nltk.download`` would
otherwise write to ``/root/nltk_data`` during ``docker build``). Without it,
the data goes to ``~/nltk_data``, which nltk searches by default.
"""

import os

import nltk

target = os.environ.get("NLTK_DATA") or os.path.expanduser("~/nltk_data")
os.makedirs(target, exist_ok=True)

for pkg in ("punkt", "punkt_tab", "vader_lexicon"):
    nltk.download(pkg, download_dir=target)
