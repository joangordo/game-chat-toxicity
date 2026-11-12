"""Mask offensive words before showing examples in reports.

Uses the toxicity lexicon released with CONDA plus a short list of stems.
Masking is automatic and imperfect: always review examples before publishing.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config

SEED_STEMS = ["fuck", "fck", "fuk", "shit", "bitch", "b1tch", "cunt", "nigg", "n1gg", "fag", "retard",
              "whore", "slut", "rape", "dick", "cock", "pussy", "bastard", "asshole", "idiot", "moron",
              "suka", "blyat", "blya", "pidor", "huj", "hui", "kurwa", "scheiss", "hure", "nazi", "hitler",
              "jew", "gay", "autist", "cancer", "kys", "stfu",
              # Polish, Russian (transliterated), German, Spanish
              "dupa", "dupe", "cwel", "chuj", "kurw", "pierd", "jeb", "pizd", "eban", "nahui", "cyka",
              "hurensohn", "fotze", "wichser", "arsch", "schlampe", "puta", "mierda", "pendej", "cabron"]
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_WORD = re.compile(r"[\w@$]+", re.UNICODE)


def load_lexicon() -> set[str]:
    p = config.RAW_CONDA / "lexicon.csv"
    if not p.exists():
        return set()
    col = pd.read_csv(p)["T"].dropna().astype(str).str.lower()
    return {w for w in col if len(w) >= 3}


def _canonical(word: str) -> str:
    w = word.lower().translate(_LEET)
    return re.sub(r"(.)\1{2,}", r"\1\1", w)  # "fuuuuck" -> "fuuck"


def is_offensive(word: str, lexicon: set[str]) -> bool:
    w = _canonical(word)
    collapsed = re.sub(r"(.)\1+", r"\1", w)
    return w in lexicon or collapsed in lexicon or any(s in w or s in collapsed for s in SEED_STEMS)


def mask(text: str, lexicon: set[str]) -> str:
    def repl(m):
        w = m.group(0)
        return w[0] + "*" * (len(w) - 1) if len(w) > 1 and is_offensive(w, lexicon) else w
    return _WORD.sub(repl, str(text))
