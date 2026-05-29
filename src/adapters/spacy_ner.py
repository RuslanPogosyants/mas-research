"""Real spaCy implementation of NerAdapter (used for the live demo only).

`spacy` + the `ru_core_news_lg` model are optional ML dependencies imported
lazily, so the module imports cleanly in CI where they are absent. The blocking
spaCy pipeline runs in a worker thread to avoid stalling the asyncio event loop.
NER entities and adjective+noun bigrams become term candidates.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any

from src.adapters.ner import TermCandidate


class SpacyNerAdapter:
    """NerAdapter backed by a spaCy Russian pipeline."""

    def __init__(self, model: str = "ru_core_news_lg") -> None:
        import spacy  # lazy: optional ml dependency

        self._nlp = spacy.load(model)

    async def extract(self, text: str) -> list[TermCandidate]:
        doc = await asyncio.to_thread(self._nlp, text)
        candidates: list[TermCandidate] = [
            TermCandidate(text=ent.text, lemma=ent.lemma_.lower(), label=ent.label_) for ent in doc.ents
        ]
        candidates.extend(self._adj_noun_bigrams(doc))
        return candidates

    @staticmethod
    def _adj_noun_bigrams(doc: Any) -> list[TermCandidate]:
        tokens: list[Any] = list(doc)
        bigrams: list[TermCandidate] = []
        for left, right in itertools.pairwise(tokens):
            if getattr(left, "pos_", "") == "ADJ" and getattr(right, "pos_", "") == "NOUN":
                surface = f"{left.text} {right.text}"
                bigrams.append(TermCandidate(text=surface, lemma=f"{left.lemma_.lower()} {right.lemma_.lower()}"))
        return bigrams
