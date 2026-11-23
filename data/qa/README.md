# Question set

`gold.json` is the reported test set: 32 questions (12 English answerable, 12 Urdu translations of those questions, 4 English unanswerable, 4 Urdu translations of the unanswerable questions).

`pilot.json` has 8 disjoint questions and is used only to choose the evidence-gate threshold. Pilot questions are not scored in the tables.

The three source papers are listed in `data/papers/catalog.json`. Each PDF states a CC BY 4.0 licence. Page numbers are the PDF page numbers, starting at 1. Every answerable item has an `evidence_quotes` field, and the unit tests check that each quotation occurs on a cited page of the extracted text.

Urdu items are translations of the English items, with the same gold pages. There was no second annotator. The set is a project diagnostic, not a community benchmark.
