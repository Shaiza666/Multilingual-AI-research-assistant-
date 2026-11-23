# Question set

`gold.json` is the reported test set. It contains English questions, Urdu translations of those questions, and unanswerable questions in both languages. The exact counts are in the file's `counts` object.

`pilot.json` is disjoint. It is used only to set the per-method evidence-gate threshold and is not included in the reported scores.

Nine documents are research papers. Three state CC BY 4.0 on the PDF. Six are ACL Anthology papers from 2019–2024; the Anthology licenses materials published in or after 2016 under CC BY 4.0. One further document is the Urdu Wikipedia article مصنوعی ذہانت, revision 11740367, under CC BY-SA 4.0. Questions with `corpus` set to `urdu_doc` ask about that article, including English questions over Urdu text.

Urdu items are translations of the English items. There was no second annotator. Each answerable item has an `evidence_quotes` field, and the tests check that the quotation occurs on a cited page.
