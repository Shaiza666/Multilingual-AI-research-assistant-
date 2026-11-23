# Measured results

Hardware: Intel(R) Xeon(R) Processor (4 logical CPUs, 15.63 GiB RAM), Python 3.12.3, device cpu.

Test questions: 32. Pilot questions used only for the evidence gate: 8.
Generator warmup (not included in per-question latency): 0.4 s.

## Retrieval

| method | group | n | Recall@5 | MRR | nDCG@5 | Hit@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 24 | 0.500 | 0.240 | 0.324 | 0.500 |
| bm25 | en | 12 | 0.833 | 0.431 | 0.555 | 0.833 |
| bm25 | ur | 12 | 0.167 | 0.049 | 0.094 | 0.167 |
| bm25 | lexical | 12 | 0.500 | 0.160 | 0.283 | 0.500 |
| bm25 | paraphrase | 12 | 0.500 | 0.319 | 0.366 | 0.500 |
| dense | all | 24 | 0.750 | 0.416 | 0.506 | 0.750 |
| dense | en | 12 | 0.833 | 0.621 | 0.658 | 0.833 |
| dense | ur | 12 | 0.667 | 0.212 | 0.355 | 0.667 |
| dense | lexical | 12 | 0.667 | 0.298 | 0.391 | 0.667 |
| dense | paraphrase | 12 | 0.833 | 0.535 | 0.621 | 0.833 |
| hybrid | all | 24 | 0.750 | 0.401 | 0.506 | 0.750 |
| hybrid | en | 12 | 0.917 | 0.597 | 0.699 | 0.917 |
| hybrid | ur | 12 | 0.583 | 0.206 | 0.313 | 0.583 |
| hybrid | lexical | 12 | 0.667 | 0.268 | 0.391 | 0.667 |
| hybrid | paraphrase | 12 | 0.833 | 0.535 | 0.621 | 0.833 |
| hybrid_rerank | all | 24 | 0.917 | 0.724 | 0.767 | 0.917 |
| hybrid_rerank | en | 12 | 0.917 | 0.786 | 0.813 | 0.917 |
| hybrid_rerank | ur | 12 | 0.917 | 0.661 | 0.721 | 0.917 |
| hybrid_rerank | lexical | 12 | 0.833 | 0.614 | 0.658 | 0.833 |
| hybrid_rerank | paraphrase | 12 | 1.000 | 0.833 | 0.877 | 1.000 |
| hybrid_translate | all | 24 | 0.708 | 0.497 | 0.561 | 0.708 |
| hybrid_translate | en | 12 | 0.917 | 0.597 | 0.699 | 0.917 |
| hybrid_translate | ur | 12 | 0.500 | 0.396 | 0.422 | 0.500 |
| hybrid_translate | lexical | 12 | 0.667 | 0.318 | 0.433 | 0.667 |
| hybrid_translate | paraphrase | 12 | 0.750 | 0.676 | 0.688 | 0.750 |
| hybrid_translate_rerank | all | 24 | 0.833 | 0.679 | 0.720 | 0.833 |
| hybrid_translate_rerank | en | 12 | 0.917 | 0.786 | 0.813 | 0.917 |
| hybrid_translate_rerank | ur | 12 | 0.750 | 0.572 | 0.627 | 0.750 |
| hybrid_translate_rerank | lexical | 12 | 0.750 | 0.566 | 0.616 | 0.750 |
| hybrid_translate_rerank | paraphrase | 12 | 0.917 | 0.792 | 0.824 | 0.917 |

## Answers

| method | group | n | F1 | content F1 | EM | abstention acc. | coverage | unanswerable abstain | citation page precision | raw unsupported rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 32 | 0.204 | 0.202 | 0.000 | 0.625 | 0.542 | 0.875 | 0.590 | 0.038 |
| bm25 | en | 16 | 0.400 | 0.400 | 0.000 | 0.938 | 1.000 | 0.750 | 0.639 | 0.042 |
| bm25 | ur | 16 | 0.007 | 0.003 | 0.000 | 0.312 | 0.083 | 1.000 | 0.000 | 0.000 |
| dense | all | 32 | 0.231 | 0.192 | 0.000 | 0.875 | 0.833 | 1.000 | 0.371 | 0.025 |
| dense | en | 16 | 0.376 | 0.376 | 0.000 | 1.000 | 1.000 | 1.000 | 0.588 | 0.000 |
| dense | ur | 16 | 0.086 | 0.007 | 0.000 | 0.750 | 0.667 | 1.000 | 0.000 | 0.062 |
| hybrid | all | 32 | 0.261 | 0.222 | 0.000 | 0.875 | 0.833 | 1.000 | 0.289 | 0.042 |
| hybrid | en | 16 | 0.437 | 0.437 | 0.000 | 1.000 | 1.000 | 1.000 | 0.458 | 0.028 |
| hybrid | ur | 16 | 0.086 | 0.007 | 0.000 | 0.750 | 0.667 | 1.000 | 0.000 | 0.062 |
| hybrid_rerank | all | 32 | 0.184 | 0.143 | 0.000 | 0.719 | 0.625 | 1.000 | 0.144 | 0.078 |
| hybrid_rerank | en | 16 | 0.282 | 0.282 | 0.000 | 0.812 | 0.750 | 1.000 | 0.241 | 0.056 |
| hybrid_rerank | ur | 16 | 0.086 | 0.003 | 0.000 | 0.625 | 0.500 | 1.000 | 0.000 | 0.111 |
| hybrid_translate | all | 32 | 0.234 | 0.193 | 0.000 | 0.750 | 0.750 | 0.750 | 0.306 | 0.037 |
| hybrid_translate | en | 16 | 0.383 | 0.383 | 0.000 | 0.938 | 0.917 | 1.000 | 0.500 | 0.030 |
| hybrid_translate | ur | 16 | 0.086 | 0.003 | 0.000 | 0.562 | 0.583 | 0.500 | 0.000 | 0.048 |
| hybrid_translate_rerank | all | 32 | 0.200 | 0.179 | 0.000 | 0.719 | 0.708 | 0.750 | 0.216 | 0.039 |
| hybrid_translate_rerank | en | 16 | 0.347 | 0.347 | 0.000 | 0.938 | 0.917 | 1.000 | 0.333 | 0.045 |
| hybrid_translate_rerank | ur | 16 | 0.053 | 0.011 | 0.000 | 0.500 | 0.500 | 0.500 | 0.000 | 0.028 |

## Latency

| method | retrieve median ms | generate median ms | end-to-end median ms | generations |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.8 | 4517.5 | 3634.9 | 20 |
| dense | 9.6 | 4602.5 | 4634.4 | 31 |
| hybrid | 9.7 | 3816.1 | 3844.6 | 32 |
| hybrid_rerank | 529.9 | 4304.6 | 4648.9 | 22 |
| hybrid_translate | 45.2 | 4573.4 | 2318.1 | 28 |
| hybrid_translate_rerank | 573.2 | 4525.6 | 5117.0 | 32 |

## Reading the numbers

The comparisons below are computed from the tables in this file.
- Highest page Recall@5 on the full answerable test set: **hybrid_rerank** at 0.917.
- Highest page Recall@5 on English answerable questions: **hybrid, hybrid_rerank, hybrid_translate, hybrid_translate_rerank (tied)** at 0.917.
- Highest page Recall@5 on Urdu answerable questions: **hybrid_rerank** at 0.917.
- Highest page Recall@5 on lexical questions: **hybrid_rerank** at 0.833.
- Highest page Recall@5 on paraphrased questions: **hybrid_rerank** at 1.000.
- Highest question-language token F1: **hybrid** at 0.261.
- Highest English token F1: **hybrid** at 0.437.
- Highest Urdu token F1: **dense, hybrid (tied)** at 0.086.
- Urdu content F1 compares the grounded English answer with the English reference (best **hybrid_translate_rerank** at 0.011). Urdu token F1 compares the final answer, after Urdu translation when the draft was English, with the Urdu reference (best **dense, hybrid (tied)** at 0.086).
- On Urdu questions the dense retriever's page recall is higher than BM25. The papers are in English, so Urdu questions share few exact tokens with the pages, while the multilingual encoder can still rank a page that states the same fact.
- BM25 page Recall@5 on lexical questions is 0.500.
- On paraphrased questions, BM25 page Recall@5 is 0.500 and dense page Recall@5 is 0.833.
- Urdu page Recall@5 was 0.917 for hybrid plus reranking, 0.667 for dense retrieval, and 0.500 for hybrid retrieval with Urdu-to-English query translation. Reranking used the original question. Translation adds an English keyword list, which helps only when the translation keeps the technical terms.

