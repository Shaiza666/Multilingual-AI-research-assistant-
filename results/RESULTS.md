# Measured results

Hardware: Intel(R) Xeon(R) Processor (4 logical CPUs, 15.63 GiB RAM), Python 3.12.3, device cpu.

Test questions: 104. Pilot questions used only for the evidence gate: 14.
Generator warmup (not included in per-question latency): 0.7 s.

## Retrieval

| method | group | n | Recall@5 | MRR | nDCG@5 | Hit@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 88 | 0.398 | 0.244 | 0.275 | 0.398 |
| bm25 | en | 44 | 0.591 | 0.332 | 0.397 | 0.591 |
| bm25 | ur | 44 | 0.205 | 0.156 | 0.152 | 0.205 |
| bm25 | lexical | 44 | 0.432 | 0.249 | 0.287 | 0.432 |
| bm25 | paraphrase | 44 | 0.364 | 0.239 | 0.262 | 0.364 |
| dense | all | 88 | 0.727 | 0.527 | 0.575 | 0.727 |
| dense | en | 44 | 0.818 | 0.566 | 0.629 | 0.818 |
| dense | ur | 44 | 0.636 | 0.487 | 0.521 | 0.636 |
| dense | lexical | 44 | 0.795 | 0.559 | 0.619 | 0.795 |
| dense | paraphrase | 44 | 0.659 | 0.494 | 0.531 | 0.659 |
| hybrid | all | 88 | 0.648 | 0.434 | 0.481 | 0.648 |
| hybrid | en | 44 | 0.727 | 0.493 | 0.547 | 0.727 |
| hybrid | ur | 44 | 0.568 | 0.375 | 0.416 | 0.568 |
| hybrid | lexical | 44 | 0.705 | 0.478 | 0.529 | 0.705 |
| hybrid | paraphrase | 44 | 0.591 | 0.391 | 0.434 | 0.591 |
| hybrid_rerank | all | 88 | 0.875 | 0.734 | 0.772 | 0.875 |
| hybrid_rerank | en | 44 | 0.932 | 0.816 | 0.846 | 0.932 |
| hybrid_rerank | ur | 44 | 0.818 | 0.652 | 0.697 | 0.818 |
| hybrid_rerank | lexical | 44 | 0.909 | 0.739 | 0.784 | 0.909 |
| hybrid_rerank | paraphrase | 44 | 0.841 | 0.729 | 0.759 | 0.841 |
| hybrid_translate | all | 88 | 0.648 | 0.431 | 0.480 | 0.648 |
| hybrid_translate | en | 44 | 0.727 | 0.493 | 0.547 | 0.727 |
| hybrid_translate | ur | 44 | 0.568 | 0.369 | 0.413 | 0.568 |
| hybrid_translate | lexical | 44 | 0.705 | 0.483 | 0.537 | 0.705 |
| hybrid_translate | paraphrase | 44 | 0.591 | 0.379 | 0.422 | 0.591 |
| hybrid_translate_rerank | all | 88 | 0.784 | 0.641 | 0.670 | 0.784 |
| hybrid_translate_rerank | en | 44 | 0.932 | 0.816 | 0.846 | 0.932 |
| hybrid_translate_rerank | ur | 44 | 0.636 | 0.467 | 0.493 | 0.636 |
| hybrid_translate_rerank | lexical | 44 | 0.773 | 0.586 | 0.630 | 0.773 |
| hybrid_translate_rerank | paraphrase | 44 | 0.795 | 0.697 | 0.709 | 0.795 |

## Answers

| method | group | n | F1 | F1 95% CI | semantic | generated-only F1 | EM | abstention acc. | coverage | unanswerable abstain | citation page precision | raw unsupported rate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 104 | 0.110 | 0.072–0.154 | 0.408 | 0.110 | 0.000 | 0.500 | 0.477 | 0.625 | 0.329 | 0.036 |
| bm25 | en | 52 | 0.198 | 0.130–0.276 | 0.500 | 0.198 | 0.000 | 0.635 | 0.568 | 1.000 | 0.392 | 0.020 |
| bm25 | ur | 52 | 0.022 | 0.012–0.033 | 0.316 | 0.022 | 0.000 | 0.365 | 0.386 | 0.250 | 0.235 | 0.059 |
| bm25 | english_papers | 96 | 0.117 | 0.074–0.161 | 0.407 | 0.117 | 0.000 | 0.500 | 0.475 | 0.625 | 0.284 | 0.013 |
| bm25 | urdu_doc | 8 | 0.041 | 0.015–0.070 | 0.423 | 0.041 | 0.000 | 0.500 | 0.500 | n/a | 0.750 | 0.250 |
| dense | all | 104 | 0.201 | 0.157–0.246 | 0.838 | 0.183 | 0.000 | 0.875 | 0.977 | 0.312 | 0.471 | 0.042 |
| dense | en | 52 | 0.319 | 0.252–0.381 | 0.856 | 0.298 | 0.000 | 0.923 | 0.977 | 0.625 | 0.547 | 0.015 |
| dense | ur | 52 | 0.084 | 0.052–0.130 | 0.820 | 0.067 | 0.000 | 0.827 | 0.977 | 0.000 | 0.395 | 0.083 |
| dense | english_papers | 96 | 0.209 | 0.163–0.256 | 0.837 | 0.188 | 0.000 | 0.865 | 0.975 | 0.312 | 0.442 | 0.027 |
| dense | urdu_doc | 8 | 0.123 | 0.056–0.204 | 0.843 | 0.123 | 0.000 | 1.000 | 1.000 | n/a | 0.750 | 0.143 |
| hybrid | all | 104 | 0.189 | 0.148–0.233 | 0.839 | 0.171 | 0.000 | 0.837 | 0.989 | 0.000 | 0.385 | 0.027 |
| hybrid | en | 52 | 0.305 | 0.240–0.371 | 0.870 | 0.279 | 0.000 | 0.846 | 1.000 | 0.000 | 0.500 | 0.015 |
| hybrid | ur | 52 | 0.072 | 0.043–0.111 | 0.809 | 0.064 | 0.000 | 0.827 | 0.977 | 0.000 | 0.267 | 0.045 |
| hybrid | english_papers | 96 | 0.196 | 0.149–0.241 | 0.841 | 0.182 | 0.000 | 0.823 | 0.988 | 0.000 | 0.380 | 0.020 |
| hybrid | urdu_doc | 8 | 0.112 | 0.058–0.167 | 0.824 | 0.069 | 0.000 | 1.000 | 1.000 | n/a | 0.438 | 0.125 |
| hybrid_rerank | all | 104 | 0.217 | 0.174–0.264 | 0.817 | 0.194 | 0.000 | 0.942 | 0.943 | 0.938 | 0.669 | 0.019 |
| hybrid_rerank | en | 52 | 0.360 | 0.295–0.424 | 0.848 | 0.339 | 0.000 | 0.962 | 0.955 | 1.000 | 0.750 | 0.014 |
| hybrid_rerank | ur | 52 | 0.074 | 0.049–0.099 | 0.787 | 0.049 | 0.000 | 0.923 | 0.932 | 0.875 | 0.585 | 0.029 |
| hybrid_rerank | english_papers | 96 | 0.228 | 0.181–0.277 | 0.825 | 0.203 | 0.000 | 0.948 | 0.950 | 0.938 | 0.638 | 0.011 |
| hybrid_rerank | urdu_doc | 8 | 0.105 | 0.051–0.160 | 0.741 | 0.105 | 0.000 | 0.875 | 0.875 | n/a | 1.000 | 0.071 |
| hybrid_translate | all | 104 | 0.180 | 0.139–0.226 | 0.839 | 0.163 | 0.000 | 0.837 | 0.989 | 0.000 | 0.397 | 0.013 |
| hybrid_translate | en | 52 | 0.305 | 0.240–0.371 | 0.870 | 0.279 | 0.000 | 0.846 | 1.000 | 0.000 | 0.500 | 0.015 |
| hybrid_translate | ur | 52 | 0.056 | 0.030–0.088 | 0.808 | 0.048 | 0.000 | 0.827 | 0.977 | 0.000 | 0.291 | 0.010 |
| hybrid_translate | english_papers | 96 | 0.186 | 0.139–0.233 | 0.840 | 0.172 | 0.000 | 0.823 | 0.988 | 0.000 | 0.392 | 0.004 |
| hybrid_translate | urdu_doc | 8 | 0.120 | 0.069–0.171 | 0.830 | 0.077 | 0.000 | 1.000 | 1.000 | n/a | 0.438 | 0.100 |
| hybrid_translate_rerank | all | 104 | 0.205 | 0.158–0.254 | 0.799 | 0.187 | 0.000 | 0.808 | 0.932 | 0.125 | 0.571 | 0.027 |
| hybrid_translate_rerank | en | 52 | 0.362 | 0.298–0.426 | 0.885 | 0.341 | 0.000 | 0.885 | 1.000 | 0.250 | 0.716 | 0.013 |
| hybrid_translate_rerank | ur | 52 | 0.048 | 0.026–0.074 | 0.713 | 0.033 | 0.000 | 0.731 | 0.864 | 0.000 | 0.404 | 0.059 |
| hybrid_translate_rerank | english_papers | 96 | 0.213 | 0.165–0.261 | 0.794 | 0.193 | 0.000 | 0.792 | 0.925 | 0.125 | 0.538 | 0.010 |
| hybrid_translate_rerank | urdu_doc | 8 | 0.127 | 0.046–0.212 | 0.849 | 0.127 | 0.000 | 1.000 | 1.000 | n/a | 0.875 | 0.143 |

## Latency

These rows are from the scoring pass that reused cached drafts. Retrieve medians are real. Generate medians near 0.1 ms are cache hits, not model speed. A cold sample is in the README.

| method | retrieve median ms | cached generate median ms | end-to-end median ms | generations |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.9 | 0.1 | 2.0 | 104 |
| dense | 8.3 | 0.1 | 19.8 | 104 |
| hybrid | 9.8 | 0.1 | 20.7 | 104 |
| hybrid_rerank | 522.9 | 0.1 | 559.5 | 84 |
| hybrid_translate | 44.8 | 0.1 | 66.7 | 104 |
| hybrid_translate_rerank | 587.2 | 0.1 | 609.0 | 104 |

## Model comparison

Retrieval is shared. Each profile is a different local GGUF model. F1 is the question-language token F1 of the primary hybrid method when that row exists, otherwise the best method for that model.

| profile | model | best method by F1 | F1 | English F1 | Urdu F1 | semantic | cached generate median ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| fast | qwen2.5-0.5b-instruct-q4_k_m.gguf | hybrid_rerank | 0.159 | 0.279 | 0.039 | 0.806 | 0.1 |
| quality | qwen2.5-1.5b-instruct-q4_k_m.gguf | hybrid_rerank | 0.217 | 0.360 | 0.074 | 0.817 | 0.1 |
| large | qwen2.5-3b-instruct-q4_k_m.gguf | hybrid_rerank | 0.203 | 0.292 | 0.113 | 0.814 | 0.1 |

The generate column above is a cache hit. Cold generate medians on three uncached hybrid-plus-reranking questions were 3.69 s (0.5B), 5.92 s (1.5B), and 11.0 s (3B). One of the three 1.5B calls took 77.9 s while another process was loading models.

## Reading the numbers

The comparisons below are computed from the tables in this file.
- Highest page Recall@5 on the full answerable test set: **hybrid_rerank** at 0.875.
- Highest page Recall@5 on English answerable questions: **hybrid_rerank, hybrid_translate_rerank (tied)** at 0.932.
- Highest page Recall@5 on Urdu answerable questions: **hybrid_rerank** at 0.818.
- Highest page Recall@5 on lexical questions: **hybrid_rerank** at 0.909.
- Highest page Recall@5 on paraphrased questions: **hybrid_rerank** at 0.841.
- Highest question-language token F1: **hybrid_rerank** at 0.217.
- Highest English token F1: **hybrid_translate_rerank** at 0.362.
- Highest Urdu token F1: **dense** at 0.084.
- Urdu content F1 compares the grounded English answer with the English reference (best **hybrid_rerank** at 0.110). Urdu token F1 compares the final answer, after Urdu translation when the draft was English, with the Urdu reference (best **dense** at 0.084).
- On Urdu questions the dense retriever's page recall is higher than BM25. The papers are in English, so Urdu questions share few exact tokens with the pages, while the multilingual encoder can still rank a page that states the same fact.
- BM25 page Recall@5 on lexical questions is 0.432.
- On paraphrased questions, BM25 page Recall@5 is 0.364 and dense page Recall@5 is 0.659.
- Urdu page Recall@5 was 0.818 for hybrid plus reranking, 0.636 for dense retrieval, and 0.568 for hybrid retrieval with Urdu-to-English query translation. Reranking used the original question. Translation adds an English keyword list, which helps only when the translation keeps the technical terms.

