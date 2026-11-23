# Measured results

Hardware: Intel(R) Xeon(R) Processor (4 logical CPUs, 15.63 GiB RAM), Python 3.12.3, device cpu.

Test questions: 104. Pilot questions used only for the evidence gate: 14.
Generator warmup (not included in per-question latency): 0.5 s.

## Retrieval

| method | group | n | Recall@5 | MRR | nDCG@5 | Hit@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 88 | 0.398 | 0.250 | 0.279 | 0.398 |
| bm25 | en | 44 | 0.591 | 0.332 | 0.397 | 0.591 |
| bm25 | ur | 44 | 0.205 | 0.167 | 0.160 | 0.205 |
| bm25 | lexical | 44 | 0.432 | 0.260 | 0.296 | 0.432 |
| bm25 | paraphrase | 44 | 0.364 | 0.239 | 0.262 | 0.364 |
| dense | all | 88 | 0.727 | 0.532 | 0.579 | 0.727 |
| dense | en | 44 | 0.818 | 0.566 | 0.629 | 0.818 |
| dense | ur | 44 | 0.636 | 0.499 | 0.529 | 0.636 |
| dense | lexical | 44 | 0.795 | 0.571 | 0.627 | 0.795 |
| dense | paraphrase | 44 | 0.659 | 0.494 | 0.531 | 0.659 |
| hybrid | all | 88 | 0.659 | 0.436 | 0.488 | 0.659 |
| hybrid | en | 44 | 0.727 | 0.493 | 0.547 | 0.727 |
| hybrid | ur | 44 | 0.591 | 0.379 | 0.429 | 0.591 |
| hybrid | lexical | 44 | 0.727 | 0.478 | 0.539 | 0.727 |
| hybrid | paraphrase | 44 | 0.591 | 0.395 | 0.437 | 0.591 |
| hybrid_rerank | all | 88 | 0.875 | 0.734 | 0.772 | 0.875 |
| hybrid_rerank | en | 44 | 0.932 | 0.816 | 0.846 | 0.932 |
| hybrid_rerank | ur | 44 | 0.818 | 0.652 | 0.697 | 0.818 |
| hybrid_rerank | lexical | 44 | 0.909 | 0.739 | 0.784 | 0.909 |
| hybrid_rerank | paraphrase | 44 | 0.841 | 0.729 | 0.759 | 0.841 |
| hybrid_translate | all | 88 | 0.648 | 0.431 | 0.480 | 0.648 |
| hybrid_translate | en | 44 | 0.727 | 0.493 | 0.547 | 0.727 |
| hybrid_translate | ur | 44 | 0.568 | 0.369 | 0.413 | 0.568 |
| hybrid_translate | lexical | 44 | 0.705 | 0.483 | 0.537 | 0.705 |
| hybrid_translate | paraphrase | 44 | 0.591 | 0.380 | 0.422 | 0.591 |
| hybrid_translate_rerank | all | 88 | 0.795 | 0.643 | 0.676 | 0.795 |
| hybrid_translate_rerank | en | 44 | 0.932 | 0.816 | 0.846 | 0.932 |
| hybrid_translate_rerank | ur | 44 | 0.659 | 0.471 | 0.506 | 0.659 |
| hybrid_translate_rerank | lexical | 44 | 0.795 | 0.590 | 0.642 | 0.795 |
| hybrid_translate_rerank | paraphrase | 44 | 0.795 | 0.697 | 0.709 | 0.795 |

## Answers

| method | group | n | F1 | F1 95% CI | semantic | generated-only F1 | EM | abstention acc. | coverage | unanswerable abstain | citation page precision | raw unsupported rate |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | all | 104 | 0.141 | 0.102–0.185 | 0.518 | 0.141 | 0.000 | 0.625 | 0.602 | 0.750 | 0.283 | 0.000 |
| bm25 | en | 52 | 0.221 | 0.155–0.291 | 0.631 | 0.221 | 0.000 | 0.769 | 0.727 | 1.000 | 0.312 | 0.000 |
| bm25 | ur | 52 | 0.060 | 0.029–0.102 | 0.404 | 0.060 | 0.000 | 0.481 | 0.477 | 0.500 | 0.238 | 0.000 |
| bm25 | english_papers | 96 | 0.146 | 0.101–0.191 | 0.515 | 0.146 | 0.000 | 0.625 | 0.600 | 0.750 | 0.229 | 0.000 |
| bm25 | urdu_doc | 8 | 0.092 | 0.022–0.172 | 0.550 | 0.092 | 0.000 | 0.625 | 0.625 | n/a | 0.800 | 0.000 |
| dense | all | 104 | 0.206 | 0.163–0.252 | 0.839 | 0.202 | 0.000 | 0.875 | 0.977 | 0.312 | 0.419 | 0.008 |
| dense | en | 52 | 0.283 | 0.220–0.348 | 0.827 | 0.277 | 0.000 | 0.904 | 0.955 | 0.625 | 0.500 | 0.000 |
| dense | ur | 52 | 0.130 | 0.077–0.184 | 0.851 | 0.126 | 0.000 | 0.846 | 1.000 | 0.000 | 0.341 | 0.019 |
| dense | english_papers | 96 | 0.209 | 0.162–0.257 | 0.837 | 0.204 | 0.000 | 0.865 | 0.975 | 0.312 | 0.385 | 0.009 |
| dense | urdu_doc | 8 | 0.176 | 0.081–0.291 | 0.854 | 0.176 | 0.000 | 1.000 | 1.000 | n/a | 0.750 | 0.000 |
| hybrid | all | 104 | 0.185 | 0.143–0.227 | 0.850 | 0.174 | 0.000 | 0.846 | 1.000 | 0.000 | 0.352 | 0.008 |
| hybrid | en | 52 | 0.295 | 0.232–0.358 | 0.864 | 0.285 | 0.000 | 0.846 | 1.000 | 0.000 | 0.455 | 0.000 |
| hybrid | ur | 52 | 0.074 | 0.044–0.107 | 0.837 | 0.063 | 0.000 | 0.846 | 1.000 | 0.000 | 0.250 | 0.019 |
| hybrid | english_papers | 96 | 0.189 | 0.144–0.232 | 0.850 | 0.179 | 0.000 | 0.833 | 1.000 | 0.000 | 0.350 | 0.009 |
| hybrid | urdu_doc | 8 | 0.139 | 0.063–0.215 | 0.849 | 0.125 | 0.000 | 1.000 | 1.000 | n/a | 0.375 | 0.000 |
| hybrid_rerank | all | 104 | 0.228 | 0.184–0.272 | 0.808 | 0.221 | 0.000 | 0.933 | 0.932 | 0.938 | 0.646 | 0.012 |
| hybrid_rerank | en | 52 | 0.331 | 0.264–0.392 | 0.838 | 0.327 | 0.000 | 0.962 | 0.955 | 1.000 | 0.762 | 0.013 |
| hybrid_rerank | ur | 52 | 0.125 | 0.079–0.179 | 0.779 | 0.115 | 0.000 | 0.904 | 0.909 | 0.875 | 0.525 | 0.012 |
| hybrid_rerank | english_papers | 96 | 0.229 | 0.183–0.275 | 0.812 | 0.221 | 0.000 | 0.938 | 0.938 | 0.938 | 0.613 | 0.006 |
| hybrid_rerank | urdu_doc | 8 | 0.216 | 0.111–0.337 | 0.777 | 0.216 | 0.000 | 0.875 | 0.875 | n/a | 1.000 | 0.071 |
| hybrid_translate | all | 104 | 0.182 | 0.141–0.225 | 0.830 | 0.176 | 0.000 | 0.827 | 0.977 | 0.000 | 0.384 | 0.026 |
| hybrid_translate | en | 52 | 0.295 | 0.232–0.358 | 0.864 | 0.285 | 0.000 | 0.846 | 1.000 | 0.000 | 0.455 | 0.000 |
| hybrid_translate | ur | 52 | 0.069 | 0.039–0.101 | 0.796 | 0.066 | 0.000 | 0.808 | 0.955 | 0.000 | 0.310 | 0.065 |
| hybrid_translate | english_papers | 96 | 0.187 | 0.139–0.232 | 0.829 | 0.181 | 0.000 | 0.812 | 0.975 | 0.000 | 0.385 | 0.019 |
| hybrid_translate | urdu_doc | 8 | 0.134 | 0.066–0.207 | 0.839 | 0.120 | 0.000 | 1.000 | 1.000 | n/a | 0.375 | 0.083 |
| hybrid_translate_rerank | all | 104 | 0.224 | 0.180–0.266 | 0.823 | 0.220 | 0.000 | 0.837 | 0.955 | 0.188 | 0.631 | 0.015 |
| hybrid_translate_rerank | en | 52 | 0.333 | 0.268–0.395 | 0.875 | 0.329 | 0.000 | 0.904 | 1.000 | 0.375 | 0.727 | 0.013 |
| hybrid_translate_rerank | ur | 52 | 0.114 | 0.077–0.155 | 0.771 | 0.110 | 0.000 | 0.769 | 0.909 | 0.000 | 0.525 | 0.018 |
| hybrid_translate_rerank | english_papers | 96 | 0.224 | 0.179–0.268 | 0.818 | 0.219 | 0.000 | 0.823 | 0.950 | 0.188 | 0.605 | 0.008 |
| hybrid_translate_rerank | urdu_doc | 8 | 0.227 | 0.121–0.344 | 0.870 | 0.227 | 0.000 | 1.000 | 1.000 | n/a | 0.875 | 0.071 |

Raw F1 and EM above are unchanged. The next table strips citations and the bilingual frame, then folds Urdu letters, digits, and punctuation. `F1 either` is the better of that score against the question-language reference and against the English reference. It is not a replacement for raw F1.

| method | group | F1 norm | F1 norm 95% CI | EM norm | F1 either | F1 either 95% CI |
| --- | --- | ---: | --- | ---: | ---: | --- |
| bm25 | all | 0.190 | 0.141–0.248 | 0.034 | 0.190 | 0.141–0.248 |
| bm25 | en | 0.289 | 0.202–0.383 | 0.068 | 0.289 | 0.202–0.383 |
| bm25 | ur | 0.092 | 0.047–0.147 | 0.000 | 0.092 | 0.047–0.147 |
| bm25 | urdu_doc | 0.189 | 0.069–0.325 | 0.000 | 0.189 | 0.069–0.325 |
| dense | all | 0.274 | 0.219–0.331 | 0.023 | 0.287 | 0.233–0.343 |
| dense | en | 0.360 | 0.274–0.447 | 0.045 | 0.360 | 0.274–0.447 |
| dense | ur | 0.188 | 0.117–0.260 | 0.000 | 0.213 | 0.146–0.284 |
| dense | urdu_doc | 0.277 | 0.145–0.448 | 0.000 | 0.277 | 0.145–0.448 |
| hybrid | all | 0.248 | 0.195–0.304 | 0.034 | 0.259 | 0.207–0.314 |
| hybrid | en | 0.378 | 0.295–0.464 | 0.068 | 0.378 | 0.295–0.464 |
| hybrid | ur | 0.119 | 0.073–0.167 | 0.000 | 0.141 | 0.094–0.190 |
| hybrid | urdu_doc | 0.258 | 0.153–0.378 | 0.000 | 0.258 | 0.153–0.378 |
| hybrid_rerank | all | 0.305 | 0.251–0.359 | 0.023 | 0.323 | 0.269–0.376 |
| hybrid_rerank | en | 0.427 | 0.343–0.505 | 0.045 | 0.427 | 0.343–0.505 |
| hybrid_rerank | ur | 0.183 | 0.122–0.255 | 0.000 | 0.219 | 0.158–0.287 |
| hybrid_rerank | urdu_doc | 0.338 | 0.198–0.484 | 0.000 | 0.338 | 0.198–0.484 |
| hybrid_translate | all | 0.235 | 0.179–0.295 | 0.034 | 0.247 | 0.193–0.305 |
| hybrid_translate | en | 0.378 | 0.295–0.464 | 0.068 | 0.378 | 0.295–0.464 |
| hybrid_translate | ur | 0.092 | 0.051–0.137 | 0.000 | 0.117 | 0.076–0.162 |
| hybrid_translate | urdu_doc | 0.178 | 0.100–0.268 | 0.000 | 0.178 | 0.100–0.268 |
| hybrid_translate_rerank | all | 0.296 | 0.241–0.351 | 0.023 | 0.299 | 0.243–0.353 |
| hybrid_translate_rerank | en | 0.430 | 0.347–0.507 | 0.045 | 0.430 | 0.347–0.507 |
| hybrid_translate_rerank | ur | 0.163 | 0.106–0.220 | 0.000 | 0.167 | 0.112–0.224 |
| hybrid_translate_rerank | urdu_doc | 0.322 | 0.177–0.484 | 0.000 | 0.322 | 0.177–0.484 |

## Latency

Retrieve times below are measured. A generate median near 0.1 ms means the draft was read from the generation cache, not that the model ran in a tenth of a millisecond.

| method | retrieve median ms | generate median ms | English generate ms | Urdu generate ms | end-to-end median ms | generations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 1.5 | 9925.8 | 6614.3 | 19746.8 | 9930.5 | 104 |
| dense | 10.6 | 10843.5 | 6402.8 | 15112.6 | 10896.1 | 104 |
| hybrid | 11.4 | 9097.9 | 6314.1 | 16102.5 | 9192.8 | 104 |
| hybrid_rerank | 563.8 | 9854.7 | 6453.1 | 14719.3 | 8119.2 | 84 |
| hybrid_translate | 47.5 | 0.5 | 0.1 | 14683.4 | 103.9 | 104 |
| hybrid_translate_rerank | 619.6 | 10230.9 | 0.1 | 16063.2 | 10990.5 | 104 |

## Reading the numbers

The comparisons below are computed from the tables in this file.
- Highest page Recall@5 on the full answerable test set: **hybrid_rerank** at 0.875.
- Highest page Recall@5 on English answerable questions: **hybrid_rerank, hybrid_translate_rerank (tied)** at 0.932.
- Highest page Recall@5 on Urdu answerable questions: **hybrid_rerank** at 0.818.
- Highest page Recall@5 on lexical questions: **hybrid_rerank** at 0.909.
- Highest page Recall@5 on paraphrased questions: **hybrid_rerank** at 0.841.
- Highest question-language token F1: **hybrid_rerank** at 0.228.
- Highest English token F1: **hybrid_translate_rerank** at 0.333.
- Highest Urdu token F1: **dense** at 0.130.
- Urdu content F1 compares the grounded English answer with the English reference (best **hybrid_rerank** at 0.118). Urdu token F1 compares the final answer, after Urdu translation when the draft was English, with the Urdu reference (best **dense** at 0.130).
- On Urdu questions the dense retriever's page recall is higher than BM25. The papers are in English, so Urdu questions share few exact tokens with the pages, while the multilingual encoder can still rank a page that states the same fact.
- BM25 page Recall@5 on lexical questions is 0.432.
- On paraphrased questions, BM25 page Recall@5 is 0.364 and dense page Recall@5 is 0.659.
- Urdu page Recall@5 was 0.818 for hybrid plus reranking, 0.636 for dense retrieval, and 0.568 for hybrid retrieval with Urdu-to-English query translation. Reranking used the original question. Translation adds an English keyword list, which helps only when the translation keeps the technical terms.

