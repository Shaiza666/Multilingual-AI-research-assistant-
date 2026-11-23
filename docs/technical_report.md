# Technical report: retrieval and grounded answers in English and Urdu

Shaiza Asghar

## 1. Problem

A reader with a folder of papers should be able to ask a question in English or Urdu and receive an answer that names the page it came from. A small local model will invent details if it is asked to answer from memory. The system therefore retrieves passages first, drafts from those passages only, and deletes any sentence that the retrieved text does not support.

The project compares six retrieval methods on one fixed question set and records CPU timings. No score in the results section was filled in by hand.

## 2. Corpus and questions

The corpus is nine research papers and one Urdu encyclopedia article. Three papers state CC BY 4.0 on the PDF: arXiv:2103.14460v1, arXiv:2103.14455v1, and arXiv:2106.16239v9. Six are ACL Anthology papers from 2019–2024 (N19-1423, D19-1410, 2020.acl-main.703, 2020.emnlp-main.550, 2023.acl-short.24, 2024.acl-short.8). The Anthology licenses material published in or after 2016 under CC BY 4.0. The Urdu document is Wikipedia article مصنوعی ذہانت, revision 11740367, under CC BY-SA 4.0.

PyMuPDF reads each page. Running headers that repeat on at least 60% of pages are removed, line-break hyphenation of lowercase letters is joined, and the text is normalised with Unicode NFKC. Chunks are windows of 160 words with a 40-word overlap, and a chunk never crosses a page boundary. The benchmark index contains 658 chunks.

`data/qa/gold.json` has 104 test questions: 44 English answerable, 44 Urdu translations, and 8 unanswerable questions in each language. Eight of those questions are about the Urdu article. `data/qa/pilot.json` has 14 disjoint questions and is used only to set evidence-gate thresholds. There was no second annotator. Unit tests check that every evidence quotation occurs on a cited page.

## 3. Retrieval

Queries and chunks are tokenised after Urdu folding: Arabic kaf and yeh map to the Urdu letters, hamza-bearing alef maps to alef, Arabic heh maps to heh goal, and both Eastern Arabic digit blocks map to ASCII. Bari yeh (ے) is not folded, because it distinguishes words such as کے and کی.

BM25L (Lv and Zhai) provides the keyword ranking. Dense retrieval encodes passages with the prefix `passage: ` and queries with `query: `, as `intfloat/multilingual-e5-small` requires, L2-normalises the vectors, and searches them with a FAISS `IndexFlatIP` index. Hybrid retrieval fuses the two ranked lists with reciprocal rank fusion at k = 60. For an Urdu question, the translation variant also translates the question with `Helsinki-NLP/opus-mt-ur-en` and fuses the BM25L and dense lists of that English query. Reranking scores the top 20 fused chunks with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

Recall@k, MRR, and nDCG@k are defined in `src/mira/metrics.py` and covered by unit tests with hand-computed expected values. They are macro-averaged over the 24 answerable test questions. Unanswerable questions have no gold pages and are excluded from retrieval averages.

## 4. Generation and grounding

Three local GGUF profiles are compared: Qwen2.5 0.5B (`fast`), 1.5B (`quality`, the default), and 3B (`large`), all Q4_K_M, via llama.cpp with a 3072-token context, 4 threads, temperature 0, and at most 160 new tokens. The prompt asks for the fact in the question's language and forbids repeating the question. `MIRA_LLM_BACKEND=transformers` or `openai` selects the other backends.

A sentence is kept when its content-word overlap with a retrieved chunk is at least 0.35 or its E5 cosine is at least 0.82, unless the sentence mostly repeats the question. Among supporting chunks, the citation is the one the retriever ranked first. If no sentence survives, the system may show the retrieved sentence most similar to the question, provided that sentence mentions any long Latin token from the question. Urdu answers that do not preserve those names and numbers are replaced by a bilingual line that keeps the English source sentence.

The evidence gate is fit per method on the pilot. It is enabled only when every finite answerable score is above every finite unanswerable score, and the threshold is the midpoint of that gap. On this pilot only the reranker separated, at threshold −3.24.

## 5. Results

The machine is an Intel Xeon with 4 logical CPUs at 2.4 GHz and 15.63 GiB RAM, Python 3.12.3, PyTorch 2.15.0+cpu. No question raised an exception. An earlier run on 3 papers and 32 questions, with Qwen2.5 0.5B and the old gate, had best token F1 0.261 and, for hybrid plus reranking, Recall@5 0.917, F1 0.184, coverage 0.625, and citation page precision 0.144. The tables below are the current 104-question run. The sets differ, so this is not a paired delta.

### Retrieval, 88 answerable questions

| method | Recall@5 | 95% CI | English | Urdu | MRR | nDCG@5 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| bm25 | 0.398 | 0.307–0.511 | 0.591 | 0.205 | 0.244 | 0.275 |
| dense | 0.727 | 0.636–0.818 | 0.818 | 0.636 | 0.527 | 0.575 |
| hybrid | 0.648 | 0.545–0.750 | 0.727 | 0.568 | 0.434 | 0.481 |
| hybrid_rerank | 0.875 | 0.807–0.943 | 0.932 | 0.818 | 0.734 | 0.772 |
| hybrid_translate | 0.648 | 0.557–0.750 | 0.727 | 0.568 | 0.431 | 0.480 |
| hybrid_translate_rerank | 0.784 | 0.693–0.864 | 0.932 | 0.636 | 0.641 | 0.670 |

Hybrid plus reranking is the best page finder. Urdu-to-English translation does not improve Urdu Recall@5 (0.568 versus 0.818). On the 8 Urdu-article questions, that reranked method has Recall@5 0.875 and BM25L has 0.500.

### Answers, Qwen2.5 1.5B

Exact match is 0.000 throughout. Semantic score is multilingual-e5 cosine, with abstention scored 0.

| method | F1 | 95% CI | English | Urdu | semantic | citation page precision | abstention accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.110 | 0.072–0.154 | 0.198 | 0.022 | 0.408 | 0.329 | 0.500 |
| dense | 0.201 | 0.157–0.246 | 0.319 | 0.084 | 0.838 | 0.471 | 0.875 |
| hybrid | 0.189 | 0.148–0.233 | 0.305 | 0.072 | 0.839 | 0.385 | 0.837 |
| hybrid_rerank | 0.217 | 0.174–0.264 | 0.360 | 0.074 | 0.817 | 0.669 | 0.942 |
| hybrid_translate | 0.180 | 0.139–0.226 | 0.305 | 0.056 | 0.839 | 0.397 | 0.837 |
| hybrid_translate_rerank | 0.205 | 0.158–0.254 | 0.362 | 0.048 | 0.799 | 0.571 | 0.808 |

Hybrid plus reranking is the best end-to-end method on this set: highest F1, highest abstention accuracy, and citation page precision 0.669. Its pilot gate abstains on 0.938 of unanswerable questions while keeping answerable coverage at 0.943. Generated-only F1 is 0.194, so most of the score comes from the draft rather than the retrieved-sentence fallback. Urdu token F1 remains 0.074, while the Urdu semantic score is 0.787, because many Urdu answers keep an English source sentence when the translation drops names.

### Same retriever, three model sizes

Hybrid plus reranking. Answer scores are the full test set. `results/latency.csv` records generate medians near 0.1 ms because that scoring pass reused cached drafts. Retrieval medians there are BM25L 0.9 ms, dense 8.3 ms, hybrid 9.8 ms, and hybrid plus reranking 523 ms.

A later cold sample, with the cache off, asked three questions that reached the model. Medians of those three generate times: 0.5B 3.69 s (4.65, 3.36, 3.69), 1.5B 5.92 s (5.92, 4.87, 77.9), 3B 11.0 s (11.4, 9.31, 11.0). The 77.9 s call is one Urdu question while the application process was loading other models. The first 3B warmup prompt took 84.1 s and is the file load, not a per-question time.

| model | F1 | English | Urdu | semantic | cold generate median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B Q4 | 0.159 | 0.279 | 0.039 | 0.806 | 3.69 s |
| 1.5B Q4 | 0.217 | 0.360 | 0.074 | 0.817 | 5.92 s |
| 3B Q4 | 0.203 | 0.292 | 0.113 | 0.814 | 11.0 s |

1.5B has the best overall F1. 3B has the best Urdu token F1.

## 6. Limitations

Urdu items are translations, and Urdu token F1 stays below 0.12. Exact match does not credit a correct fact wrapped in an extra clause. The pilot is 14 questions, and five of the six gates stay off. The Urdu-article slice is 8 questions. mMARCO did not supervise Urdu.

## 7. Reproducing the run

```bash
export PYTHONPATH=src
python scripts/download_models.py
python scripts/download_papers.py
python scripts/build_urdu_doc.py
python -m mira.cli ingest data/papers/pdf --index data/indexes/benchmark
python -m mira.cli evaluate --profiles fast quality large --out results
pytest
```
