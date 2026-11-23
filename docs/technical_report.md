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

English questions use Qwen2.5 1.5B Instruct Q4. Urdu questions use Qwen2.5 3B Instruct Q4, with at most 96 new tokens. The prompt asks for one short sentence and tells the model to keep names and numbers in Latin script. `MIRA_LLM_BACKEND=transformers` or `openai` selects the other backends.

A sentence is kept when its content-word overlap with a retrieved chunk is at least 0.35 or its E5 cosine is at least 0.82, unless the sentence mostly repeats the question. Only the first supported sentence is shown. Among supporting chunks, the citation is the one the retriever ranked first. If the draft is unusable, the system may show a retrieved sentence. An answer that already contains Urdu words is kept. Otherwise `opus-mt-en-ur` translates it after Latin names and numbers are masked, and those terms are restored. If that check fails, the English sentence is kept inside an Urdu frame.

PDF text is normalised with NFKC. A small map then repairs Urdu letters that this article's font extracted as unrelated code points, and parentheses are swapped only in mostly Arabic text. The benchmark index was rebuilt after that repair.

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

The table above is the previous 104-question run, with Qwen2.5 1.5B for every question. Urdu token F1 on hybrid reranking was 0.074 (95% CI 0.049–0.099).

### Urdu revision

The next run keeps the same questions, the same gate rule, and hybrid reranking as the primary retriever. Urdu questions use Qwen2.5 3B. The Urdu article text in the index is the repaired extraction. Raw F1 is unchanged as a definition. Normalized F1 splits a digit run from a following Urdu letter and drops citations and the bilingual frame.

| slice | raw F1 | 95% CI | normalized F1 | semantic | citation page precision |
| --- | ---: | --- | ---: | ---: | ---: |
| Urdu questions, previous | 0.074 | 0.049–0.099 | — | 0.787 | 0.585 |
| Urdu questions, this run | 0.125 | 0.079–0.179 | 0.183 | 0.779 | 0.525 |
| Urdu article, previous | 0.105 | 0.051–0.160 | — | 0.741 | 1.000 |
| Urdu article, this run | 0.216 | 0.111–0.337 | 0.338 | 0.777 | 1.000 |
| All 104, this run | 0.228 | 0.184–0.272 | 0.305 | 0.808 | 0.646 |

Raw exact match is 0.000. Normalized exact match is 0.023 on all 104 questions and 0.000 on the Urdu questions. The raw-F1 intervals for the Urdu questions overlap. On this uncached pass, hybrid reranking generate medians are 6.45 s in English (1.5B) and 14.7 s in Urdu (3B). Retrieve median is 564 ms.

## 6. Limitations

Urdu items are translations. Twelve Urdu answers still keep an English sentence. Raw exact match does not credit a correct year written as `1950ء` inside a sentence; normalized F1 does. The pilot is 14 questions, and five of the six gates stay off. The Urdu-article slice is 8 questions. mMARCO did not supervise Urdu. NLLB was not in this run.

## 7. Reproducing the run

```bash
export PYTHONPATH=src
python scripts/download_models.py
python scripts/download_papers.py
python scripts/build_urdu_doc.py
python -m mira.cli ingest data/papers/pdf --index data/indexes/benchmark
python -m mira.cli evaluate --profiles quality --out results
pytest
```
