# Technical report: retrieval and grounded answers in English and Urdu

Shaiza Asghar

## 1. Problem

A reader with a folder of papers should be able to ask a question in English or Urdu and receive an answer that names the page it came from. A small local model will invent details if it is asked to answer from memory. The system therefore retrieves passages first, drafts from those passages only, and deletes any sentence that the retrieved text does not support.

The project compares six retrieval methods on one fixed question set and records CPU timings. No score in the results section was filled in by hand.

## 2. Corpus and questions

The corpus is three CC BY 4.0 papers downloaded at a pinned arXiv version:

- Christian Hansen, Casper Hansen, Jakob Grue Simonsen, Stephen Alstrup, and Christina Lioma. *Unsupervised Multi-Index Semantic Hashing.* The Web Conference 2021. arXiv:2103.14460v1. 10 pages.
- Christian Hansen, Casper Hansen, Jakob Grue Simonsen, and Christina Lioma. *Projected Hamming Dissimilarity for Bit-Level Importance Coding in Collaborative Filtering.* The Web Conference 2021. arXiv:2103.14455v1. 9 pages.
- Tomasz J. Piotrowski, Renato L. G. Cavalcante, and Mateusz Gabor. *Fixed points of nonnegative neural networks.* Journal of Machine Learning Research 25 (2024) 1–40. arXiv:2106.16239v9.

`scripts/download_papers.py` keeps a file only when the PDF text states CC BY 4.0 or CC BY-SA 4.0. The extracted text is stored with an attribution header so each gold quotation can be checked without the binary PDF.

PyMuPDF reads each page. Running headers that repeat on at least 60% of pages are removed, line-break hyphenation of lowercase letters is joined, and the text is normalised with Unicode NFKC. Chunks are windows of 160 words with a 40-word overlap, and a chunk never crosses a page boundary. The benchmark index contains 289 chunks.

The test file `data/qa/gold.json` has 32 questions. Twelve are English and answerable. Twelve are Urdu translations of those twelve, with the same gold pages and the same English reference stored in `answer_en`. Four English questions and their Urdu translations ask about topics that do not occur in the papers (BLEU, COVID-19 vaccines, LLaMA, photosynthesis). `data/qa/pilot.json` has eight other questions and is used only to set evidence-gate thresholds. There was no second annotator. Unit tests check that every evidence quotation occurs on a cited page and that the pilot and test questions do not overlap.

## 3. Retrieval

Queries and chunks are tokenised after Urdu folding: Arabic kaf and yeh map to the Urdu letters, hamza-bearing alef maps to alef, Arabic heh maps to heh goal, and both Eastern Arabic digit blocks map to ASCII. Bari yeh (ے) is not folded, because it distinguishes words such as کے and کی.

BM25L (Lv and Zhai) provides the keyword ranking. Dense retrieval encodes passages with the prefix `passage: ` and queries with `query: `, as `intfloat/multilingual-e5-small` requires, L2-normalises the vectors, and searches them with a FAISS `IndexFlatIP` index. Hybrid retrieval fuses the two ranked lists with reciprocal rank fusion at k = 60. For an Urdu question, the translation variant also translates the question with `Helsinki-NLP/opus-mt-ur-en` and fuses the BM25L and dense lists of that English query. Reranking scores the top 20 fused chunks with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

Recall@k, MRR, and nDCG@k are defined in `src/mira/metrics.py` and covered by unit tests with hand-computed expected values. They are macro-averaged over the 24 answerable test questions. Unanswerable questions have no gold pages and are excluded from retrieval averages.

## 4. Generation and grounding

The default generator is `Qwen/Qwen2.5-0.5B-Instruct` in the GGUF file `qwen2.5-0.5b-instruct-q4_k_m.gguf`, loaded by llama.cpp with a 3072-token context, 4 threads, temperature 0, and at most 160 new tokens. The prompt tells the model to answer in the question's language, to cite passage ids, and to reply `NOT_FOUND` when the passages do not contain the answer. `MIRA_LLM_BACKEND=transformers` or `openai` selects the other backends. Missing credentials do not change the default.

After generation, each sentence is compared with the retrieved chunks. It is supported if its content-word overlap is at least 0.35 or its E5 cosine with a chunk is at least 0.82. Supported sentences are shown with the id of the best chunk. If none survive, the system abstains. When the question is Urdu and the grounded draft is English, `Helsinki-NLP/opus-mt-en-ur` translates the sentences and the chunk ids are copied back onto them.

The raw unsupported rate is measured on the draft, before deletion. Citation page precision is measured on the citations that remain. These are different quantities: grounding can make the shown answer faithful while the raw draft still contained unsupported sentences, and a faithful citation can point at a page that repeats a fact but was not the annotated gold page.

## 5. Results

Hardware, model names, and every table below are taken from `results/metrics.json` produced by `python -m mira.cli evaluate --out results`.

The committed run used an Intel Xeon CPU at 2.4 GHz with 4 logical CPUs and 15.63 GiB RAM, Python 3.12.3, PyTorch 2.15.0+cpu, llama-cpp-python 0.3.37, sentence-transformers 6.3.0, and FAISS 1.16.0. There were no exceptions. The 0.4 second warmup recorded in `results/RESULTS.md` is a one-line prompt issued while the GGUF file was already in the page cache. Per-question generation medians are 3.8–4.6 seconds.

### Retrieval, 24 answerable questions

| method | Recall@5 | English | Urdu | MRR | nDCG@5 | Hit@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.500 | 0.833 | 0.167 | 0.240 | 0.324 | 0.500 |
| dense | 0.750 | 0.833 | 0.667 | 0.416 | 0.506 | 0.750 |
| hybrid | 0.750 | 0.917 | 0.583 | 0.401 | 0.506 | 0.750 |
| hybrid_rerank | 0.917 | 0.917 | 0.917 | 0.724 | 0.767 | 0.917 |
| hybrid_translate | 0.708 | 0.917 | 0.500 | 0.497 | 0.561 | 0.708 |
| hybrid_translate_rerank | 0.833 | 0.917 | 0.750 | 0.679 | 0.720 | 0.833 |

Paraphrase Recall@5 was 0.500 (BM25L), 0.833 (dense), and 1.000 (hybrid plus reranking). Lexical Recall@5 was 0.500, 0.667, and 0.833 for the same three methods.

Hybrid plus the cross-encoder is the best page finder. English Recall@5 ties at 0.917 for every fused method. The gap is on Urdu, where BM25L falls to 0.167 because the Urdu wording does not repeat the English page, while reranking the original Urdu question reaches 0.917 and lifts Urdu MRR from 0.212 (dense) to 0.661. Adding an Urdu-to-English translation to the fused lists lowers Urdu Recall@5 to 0.500, and reranking that translated list reaches only 0.750. On this set the translation is a worse query than the Urdu question.

### Answers, 32 test questions

Token F1 treats an abstention on an answerable question as 0. Exact match was 0.000 for every method.

| method | F1 | English F1 | Urdu F1 | abstention accuracy | coverage | unanswerable abstention | citation page precision | raw unsupported rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25 | 0.204 | 0.400 | 0.007 | 0.625 | 0.542 | 0.875 | 0.590 | 0.038 |
| dense | 0.231 | 0.376 | 0.086 | 0.875 | 0.833 | 1.000 | 0.371 | 0.025 |
| hybrid | 0.261 | 0.437 | 0.086 | 0.875 | 0.833 | 1.000 | 0.289 | 0.042 |
| hybrid_rerank | 0.184 | 0.282 | 0.086 | 0.719 | 0.625 | 1.000 | 0.144 | 0.078 |
| hybrid_translate | 0.234 | 0.383 | 0.086 | 0.750 | 0.750 | 0.750 | 0.306 | 0.037 |
| hybrid_translate_rerank | 0.200 | 0.347 | 0.053 | 0.719 | 0.708 | 0.750 | 0.216 | 0.039 |

Hybrid retrieval has the highest F1 even though it is not the best retriever. Its evidence gate stayed off, it answered every English answerable question, and it abstained on every unanswerable question. The reranked hybrid list finds more gold pages, but the pilot set a gate (threshold −1.25 on the cross-encoder logit, pilot accuracy 1.00) that also abstains on some answerable questions, and coverage falls to 0.625. Those abstentions score 0 F1, so the better ranking does not become a better answer score with this generator.

Urdu F1 does not rise with retrieval quality. The best value is 0.086. For the Urdu question "MISH کس کا مخفف ہے؟" the grounded output was "MISH کس کا مخفف ہے۔", a restatement of the question. The embedding rule accepted it because the sentence is near a MISH passage, and the Urdu reference (`Multi-Index Semantic Hashing`) shares no tokens with that restatement. Content F1 against the English reference, before any answer translation, peaks at 0.011 on the Urdu slice. BM25L abstains on almost every answerable Urdu question (coverage 0.083), which is preferable to a fabricated expansion and is why its Urdu F1 is 0.007.

Two English failure modes showed up in the saved predictions. The autoencoder question was answered by hybrid retrieval as widths 200, 100, and 784, with the learning rate cited from arXiv:2103.14460 rather than from the neural-network paper; the raw draft's mention of 30 epochs was dropped by the support check. Citation page precision (0.144–0.590) is much lower than Recall@5 because a citation can be lexically supported by a non-gold page that repeats a number or a name. Urdu citation page precision is 0.000 in the aggregate: the citations that remain do not land on the annotated page.

Raw unsupported rates of 0.025–0.078 mean most draft sentences were either copied from a passage or removed. They do not mean the remaining sentence is the answer to the question.

### Latency

Median retrieval is 0.8 ms for BM25L and 9.6–9.7 ms for dense and hybrid search after the encoder is loaded. Reranking adds about 530 ms. Generation medians are 3816–4603 ms. BM25L's end-to-end median (3635 ms) is below its generation median because 12 of 32 questions never called the generator. The same pattern appears for hybrid plus reranking (22 generations) and for translated hybrid search (28 generations).

Charts are in `results/figures/`. Row-level outputs are in `results/predictions.jsonl`.

## 6. Limitations

The Urdu questions are translations, so a method can look stronger on Urdu simply by matching English names that the translation kept. The papers are English, so this run does not measure retrieval over Urdu prose. The 0.5B draft model is weak at Urdu generation, and the reported Urdu F1 includes the machine translation of that draft. The pilot has eight questions, so the evidence gate is either clearly separable or left off. mMARCO did not supervise Urdu, so the reranker's Urdu gains are specific to this set until they are repeated elsewhere.

## 7. Reproducing the run

```bash
export PYTHONPATH=src
python scripts/download_models.py
python scripts/download_papers.py
python -m mira.cli ingest data/papers/pdf --index data/indexes/benchmark
python -m mira.cli evaluate --out results
pytest
```

The machine used for the committed `results/` directory is recorded at the top of `results/RESULTS.md`.
