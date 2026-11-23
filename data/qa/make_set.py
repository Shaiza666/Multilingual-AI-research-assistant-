"""Assemble the test and pilot files and check that every quotation is on a cited page."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = ROOT / "data" / "papers" / "text"


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def load_pages() -> dict[tuple[str, int], str]:
    pages: dict[tuple[str, int], str] = {}
    for path in TEXT.glob("*.txt"):
        raw = path.read_text(encoding="utf-8")
        for part in raw.split("===== PAGE ")[1:]:
            head, _, body = part.partition("=====")
            number = int(head.strip().split()[0])
            pages[(path.stem, number)] = squash(body)
    return pages


def item(**kwargs):
    kwargs.setdefault("answerable", True)
    kwargs.setdefault("translation_of", None)
    kwargs.setdefault("corpus", "papers")
    kwargs.setdefault("evidence_quotes", [])
    return kwargs


def pair(base: dict, ur_id: str, question: str, answer: str) -> list[dict]:
    en = dict(base)
    ur = dict(base)
    ur.update(
        {
            "id": ur_id,
            "language": "ur",
            "question": question,
            "answer": answer,
            "answer_en": base["answer"],
            "translation_of": base["id"],
        }
    )
    return [en, ur]


PAGES = load_pages()


def check(rows: list[dict]) -> None:
    missing = []
    for row in rows:
        if not row["answerable"]:
            continue
        for quote in row["evidence_quotes"]:
            folded = squash(quote)
            found = any(folded in PAGES.get((doc["doc_id"], doc["page"]), "") for doc in row["gold_pages"])
            if not found:
                missing.append((row["id"], quote, row["gold_pages"]))
    if missing:
        for item_id, quote, pages in missing:
            print("MISSING", item_id, pages, quote)
        raise SystemExit(f"{len(missing)} quotations were not on the cited page")


def page(doc: str, number: int) -> list[dict]:
    return [{"doc_id": doc, "page": number}]


def main() -> None:
    old = json.loads((ROOT / "data" / "qa" / "gold.json").read_text(encoding="utf-8"))
    keep_ids = {
        "mish-en-1", "mish-ur-1", "mish-en-2", "mish-ur-2", "mish-en-3", "mish-ur-3", "mish-en-4", "mish-ur-4",
        "phd-en-1", "phd-ur-1", "phd-en-2", "phd-ur-2", "phd-en-3", "phd-ur-3", "phd-en-4", "phd-ur-4",
        "nn-en-1", "nn-ur-1", "nn-en-2", "nn-ur-2", "nn-en-3", "nn-ur-3", "nn-en-4", "nn-ur-4",
    }
    kept = []
    for row in old["items"]:
        if row["id"] not in keep_ids:
            continue
        row = dict(row)
        row["corpus"] = "papers"
        kept.append(row)

    extra_en = [
        item(
            id="bert-en-1",
            language="en",
            query_type="lexical",
            question="What does BERT stand for?",
            answer="Bidirectional Encoder Representations from Transformers",
            answer_en="Bidirectional Encoder Representations from Transformers",
            gold_pages=page("N19-1423", 1),
            evidence_quotes=["Bidirectional Encoder Representations from Transformers"],
        ),
        item(
            id="bert-en-2",
            language="en",
            query_type="paraphrase",
            question="Which two text collections were used to pre-train BERT, and how large are they?",
            answer="The BooksCorpus has 800M words and English Wikipedia has 2,500M words.",
            answer_en="The BooksCorpus has 800M words and English Wikipedia has 2,500M words.",
            gold_pages=page("N19-1423", 5),
            evidence_quotes=["BooksCorpus (800M words)", "English Wikipedia (2,500M words)"],
        ),
        item(
            id="bert-en-3",
            language="en",
            query_type="lexical",
            question="How many parameters do BERTBASE and BERTLARGE have?",
            answer="BERTBASE has 110M parameters and BERTLARGE has 340M parameters.",
            answer_en="BERTBASE has 110M parameters and BERTLARGE has 340M parameters.",
            gold_pages=page("N19-1423", 3),
            evidence_quotes=["110M", "Total Parameters=340M"],
        ),
        item(
            id="bert-en-4",
            language="en",
            query_type="paraphrase",
            question="What does BERT's masked language model do to some input tokens?",
            answer="It randomly masks some tokens and predicts the original vocabulary id.",
            answer_en="It randomly masks some tokens and predicts the original vocabulary id.",
            gold_pages=page("N19-1423", 1),
            evidence_quotes=["masked language model randomly masks some of the tokens"],
        ),
        item(
            id="bert-en-5",
            language="en",
            query_type="lexical",
            question="Who are the authors of the BERT paper, and which lab are they from?",
            answer="Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova at Google AI Language.",
            answer_en="Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova at Google AI Language.",
            gold_pages=page("N19-1423", 1),
            evidence_quotes=["Jacob Devlin", "Google AI Language"],
        ),
        item(
            id="sbert-en-1",
            language="en",
            query_type="lexical",
            question="What does the title say Sentence-BERT uses to build sentence embeddings?",
            answer="Siamese BERT-Networks",
            answer_en="Siamese BERT-Networks",
            gold_pages=page("D19-1410", 1),
            evidence_quotes=["Sentence Embeddings using Siamese BERT-Networks"],
        ),
        item(
            id="sbert-en-2",
            language="en",
            query_type="paraphrase",
            question="How expensive is it for BERT to find the most similar pair among 10,000 sentences?",
            answer="About 50 million inference computations, roughly 65 hours.",
            answer_en="About 50 million inference computations, roughly 65 hours.",
            gold_pages=page("D19-1410", 1),
            evidence_quotes=["10,000 sentences requires about 50 million", "(~65 hours) with BERT"],
        ),
        item(
            id="sbert-en-3",
            language="en",
            query_type="lexical",
            question="Who wrote Sentence-BERT?",
            answer="Nils Reimers and Iryna Gurevych",
            answer_en="Nils Reimers and Iryna Gurevych",
            gold_pages=page("D19-1410", 1),
            evidence_quotes=["Nils Reimers and Iryna Gurevych"],
        ),
        item(
            id="sbert-en-4",
            language="en",
            query_type="paraphrase",
            question="Where was the Sentence-BERT paper presented?",
            answer="EMNLP-IJCNLP 2019 in Hong Kong, 3-7 November 2019.",
            answer_en="EMNLP-IJCNLP 2019 in Hong Kong, 3-7 November 2019.",
            gold_pages=page("D19-1410", 1),
            evidence_quotes=["Hong Kong, China, November 3–7, 2019"],
        ),
        item(
            id="bart-en-1",
            language="en",
            query_type="lexical",
            question="How is BART trained, in one sentence from the abstract?",
            answer="BART is a denoising autoencoder that corrupts text and learns to reconstruct it.",
            answer_en="BART is a denoising autoencoder that corrupts text and learns to reconstruct it.",
            gold_pages=page("2020.acl-main.703", 1),
            evidence_quotes=["BART, a denoising autoencoder for pretraining sequence-to-sequence models"],
        ),
        item(
            id="bart-en-2",
            language="en",
            query_type="paraphrase",
            question="On which benchmarks does BART match RoBERTa?",
            answer="GLUE and SQuAD",
            answer_en="GLUE and SQuAD",
            gold_pages=page("2020.acl-main.703", 1),
            evidence_quotes=["matches the performance of RoBERTa on GLUE and SQuAD"],
        ),
        item(
            id="bart-en-3",
            language="en",
            query_type="lexical",
            question="Which organisation produced BART?",
            answer="Facebook AI",
            answer_en="Facebook AI",
            gold_pages=page("2020.acl-main.703", 1),
            evidence_quotes=["Facebook AI"],
        ),
        item(
            id="bart-en-5",
            language="en",
            query_type="paraphrase",
            question="What summarization gain does the BART abstract report?",
            answer="Gains of up to 3.5 ROUGE.",
            answer_en="Gains of up to 3.5 ROUGE.",
            gold_pages=page("2020.acl-main.703", 1),
            evidence_quotes=["gains of up to 3.5"],
        ),
        item(
            id="bert-en-6",
            language="en",
            query_type="paraphrase",
            question="How much task-specific architecture does BERT need at fine-tuning time?",
            answer="Just one additional output layer.",
            answer_en="Just one additional output layer.",
            gold_pages=page("N19-1423", 1),
            evidence_quotes=["one additional output layer"],
        ),
        item(
            id="dpr-en-5",
            language="en",
            query_type="lexical",
            question="Which lab produced the dense passage retrieval paper?",
            answer="Facebook AI",
            answer_en="Facebook AI",
            gold_pages=page("2020.emnlp-main.550", 1),
            evidence_quotes=["Facebook AI"],
        ),
        item(
            id="bart-en-4",
            language="en",
            query_type="paraphrase",
            question="Who is the first listed author of the BART paper?",
            answer="Mike Lewis",
            answer_en="Mike Lewis",
            gold_pages=page("2020.acl-main.703", 1),
            evidence_quotes=["Mike Lewis"],
        ),
        item(
            id="dpr-en-1",
            language="en",
            query_type="lexical",
            question="By how much does the dense retriever beat a strong Lucene-BM25 system on top-20 passage retrieval?",
            answer="By 9% to 19% absolute.",
            answer_en="By 9% to 19% absolute.",
            gold_pages=page("2020.emnlp-main.550", 1),
            evidence_quotes=["9%-19% absolute in terms of top-20 passage retrieval accuracy"],
        ),
        item(
            id="dpr-en-2",
            language="en",
            query_type="paraphrase",
            question="Which open-domain QA setting is used when the paper compares end-to-end accuracy with ORQA?",
            answer="The open Natural Questions setting.",
            answer_en="The open Natural Questions setting.",
            gold_pages=page("2020.emnlp-main.550", 2),
            evidence_quotes=["open Natural Questions setting"],
        ),
        item(
            id="dpr-en-3",
            language="en",
            query_type="lexical",
            question="Who is the first listed author of Dense Passage Retrieval?",
            answer="Vladimir Karpukhin",
            answer_en="Vladimir Karpukhin",
            gold_pages=page("2020.emnlp-main.550", 1),
            evidence_quotes=["Vladimir Karpukhin"],
        ),
        item(
            id="dpr-en-4",
            language="en",
            query_type="paraphrase",
            question="Where and when was the dense passage retrieval paper presented?",
            answer="EMNLP 2020, 16-20 November 2020.",
            answer_en="EMNLP 2020, 16-20 November 2020.",
            gold_pages=page("2020.emnlp-main.550", 1),
            evidence_quotes=["November 16–20, 2020"],
        ),
        item(
            id="arith-en-1",
            language="en",
            query_type="lexical",
            question="Which digit of an n-digit by m-digit product can LLMs often predict without chain of thought?",
            answer="The first digit.",
            answer_en="The first digit.",
            gold_pages=page("2024.acl-short.8", 1),
            evidence_quotes=["predict the first digit of n-digit by m-digit multiplication"],
        ),
        item(
            id="arith-en-2",
            language="en",
            query_type="paraphrase",
            question="Which digit of a long multiplication do the authors say models often fail to predict?",
            answer="The last digit, a task equivalent to 1-digit by 1-digit multiplication.",
            answer_en="The last digit, a task equivalent to 1-digit by 1-digit multiplication.",
            gold_pages=page("2024.acl-short.8", 1),
            evidence_quotes=["predict the last digit of an n-digit by m-digit multiplication"],
        ),
        item(
            id="arith-en-3",
            language="en",
            query_type="lexical",
            question="Which university are the arithmetic-paper authors from?",
            answer="The University of Tokyo.",
            answer_en="The University of Tokyo.",
            gold_pages=page("2024.acl-short.8", 1),
            evidence_quotes=["University of Tokyo"],
        ),
        item(
            id="arith-en-4",
            language="en",
            query_type="paraphrase",
            question="Who is the first listed author of the arithmetic short paper?",
            answer="Andrew Gambardella",
            answer_en="Andrew Gambardella",
            gold_pages=page("2024.acl-short.8", 1),
            evidence_quotes=["Andrew Gambardella"],
        ),
        item(
            id="peft-en-1",
            language="en",
            query_type="lexical",
            question="What lightweight parameters does the paper give as an example that can be stored and shared?",
            answer="Adapters.",
            answer_en="Adapters.",
            gold_pages=page("2023.acl-short.24", 1),
            evidence_quotes=["parameters, such as adapters, can be easily stored and shared"],
        ),
        item(
            id="peft-en-2",
            language="en",
            query_type="paraphrase",
            question="What is the title claim about parameter-efficient weight ensembling?",
            answer="It facilitates task-level knowledge transfer.",
            answer_en="It facilitates task-level knowledge transfer.",
            gold_pages=page("2023.acl-short.24", 1),
            evidence_quotes=["Parameter-efficient Weight Ensembling Facilitates Task-level Knowledge Transfer"],
        ),
        item(
            id="peft-en-3",
            language="en",
            query_type="lexical",
            question="Which university are the weight-ensembling authors affiliated with?",
            answer="Tsinghua University.",
            answer_en="Tsinghua University.",
            gold_pages=page("2023.acl-short.24", 1),
            evidence_quotes=["Tsinghua University"],
        ),
        item(
            id="peft-en-4",
            language="en",
            query_type="paraphrase",
            question="Who is the first listed author of the weight-ensembling short paper?",
            answer="Xingtai Lv",
            answer_en="Xingtai Lv",
            gold_pages=page("2023.acl-short.24", 1),
            evidence_quotes=["Xingtai Lv"],
        ),
        item(
            id="urdu-en-1",
            language="en",
            query_type="lexical",
            corpus="urdu_doc",
            question="In which year did Alan Turing propose the Turing test, according to the Urdu article?",
            answer="1950",
            answer_en="1950",
            gold_pages=page("urdu-ai", 1),
            evidence_quotes=["1950ء میں ایلن ٹورنگ نے ٹورنگ ٹیسٹ"],
        ),
        item(
            id="urdu-en-2",
            language="en",
            query_type="paraphrase",
            corpus="urdu_doc",
            question="At which conference was the term artificial intelligence coined, and in what year?",
            answer="The Dartmouth conference in 1956.",
            answer_en="The Dartmouth conference in 1956.",
            gold_pages=page("urdu-ai", 1),
            evidence_quotes=["1956ء میں ڈارٹ ماتھ کانفرنس"],
        ),
        item(
            id="urdu-en-3",
            language="en",
            query_type="lexical",
            corpus="urdu_doc",
            question="What does the Urdu article say machine learning is, relative to artificial intelligence?",
            answer="A subfield of artificial intelligence.",
            answer_en="A subfield of artificial intelligence.",
            gold_pages=page("urdu-ai", 1),
            evidence_quotes=["مشین لرننگ ہے، جو مصنوعی ذہانت کا ایک ذیلی شعبہ"],
        ),
        item(
            id="urdu-en-4",
            language="en",
            query_type="paraphrase",
            corpus="urdu_doc",
            question="How many types of artificial intelligence does the Urdu article list, and what is the first type called in English?",
            answer="Four types. The first is Reactive Machines.",
            answer_en="Four types. The first is Reactive Machines.",
            gold_pages=page("urdu-ai", 2),
            evidence_quotes=["چار اقسام میں تقسیم", "رد عمل مشینیں (Reactive Machines)"],
        ),
    ]
    urdu = [
        ("bert-ur-1", "bert-en-1", "BERT کس کا مخفف ہے؟", "Bidirectional Encoder Representations from Transformers"),
        ("bert-ur-2", "bert-en-2", "BERT کی پیشگی تربیت کے لیے کون سے دو متنی مجموعے استعمال ہوئے اور وہ کتنے بڑے ہیں؟", "BooksCorpus میں 800M الفاظ ہیں اور English Wikipedia میں 2,500M الفاظ ہیں۔"),
        ("bert-ur-3", "bert-en-3", "BERTBASE اور BERTLARGE میں کتنے پیرامیٹرز ہیں؟", "BERTBASE میں 110M اور BERTLARGE میں 340M پیرامیٹرز ہیں۔"),
        ("bert-ur-4", "bert-en-4", "BERT کا masked language model کچھ ان پٹ ٹوکنز کے ساتھ کیا کرتا ہے؟", "وہ کچھ ٹوکنز کو بے ترتیب چھپاتا ہے اور اصل vocabulary id کی پیش گوئی کرتا ہے۔"),
        ("bert-ur-5", "bert-en-5", "BERT کے مقالے کے مصنف کون ہیں اور وہ کس لیب سے ہیں؟", "Jacob Devlin، Ming-Wei Chang، Kenton Lee، اور Kristina Toutanova، Google AI Language سے۔"),
        ("sbert-ur-1", "sbert-en-1", "Sentence-BERT جملوں کے ایمبیڈنگز کے لیے کس ڈھانچے کا استعمال کرتا ہے؟", "Siamese BERT-Networks"),
        ("sbert-ur-2", "sbert-en-2", "10,000 جملوں میں سب سے ملتا جلتا جوڑا ڈھونڈنے کے لیے BERT کتنا مہنگا ہے؟", "تقریباً 50 ملین انفرنس حسابات، یعنی قریب 65 گھنٹے۔"),
        ("sbert-ur-3", "sbert-en-3", "Sentence-BERT کس نے لکھا؟", "Nils Reimers اور Iryna Gurevych"),
        ("sbert-ur-4", "sbert-en-4", "Sentence-BERT کا مقالہ کہاں پیش کیا گیا؟", "EMNLP-IJCNLP 2019، ہانگ کانگ، 3 تا 7 نومبر 2019۔"),
        ("bart-ur-1", "bart-en-1", "خلاصے کے مطابق BART کی تربیت کیسے ہوتی ہے؟", "BART ایک denoising autoencoder ہے جو متن کو خراب کر کے اصل متن دوبارہ بناتا ہے۔"),
        ("bart-ur-2", "bart-en-2", "BART کن بینچ مارکس پر RoBERTa کے برابر کارکردگی دکھاتا ہے؟", "GLUE اور SQuAD"),
        ("bart-ur-3", "bart-en-3", "BART کس ادارے نے بنایا؟", "Facebook AI"),
        ("bart-ur-4", "bart-en-4", "BART کے مقالے کا پہلا مصنف کون ہے؟", "Mike Lewis"),
        ("bart-ur-5", "bart-en-5", "BART کا خلاصہ خلاصہ سازی میں کتنا فائدہ بتاتا ہے؟", "3.5 ROUGE تک کا فائدہ۔"),
        ("bert-ur-6", "bert-en-6", "فائن ٹیوننگ کے وقت BERT کو کتنی اضافی تہہ درکار ہوتی ہے؟", "صرف ایک اضافی آؤٹ پٹ تہہ۔"),
        ("dpr-ur-5", "dpr-en-5", "dense passage retrieval کا مقالہ کس لیب نے بنایا؟", "Facebook AI"),
        ("dpr-ur-1", "dpr-en-1", "dense retriever، Lucene-BM25 سے top-20 passage retrieval پر کتنا آگے ہے؟", "9% سے 19% مطلق۔"),
        ("dpr-ur-2", "dpr-en-2", "ORQA سے آخری سوال جواب کی درستگی کا موازنہ کس ترتیب میں کیا گیا؟", "open Natural Questions کی ترتیب۔"),
        ("dpr-ur-3", "dpr-en-3", "Dense Passage Retrieval کا پہلا مصنف کون ہے؟", "Vladimir Karpukhin"),
        ("dpr-ur-4", "dpr-en-4", "dense passage retrieval کا مقالہ کب پیش کیا گیا؟", "EMNLP 2020، 16 تا 20 نومبر 2020۔"),
        ("arith-ur-1", "arith-en-1", "chain of thought کے بغیر ماڈل ضرب کے کس ہندسے کی اکثر پیش گوئی کر لیتے ہیں؟", "پہلا ہندسہ۔"),
        ("arith-ur-2", "arith-en-2", "مصنفین کے مطابق لمبی ضرب کا کون سا ہندسہ ماڈل اکثر غلط بتاتے ہیں؟", "آخری ہندسہ، جو ایک ہندسے کو ایک ہندسے سے ضرب دینے کے برابر ہے۔"),
        ("arith-ur-3", "arith-en-3", "حساب کے مقالے کے مصنف کس یونیورسٹی سے ہیں؟", "University of Tokyo"),
        ("arith-ur-4", "arith-en-4", "حساب والے مختصر مقالے کا پہلا مصنف کون ہے؟", "Andrew Gambardella"),
        ("peft-ur-1", "peft-en-1", "مقالہ کن ہلکے پیرامیٹرز کی مثال دیتا ہے جنہیں محفوظ اور شیئر کیا جا سکتا ہے؟", "Adapters"),
        ("peft-ur-2", "peft-en-2", "parameter-efficient weight ensembling کے عنوان کا دعویٰ کیا ہے؟", "یہ task-level knowledge transfer کو آسان بناتا ہے۔"),
        ("peft-ur-3", "peft-en-3", "weight ensembling کے مصنف کس یونیورسٹی سے وابستہ ہیں؟", "Tsinghua University"),
        ("peft-ur-4", "peft-en-4", "weight ensembling والے مختصر مقالے کا پہلا مصنف کون ہے؟", "Xingtai Lv"),
        ("urdu-ur-1", "urdu-en-1", "اردو مضمون کے مطابق ایلن ٹورنگ نے ٹورنگ ٹیسٹ کس سال تجویز کیا؟", "1950"),
        ("urdu-ur-2", "urdu-en-2", "مصنوعی ذہانت کی اصطلاح کس کانفرنس میں اور کس سال وضع ہوئی؟", "1956 میں ڈارٹ ماتھ کانفرنس میں۔"),
        ("urdu-ur-3", "urdu-en-3", "اردو مضمون مشین لرننگ کو مصنوعی ذہانت کے مقابلے میں کیا کہتا ہے؟", "مصنوعی ذہانت کا ایک ذیلی شعبہ۔"),
        ("urdu-ur-4", "urdu-en-4", "اردو مضمون مصنوعی ذہانت کی کتنی اقسام بتاتا ہے اور انگریزی میں پہلی قسم کا نام کیا ہے؟", "چار اقسام۔ پہلی Reactive Machines ہے۔"),
    ]
    by_id = {row["id"]: row for row in extra_en}
    extra = []
    for ur_id, en_id, question, answer in urdu:
        extra.extend(pair(by_id[en_id], ur_id, question, answer))

    un_en = [
        ("u-en-1", "What photosynthesis rate did the experiments measure?"),
        ("u-en-2", "Which cricket World Cup final does the corpus describe?"),
        ("u-en-3", "What piano sonata do the authors analyze?"),
        ("u-en-4", "Which Shakespeare play is used as a running example?"),
        ("u-en-5", "What bitcoin price do the authors forecast?"),
        ("u-en-6", "Which COVID-19 vaccine efficacy is reported?"),
        ("u-en-7", "How many moons of Jupiter does the paper catalogue?"),
        ("u-en-8", "What sourdough hydration do the authors recommend?"),
    ]
    un_ur = [
        ("u-ur-1", "u-en-1", "تجربوں نے ضیائی تالیف کی کون سی شرح ناپی؟"),
        ("u-ur-2", "u-en-2", "یہ مجموعہ کرکٹ ورلڈ کپ کے کس فائنل کو بیان کرتا ہے؟"),
        ("u-ur-3", "u-en-3", "مصنفین کس پیانو سوناٹا کا تجزیہ کرتے ہیں؟"),
        ("u-ur-4", "u-en-4", "مثال کے طور پر شیکسپیئر کا کون سا ڈرامہ استعمال ہوا ہے؟"),
        ("u-ur-5", "u-en-5", "مصنفین بٹ کوائن کی کون سی قیمت کی پیش گوئی کرتے ہیں؟"),
        ("u-ur-6", "u-en-6", "کووڈ-19 کی کس ویکسین کی افادیت بیان کی گئی ہے؟"),
        ("u-ur-7", "u-en-7", "مقالہ مشتری کے کتنے چاند گنتا ہے؟"),
        ("u-ur-8", "u-en-8", "مصنفین ساورڈو کی کون سی ہائیڈریشن تجویز کرتے ہیں؟"),
    ]
    unanswerable = []
    for qid, question in un_en:
        unanswerable.append(
            item(
                id=qid,
                language="en",
                query_type="unanswerable",
                answerable=False,
                question=question,
                answer="",
                answer_en="",
                gold_pages=[],
            )
        )
    for ur_id, en_id, question in un_ur:
        unanswerable.append(
            item(
                id=ur_id,
                language="ur",
                query_type="unanswerable",
                answerable=False,
                question=question,
                answer="",
                answer_en="",
                gold_pages=[],
                translation_of=en_id,
            )
        )

    test_items = kept + extra + unanswerable
    check(test_items)
    blob = "\n".join(path.read_text(encoding="utf-8").lower() for path in TEXT.glob("*.txt"))
    absent = ["photosynthesis", "cricket", "piano", "shakespeare", "bitcoin", "sourdough", "jupiter"]
    # vaccine may occur; only keep terms that are truly absent
    absent = [term for term in absent if term not in blob]
    leaked = [term for term in ["photosynthesis", "cricket", "piano", "shakespeare", "bitcoin", "sourdough"] if term in blob]
    if leaked:
        raise SystemExit(f"unanswerable topic appears in the corpus: {leaked}")

    old_pilot = json.loads((ROOT / "data" / "qa" / "pilot.json").read_text(encoding="utf-8"))
    pilot_keep = {
        "p-mish-en", "p-mish-ur", "p-phd-en", "p-phd-ur", "p-nn-en", "p-nn-ur", "p-u-en", "p-u-ur",
    }
    pilot_items = []
    for row in old_pilot["items"]:
        if row["id"] not in pilot_keep:
            continue
        row = dict(row)
        row["corpus"] = "papers"
        pilot_items.append(row)
    more_pilot = [
        item(
            id="p-bert-en",
            language="en",
            query_type="lexical",
            question="In which city and year was the BERT paper presented?",
            answer="Minneapolis, Minnesota, in June 2019.",
            answer_en="Minneapolis, Minnesota, in June 2019.",
            gold_pages=page("N19-1423", 1),
            evidence_quotes=["Minneapolis, Minnesota, June 2 - June 7, 2019"],
        ),
        item(
            id="p-dpr-en",
            language="en",
            query_type="lexical",
            question="What absolute top-20 gain over Lucene-BM25 does the dense retriever report?",
            answer="9% to 19%.",
            answer_en="9% to 19%.",
            gold_pages=page("2020.emnlp-main.550", 1),
            evidence_quotes=["9%-19% absolute"],
        ),
    ]
    pilot_items.extend(pair(more_pilot[0], "p-bert-ur", "BERT کا مقالہ کس شہر اور کس سال پیش کیا گیا؟", "جون 2019 میں Minneapolis، Minnesota میں۔"))
    pilot_items.extend(pair(more_pilot[1], "p-dpr-ur", "dense retriever Lucene-BM25 پر top-20 میں کتنا مطلق اضافہ بتاتا ہے؟", "9% سے 19%۔"))
    pilot_items.append(
        item(
            id="p-u-en-2",
            language="en",
            query_type="unanswerable",
            answerable=False,
            question="Which sourdough starter schedule do the authors publish?",
            answer="",
            answer_en="",
            gold_pages=[],
        )
    )
    pilot_items.append(
        item(
            id="p-u-ur-2",
            language="ur",
            query_type="unanswerable",
            answerable=False,
            question="مصنفین ساورڈو سٹارٹر کا کون سا شیڈول شائع کرتے ہیں؟",
            answer="",
            answer_en="",
            gold_pages=[],
            translation_of="p-u-en-2",
        )
    )
    check(pilot_items)
    test_questions = {row["question"] for row in test_items}
    overlap = [row["id"] for row in pilot_items if row["question"] in test_questions]
    if overlap:
        raise SystemExit(f"pilot overlaps test: {overlap}")

    construction = (
        "The test set has English questions, Urdu translations of those questions, and unanswerable questions in both languages. "
        "Gold pages are 1-based pages of the ingested PDF. Every answerable item stores a short quotation that occurs on a cited page. "
        "Urdu questions and Urdu reference answers were written as translations of the English items; they are not an independently collected Urdu information-need study, and there was no second annotator. "
        "Nine documents are research papers. Three state CC BY 4.0 inside the PDF (arXiv 2103.14460v1, 2103.14455v1, and 2106.16239v9). "
        "Six are ACL Anthology papers from 2019-2024. The Anthology states that materials published in or after 2016 are under CC BY 4.0 (https://aclanthology.org/). "
        "One additional document is the Urdu Wikipedia article مصنوعی ذہانت, revision 11740367, under CC BY-SA 4.0. "
        "Questions tagged corpus=urdu_doc are about that article, including English questions over Urdu text. "
        "The pilot file is disjoint and is used only to set evidence-gate thresholds. It is not scored in the reported tables. "
        "This is a project benchmark, not a community dataset."
    )
    n_en = sum(1 for row in test_items if row["language"] == "en" and row["answerable"])
    n_ur = sum(1 for row in test_items if row["language"] == "ur" and row["answerable"])
    n_uen = sum(1 for row in test_items if row["language"] == "en" and not row["answerable"])
    n_uur = sum(1 for row in test_items if row["language"] == "ur" and not row["answerable"])
    gold = {
        "name": "mira-test",
        "version": 2,
        "built": "2026-11-23",
        "counts": {
            "papers": 9,
            "urdu_documents": 1,
            "test_items": len(test_items),
            "answerable_en": n_en,
            "answerable_ur": n_ur,
            "unanswerable_en": n_uen,
            "unanswerable_ur": n_uur,
        },
        "absent_terms": ["photosynthesis", "cricket", "piano", "shakespeare", "bitcoin", "sourdough"],
        "construction": construction,
        "items": test_items,
    }
    pilot = {
        "name": "mira-pilot",
        "version": 2,
        "built": "2026-11-23",
        "counts": {"pilot_items": len(pilot_items)},
        "construction": "Disjoint from gold.json. Used only to choose per-method evidence-gate thresholds. Not included in reported scores.",
        "items": pilot_items,
    }
    (ROOT / "data" / "qa" / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "data" / "qa" / "pilot.json").write_text(json.dumps(pilot, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test", len(test_items), "en", n_en, "ur", n_ur, "un", n_uen, n_uur)
    print("pilot", len(pilot_items))


if __name__ == "__main__":
    main()
