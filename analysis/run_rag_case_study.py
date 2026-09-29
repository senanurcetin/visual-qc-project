"""Steel QC knowledge assistant: build, evaluate and project an embedding-based RAG system.

Pipeline (everything runs locally, no paid API):
  1. Fetch a fixed Wikipedia knowledge base (CC BY-SA) and chunk it into passages.
  2. Build an evaluation set: LLM-generated questions with a known source passage, plus a
     hand-curated set with article-level labels.
  3. Benchmark retrievers (TF-IDF baseline vs. three sentence-embedding models).
  4. Index the selected embeddings in a Chroma vector database; all published retrieval
     results come from that index.
  5. Generate grounded answers (and closed-book control answers) with a small open LLM.
  6. Score faithfulness with an NLI cross-encoder: is each answer sentence entailed by
     the retrieved passages?
  7. Project the embeddings to 3D/2D with UMAP and measure how much the projection distorts
     neighbourhoods, so the visual can be honest about it.

Run with the dependencies in requirements-rag.txt:
    python analysis/run_rag_case_study.py
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.rag import metrics as M  # noqa: E402
from analysis.rag.chunking import chunk_sections, split_sections  # noqa: E402
from analysis.rag.corpus import LICENSE, TOPICS, article_topics, fetch_article  # noqa: E402

CACHE = ROOT / "analysis" / ".cache" / "rag"
OUT = ROOT / "docs" / "data" / "rag-knowledge-assistant"
SEED = 42
MAX_CHUNKS_PER_ARTICLE = 32
QUESTIONS_PER_ARTICLE = 2
TOP_K = 4  # passages handed to the LLM
EVAL_K = 10
ENTAILMENT_THRESHOLD = 0.5
UMAP_MIN_DIST = 0.4
ABSTAIN = "I don't know based on the provided sources."

GENERATORS = {
    "qwen2.5-1.5b": {"name": "Qwen/Qwen2.5-1.5B-Instruct", "label": "Qwen2.5-1.5B (fp16)", "quantization": None},
    "qwen2.5-3b": {"name": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5-3B (4-bit NF4)", "quantization": "nf4"},
}
QUESTION_GENERATOR = "qwen2.5-1.5b"  # writes the synthetic evaluation questions
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
EMBEDDERS = {
    "minilm-l6": {"name": "sentence-transformers/all-MiniLM-L6-v2", "label": "MiniLM-L6 (384-d)", "query_prefix": ""},
    "bge-small": {"name": "BAAI/bge-small-en-v1.5", "label": "BGE-small (384-d)",
                  "query_prefix": "Represent this sentence for searching relevant passages: "},
    "bge-base": {"name": "BAAI/bge-base-en-v1.5", "label": "BGE-base (768-d)",
                 "query_prefix": "Represent this sentence for searching relevant passages: "},
}

# Hand-curated questions with article-level relevance labels. The first six are the
# showcase questions rendered on the /rag page.
CURATED_QUESTIONS = [
    ("What causes mill scale on hot-rolled steel and how is it removed before painting?", ["Mill scale", "Pickling (metal)"]),
    ("How does pitting corrosion start on stainless steel?", ["Pitting corrosion", "Stainless steel", "Passivation (chemistry)"]),
    ("When is a process considered out of statistical control on a control chart?", ["Control chart", "Statistical process control"]),
    ("How can machine vision detect surface defects on a production line?", ["Machine vision", "Automated optical inspection", "Visual inspection"]),
    ("Why do quality teams sort defect causes with a Pareto chart?", ["Pareto chart", "Pareto principle"]),
    ("What image structure does a histogram of oriented gradients descriptor capture?", ["Histogram of oriented gradients"]),
    ("What is the difference between hot rolling and cold rolling of steel?", ["Rolling (metalworking)", "Hot working", "Cold working"]),
    ("Why are non-metallic inclusions harmful to steel quality?", ["Non-metallic inclusions", "Steelmaking"]),
    ("How does acid pickling clean the surface of steel strip?", ["Pickling (metal)"]),
    ("How does a zinc coating protect steel from rusting?", ["Galvanization", "Hot-dip galvanization", "Corrosion"]),
    ("What does the Cpk index say about whether a process can meet its specification limits?", ["Process capability index"]),
    ("How is a failure mode prioritised in an FMEA?", ["Failure mode and effects analysis"]),
    ("How does ultrasonic testing find internal flaws in metal?", ["Ultrasonic testing", "Nondestructive testing"]),
    ("Which surface cracks can magnetic particle inspection reveal, and on which materials does it work?", ["Magnetic particle inspection"]),
    ("How does a random forest reduce overfitting compared with a single decision tree?", ["Random forest"]),
    ("What texture information does a Gabor filter extract from an image?", ["Gabor filter"]),
    ("How is surface roughness measured and expressed?", ["Surface roughness", "Surface finish"]),
    ("Why does shot peening improve the fatigue life of metal parts?", ["Shot peening", "Fatigue (material)"]),
    ("How does acceptance sampling decide whether to accept a production lot?", ["Acceptance sampling"]),
    ("What is the goal of root cause analysis after a quality failure?", ["Root-cause analysis"]),
    ("How does continuous casting turn molten steel into slabs?", ["Continuous casting", "Steelmaking"]),
    ("How can anomaly detection flag unusual products when defect examples are rare?", ["Anomaly detection"]),
]
SHOWCASE_COUNT = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote docs/data/rag-knowledge-assistant/{name}")


def r4(x):
    return None if x is None else round(float(x), 4)


# ── 1. Corpus ─────────────────────────────────────────────────────────────────
def build_corpus() -> tuple[list[dict], list[dict]]:
    articles, chunks = [], []
    for title, topic in article_topics():
        article = fetch_article(title, CACHE / "wiki")
        passages = chunk_sections(split_sections(article["extract"]))[:MAX_CHUNKS_PER_ARTICLE]
        articles.append({
            "title": article["title"], "topic": topic, "url": article["url"],
            "revid": article["revid"], "chunks": len(passages),
        })
        for p in passages:
            heading = article["title"] if p.section == "Introduction" else f"{article['title']} — {p.section}"
            chunks.append({
                "id": f"c{len(chunks):04d}", "article": article["title"], "topic": topic,
                "section": p.section, "text": p.text, "embed_text": f"{heading}: {p.text}",
            })
    return articles, chunks


# ── Local LLM helpers ─────────────────────────────────────────────────────────
class Generator:
    def __init__(self, key: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        spec = GENERATORS[key]
        self.torch = torch
        self.name = spec["name"]
        self.tok = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        quant = None
        if spec["quantization"] == "nf4":
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, dtype=torch.float16, quantization_config=quant,
            device_map="cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.cache_path = CACHE / "generations.json"
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()

    def _key(self, messages, max_new_tokens):
        return hashlib.sha1(json.dumps([self.name, messages, max_new_tokens]).encode()).hexdigest()

    def generate(self, batch: list[list[dict]], max_new_tokens: int = 200, batch_size: int = 8) -> list[str]:
        outputs: dict[int, str] = {}
        todo = []
        for i, messages in enumerate(batch):
            key = self._key(messages, max_new_tokens)
            if key in self.cache:
                outputs[i] = self.cache[key]
            else:
                todo.append(i)
        for start in range(0, len(todo), batch_size):
            idx = todo[start:start + batch_size]
            prompts = [self.tok.apply_chat_template(batch[i], tokenize=False, add_generation_prompt=True) for i in idx]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            for row, i in enumerate(idx):
                text = self.tok.decode(gen[row, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                outputs[i] = text
                self.cache[self._key(batch[i], max_new_tokens)] = text
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            log(f"  generated {min(start + batch_size, len(todo))}/{len(todo)}")
        return [outputs[i] for i in range(len(batch))]


def load_generator(key: str, attempts: int = 3) -> Generator:
    """Windows can refuse to memory-map large checkpoints when the page file is busy; retry."""
    import gc

    for attempt in range(1, attempts + 1):
        try:
            return Generator(key)
        except OSError as exc:
            if attempt == attempts:
                raise
            log(f"  loading {key} failed ({exc}); retrying in 30s")
            gc.collect()
            time.sleep(30)


# ── 2. Evaluation set ─────────────────────────────────────────────────────────
def question_prompt(chunk: dict) -> list[dict]:
    return [
        {"role": "system", "content": "You write evaluation questions for a document search system used by steel quality engineers."},
        {"role": "user", "content": (
            f"Passage from the Wikipedia article \"{chunk['article']}\":\n\"\"\"{chunk['text']}\"\"\"\n\n"
            "Write ONE question that this passage answers. Rules:\n"
            "- The question must make sense on its own: name the subject explicitly and never refer to "
            "'the passage', 'the text' or 'the article'.\n"
            "- Ask about one specific fact, cause, method or definition from the passage.\n"
            "- Use your own wording instead of copying long phrases.\n"
            "Output only the question."
        )},
    ]


def valid_question(q: str) -> bool:
    low = q.lower()
    return (q.endswith("?") and 5 <= len(q.split()) <= 32 and "\n" not in q
            and not any(bad in low for bad in ("passage", "the text", "the article", "mentioned above")))


def build_eval_set(chunks: list[dict], generator: Generator) -> list[dict]:
    rng = random.Random(SEED)
    by_article: dict[str, list[dict]] = {}
    for c in chunks:
        if len(c["text"].split()) >= 70:
            by_article.setdefault(c["article"], []).append(c)
    sampled = [c for group in by_article.values() for c in rng.sample(group, min(QUESTIONS_PER_ARTICLE, len(group)))]
    raw = generator.generate([question_prompt(c) for c in sampled], max_new_tokens=60)
    items = []
    for c, q in zip(sampled, raw, strict=False):
        q = q.strip().strip('"').splitlines()[0].strip() if q.strip() else ""
        if valid_question(q):
            items.append({"question": q, "gold_chunk": c["id"], "gold_article": c["article"],
                          "topic": c["topic"], "lexical_overlap": r4(M.lexical_overlap(q, c["text"]))})
    log(f"eval set: {len(items)} valid of {len(sampled)} generated questions")
    return items


# ── 3. Retrieval benchmark ────────────────────────────────────────────────────
def embed_corpus(key: str, chunks: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    path = CACHE / f"emb-{key}.npy"
    texts_hash = hashlib.sha1("".join(c["embed_text"] for c in chunks).encode()).hexdigest()[:12]
    meta = CACHE / f"emb-{key}.hash"
    if path.exists() and meta.exists() and meta.read_text() == texts_hash:
        return np.load(path)
    model = SentenceTransformer(EMBEDDERS[key]["name"], device="cuda")
    vectors = model.encode([c["embed_text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False).astype(np.float32)
    np.save(path, vectors)
    meta.write_text(texts_hash)
    return vectors


def embed_queries(key: str, questions: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDERS[key]["name"], device="cuda")
    prefix = EMBEDDERS[key]["query_prefix"]
    return model.encode([prefix + q for q in questions], batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32)


def rank_dense(doc_vecs: np.ndarray, query_vecs: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(query_vecs @ doc_vecs.T), axis=1)[:, :k]


def rank_tfidf(chunks: list[dict], questions: list[str], k: int) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    docs = vec.fit_transform([c["embed_text"] for c in chunks])
    scores = (vec.transform(questions) @ docs.T).toarray()
    return np.argsort(-scores, axis=1)[:, :k]


def score_rankings(ranked_idx: np.ndarray, chunks: list[dict], eval_items: list[dict], curated: list[dict]) -> dict:
    n = len(eval_items)
    ids = [[chunks[i]["id"] for i in row] for row in ranked_idx[:n]]
    arts = [[chunks[i]["article"] for i in row] for row in ranked_idx[:n]]
    gold_ids = [e["gold_chunk"] for e in eval_items]
    gold_arts = [e["gold_article"] for e in eval_items]
    curated_rows = ranked_idx[n:]
    curated_hit5 = np.mean([
        any(chunks[i]["article"] in c["gold_articles"] for i in row[:5]) for row, c in zip(curated_rows, curated, strict=False)])
    curated_hit1 = np.mean([chunks[row[0]]["article"] in c["gold_articles"] for row, c in zip(curated_rows, curated, strict=False)])
    return {
        "hit_at_1": r4(M.hit_at_k(ids, gold_ids, 1)),
        "hit_at_4": r4(M.hit_at_k(ids, gold_ids, TOP_K)),
        "hit_at_10": r4(M.hit_at_k(ids, gold_ids, 10)),
        "mrr_at_10": r4(M.mean_reciprocal_rank(ids, gold_ids, 10)),
        "article_hit_at_4": r4(M.hit_at_k(arts, gold_arts, TOP_K)),
        "curated_article_hit_at_1": r4(curated_hit1),
        "curated_article_hit_at_5": r4(curated_hit5),
    }


# ── 4. Vector database ────────────────────────────────────────────────────────
def build_chroma(chunks: list[dict], vectors: np.ndarray, model_key: str):
    import chromadb

    client = chromadb.PersistentClient(path=str(CACHE / "chroma"))
    try:
        client.delete_collection("steel_qc_kb")
    except Exception:
        pass
    collection = client.create_collection(
        "steel_qc_kb", metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDERS[model_key]["name"]})
    for start in range(0, len(chunks), 500):
        batch = chunks[start:start + 500]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors[start:start + 500].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"article": c["article"], "topic": c["topic"], "section": c["section"]} for c in batch],
        )
    return collection


def chroma_search(collection, query_vecs: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
    res = collection.query(query_embeddings=query_vecs.tolist(), n_results=k, include=["distances"])
    return [[(i, 1.0 - d) for i, d in zip(ids, dists, strict=False)] for ids, dists in zip(res["ids"], res["distances"], strict=False)]


# ── 5–6. Answers and faithfulness ─────────────────────────────────────────────
RAG_PROMPTS = {
    "v1_basic": (
        "You are a quality-engineering assistant for a steel strip mill. Answer using only the numbered "
        "context passages and cite the passages you use like [1]. If the passages do not contain the "
        f"answer, reply exactly: {ABSTAIN}"),
    "v2_strict": (
        "You are a quality-engineering assistant for a steel strip mill. Answer in English using only facts "
        "that are explicitly stated in the numbered context passages. Do not add background knowledge, "
        "examples or conclusions that the passages do not state. Stay close to the passages' wording and "
        f"cite a passage for every sentence like [1]. If the passages do not contain the answer, reply exactly: {ABSTAIN}"),
}


def rag_prompt(question: str, passages: list[dict], version: str) -> list[dict]:
    context = "\n\n".join(
        f"[{n}] ({p['article']} — {p['section']}) {p['text']}" for n, p in enumerate(passages, 1))
    length = "2-4 sentences" if version == "v1_basic" else "2-3 sentences"
    return [
        {"role": "system", "content": RAG_PROMPTS[version]},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer in {length}."},
    ]


def closed_book_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a quality-engineering assistant for a steel strip mill."},
        {"role": "user", "content": f"Question: {question}\nAnswer in 2-4 sentences."},
    ]


class FaithfulnessScorer:
    def __init__(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(NLI_MODEL, device="cuda")
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def score(self, answer: str, passages: list[str]) -> dict:
        if ABSTAIN.lower().rstrip(".") in answer.lower():
            return {"abstained": True, "faithfulness": None, "sentences": []}
        sentences = M.answer_sentences(answer)
        if not sentences:
            return {"abstained": False, "faithfulness": None, "sentences": []}
        windows = [(n, w) for n, passage in enumerate(passages) for w in M.premise_windows(passage)]
        pairs = [(w, s) for s in sentences for _, w in windows]
        probs = self.model.predict(pairs, apply_softmax=True, batch_size=64, show_progress_bar=False)
        ent = np.asarray(probs)[:, self.entail].reshape(len(sentences), len(windows))
        best_window = ent.argmax(axis=1)
        best = ent.max(axis=1)
        return {
            "abstained": False,
            "faithfulness": r4(M.faithfulness(best, ENTAILMENT_THRESHOLD)),
            "sentences": [{"text": s, "entailment": r4(p), "supported": bool(p >= ENTAILMENT_THRESHOLD),
                           "best_passage": windows[best_window[i]][0] + 1,
                           "evidence": windows[best_window[i]][1] if p >= ENTAILMENT_THRESHOLD else None}
                          for i, (s, p) in enumerate(zip(sentences, best, strict=False))],
        }


def summarize_faithfulness(scores: list[dict]) -> dict:
    answered = [s for s in scores if not s["abstained"] and s["faithfulness"] is not None]
    values = np.array([s["faithfulness"] for s in answered]) if answered else np.array([])
    sentences = [x for s in answered for x in s["sentences"]]
    return {
        "questions": len(scores),
        "answered": len(answered),
        "abstention_rate": r4(np.mean([s["abstained"] for s in scores])) if scores else None,
        "mean_faithfulness": r4(values.mean()) if values.size else None,
        "fully_supported_share": r4(np.mean(values == 1.0)) if values.size else None,
        "sentence_support_rate": r4(np.mean([x["supported"] for x in sentences])) if sentences else None,
        "distribution": [r4(v) for v in values.tolist()],
    }


# ── 7. Projection ─────────────────────────────────────────────────────────────
def fit_projection(vectors: np.ndarray, dims: int):
    import umap

    reducer = umap.UMAP(n_components=dims, n_neighbors=15, min_dist=UMAP_MIN_DIST, metric="cosine", random_state=SEED)
    coords = reducer.fit_transform(vectors)
    return reducer, coords


def normalise(coords: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    center = coords.mean(axis=0)
    scale = np.abs(coords - center).max()
    return (coords - center) / scale, (extra - center) / scale, {"center": center.tolist(), "scale": float(scale)}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)

    log("1/7 corpus")
    articles, chunks = build_corpus()
    log(f"  {len(articles)} articles, {len(chunks)} passages")

    log("2/7 evaluation set")
    generator = load_generator(QUESTION_GENERATOR)
    eval_items = build_eval_set(chunks, generator)
    curated = [{"question": q, "gold_articles": g, "showcase": i < SHOWCASE_COUNT}
               for i, (q, g) in enumerate(CURATED_QUESTIONS)]
    all_questions = [e["question"] for e in eval_items] + [c["question"] for c in curated]

    log("3/7 retrieval benchmark")
    benchmark = [{"key": "tfidf", "label": "TF-IDF (keyword baseline)", "dimensions": None,
                  **score_rankings(rank_tfidf(chunks, all_questions, EVAL_K), chunks, eval_items, curated)}]
    corpus_vectors, query_vectors = {}, {}
    for key, spec in EMBEDDERS.items():
        corpus_vectors[key] = embed_corpus(key, chunks)
        query_vectors[key] = embed_queries(key, all_questions)
        ranked = rank_dense(corpus_vectors[key], query_vectors[key], EVAL_K)
        benchmark.append({"key": key, "label": spec["label"], "model": spec["name"],
                          "dimensions": int(corpus_vectors[key].shape[1]),
                          **score_rankings(ranked, chunks, eval_items, curated)})
        log(f"  {spec['label']}: MRR@10={benchmark[-1]['mrr_at_10']}")
    dense = [b for b in benchmark if b["key"] != "tfidf"]
    best = max(dense, key=lambda b: (b["mrr_at_10"], -b["dimensions"]))
    selected = best["key"]
    for b in benchmark:
        b["selected"] = b["key"] == selected
    log(f"  selected {EMBEDDERS[selected]['label']}")
    vectors = corpus_vectors[selected]
    qvecs = query_vectors[selected]

    log("4/7 Chroma index")
    collection = build_chroma(chunks, vectors, selected)
    hits = chroma_search(collection, qvecs, EVAL_K)
    brute = rank_dense(vectors, qvecs, TOP_K)
    chroma_agreement = np.mean([
        [cid for cid, _ in h[:TOP_K]] == [chunks[i]["id"] for i in b] for h, b in zip(hits, brute, strict=False)])
    log(f"  Chroma top-{TOP_K} identical to exact search for {chroma_agreement:.1%} of queries")
    by_id = {c["id"]: c for c in chunks}
    index_of = {c["id"]: i for i, c in enumerate(chunks)}

    log("5/7 answers")
    retrieved = [[by_id[cid] for cid, _ in h[:TOP_K]] for h in hits]
    answers_by_config: dict[str, list[str]] = {}
    closed_by_generator: dict[str, list[str]] = {}
    for gkey in GENERATORS:
        log(f"  generator {GENERATORS[gkey]['label']}")
        gen = generator if gkey == QUESTION_GENERATOR else load_generator(gkey)
        for version in RAG_PROMPTS:
            answers_by_config[f"{gkey}/{version}"] = gen.generate(
                [rag_prompt(q, p, version) for q, p in zip(all_questions, retrieved, strict=False)], max_new_tokens=220)
        closed_by_generator[gkey] = gen.generate([closed_book_prompt(q) for q in all_questions], max_new_tokens=220)
        gen.close()
    del generator

    log("6/7 faithfulness")
    scorer = FaithfulnessScorer()
    passages = [[p["text"] for p in ps] for ps in retrieved]
    scores_by_config = {cfg: [scorer.score(a, ps) for a, ps in zip(answers, passages, strict=False)]
                        for cfg, answers in answers_by_config.items()}
    closed_scores_by_generator = {g: [scorer.score(a, ps) for a, ps in zip(answers, passages, strict=False)]
                                  for g, answers in closed_by_generator.items()}
    config_summaries = {cfg: summarize_faithfulness(sc) for cfg, sc in scores_by_config.items()}
    closed_summaries = {g: summarize_faithfulness(sc) for g, sc in closed_scores_by_generator.items()}
    selected_config = max(config_summaries, key=lambda c: config_summaries[c]["mean_faithfulness"] or 0)
    selected_generator, selected_prompt = selected_config.split("/")
    log(f"  mean faithfulness: { {c: s['mean_faithfulness'] for c, s in config_summaries.items()} }")
    log(f"  selected {selected_config}")
    rag_answers, rag_scores = answers_by_config[selected_config], scores_by_config[selected_config]
    closed_answers = closed_by_generator[selected_generator]
    closed_scores = closed_scores_by_generator[selected_generator]
    generator_name = GENERATORS[selected_generator]["name"]

    log("7/7 projection")
    showcase_idx = [len(eval_items) + i for i in range(SHOWCASE_COUNT)]
    reducer3, coords3 = fit_projection(vectors, 3)
    reducer2, coords2 = fit_projection(vectors, 2)
    q3 = reducer3.transform(qvecs)
    q2 = reducer2.transform(qvecs)
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness
    pca3 = PCA(n_components=3, random_state=SEED).fit_transform(vectors)
    fidelity = {
        "neighbors_k": 10,
        "trustworthiness": {
            "umap_3d": r4(trustworthiness(vectors, coords3, n_neighbors=10, metric="cosine")),
            "umap_2d": r4(trustworthiness(vectors, coords2, n_neighbors=10, metric="cosine")),
            "pca_3d": r4(trustworthiness(vectors, pca3, n_neighbors=10, metric="cosine")),
        },
    }
    true_top = [[index_of[cid] for cid, _ in h[:TOP_K]] for h in hits]
    overlap3 = [M.neighbor_overlap(t, M.nearest_indices(coords3, q, TOP_K)) for t, q in zip(true_top, q3, strict=False)]
    overlap2 = [M.neighbor_overlap(t, M.nearest_indices(coords2, q, TOP_K)) for t, q in zip(true_top, q2, strict=False)]
    fidelity["query_neighbor_overlap"] = {
        "k": TOP_K,
        "mean_overlap_3d": r4(np.mean(overlap3)), "mean_overlap_2d": r4(np.mean(overlap2)),
        "share_all_match_3d": r4(np.mean(np.array(overlap3) == TOP_K)),
        "share_none_match_3d": r4(np.mean(np.array(overlap3) == 0)),
        "queries": len(overlap3),
    }
    n3, q3n, norm3 = normalise(coords3, q3)
    n2, q2n, norm2 = normalise(coords2, q2)

    # ── Artifacts ──
    topics = list(TOPICS)
    article_names = [a["title"] for a in articles]
    write_json("corpus.json", {
        "source": "English Wikipedia", "license": LICENSE, "articles": articles,
        "passages": len(chunks), "max_passages_per_article": MAX_CHUNKS_PER_ARTICLE,
        "chunking": "Section-aware sentence packing, ~140 words (max 200), never crossing a section",
    })
    write_json("embedding-map.json", {
        "model": EMBEDDERS[selected]["name"], "dimensions": int(vectors.shape[1]),
        "projection": {"method": "UMAP", "metric": "cosine", "n_neighbors": 15, "min_dist": UMAP_MIN_DIST,
                       "random_state": SEED, "normalisation_3d": norm3, "normalisation_2d": norm2},
        "topics": topics, "articles": article_names,
        "fields": ["x", "y", "z", "x2", "y2", "topic", "article"],
        "points": [[r4(a[0]), r4(a[1]), r4(a[2]), r4(b[0]), r4(b[1]), topics.index(c["topic"]),
                    article_names.index(c["article"])] for a, b, c in zip(n3, n2, chunks, strict=False)],
        "ids": [c["id"] for c in chunks],
        "sections": [c["section"] for c in chunks],
        "previews": [c["text"][:220].rsplit(" ", 1)[0] + "…" if len(c["text"]) > 220 else c["text"] for c in chunks],
    })

    samples = []
    for rank_pos, qi in enumerate(showcase_idx):
        c = curated[rank_pos]
        near3 = M.nearest_indices(coords3, q3[qi], TOP_K)
        near2 = M.nearest_indices(coords2, q2[qi], TOP_K)
        samples.append({
            "question": c["question"], "gold_articles": c["gold_articles"],
            "query_3d": [r4(v) for v in q3n[qi]], "query_2d": [r4(v) for v in q2n[qi]],
            "retrieved": [{"rank": n + 1, "id": cid, "index": index_of[cid], "similarity": r4(sim),
                           "article": by_id[cid]["article"], "section": by_id[cid]["section"],
                           "topic": by_id[cid]["topic"], "text": by_id[cid]["text"]}
                          for n, (cid, sim) in enumerate(hits[qi][:TOP_K])],
            "nearest_in_3d": near3, "nearest_in_2d": near2,
            "overlap_3d": M.neighbor_overlap(true_top[qi], near3),
            "overlap_2d": M.neighbor_overlap(true_top[qi], near2),
            "answer": rag_answers[qi], "faithfulness": rag_scores[qi],
            "closed_book_answer": closed_answers[qi], "closed_book_faithfulness": closed_scores[qi],
        })
    write_json("sample-queries.json", {"top_k": TOP_K, "precomputed": True, "generator": generator_name,
                                       "prompt": selected_prompt,
                                       "nli_model": NLI_MODEL, "samples": samples})

    write_json("eval-set.json", {
        "generated": {"generator": GENERATORS[QUESTION_GENERATOR]["name"], "questions_per_article": QUESTIONS_PER_ARTICLE, "items": eval_items,
                      "mean_lexical_overlap": r4(np.mean([e["lexical_overlap"] for e in eval_items]))},
        "curated": curated,
    })
    write_json("retrieval-benchmark.json", {"top_k": TOP_K, "generated_questions": len(eval_items),
                                            "curated_questions": len(curated), "methods": benchmark})

    n = len(eval_items)
    generation = {
        "generator": generator_name, "nli_model": NLI_MODEL, "entailment_threshold": ENTAILMENT_THRESHOLD,
        "top_k": TOP_K, "premise": "max over 1-3 sentence windows of the retrieved passages",
        "selected_config": selected_config, "selected_generator": selected_generator,
        "selected_prompt": selected_prompt, "prompts": RAG_PROMPTS, "generators": GENERATORS,
        "by_config": {c: {"generator": c.split("/")[0], "prompt": c.split("/")[1],
                          **{k: x for k, x in sm.items() if k != "distribution"}} for c, sm in config_summaries.items()},
        "closed_book_by_generator": {g: {k: x for k, x in sm.items() if k != "distribution"}
                                     for g, sm in closed_summaries.items()},
        "rag": summarize_faithfulness(rag_scores),
        "closed_book": summarize_faithfulness(closed_scores),
        "rag_generated_only": summarize_faithfulness(rag_scores[:n]),
        "rag_curated_only": summarize_faithfulness(rag_scores[n:]),
        "gold_in_context_vs_faithfulness": {
            "gold_retrieved": summarize_faithfulness(
                [s for s, h, e in zip(rag_scores[:n], hits[:n], eval_items, strict=False) if e["gold_chunk"] in [x for x, _ in h[:TOP_K]]]),
            "gold_missed": summarize_faithfulness(
                [s for s, h, e in zip(rag_scores[:n], hits[:n], eval_items, strict=False) if e["gold_chunk"] not in [x for x, _ in h[:TOP_K]]]),
        },
    }
    for block in ("rag_generated_only", "rag_curated_only"):
        generation[block].pop("distribution")
    for block in generation["gold_in_context_vs_faithfulness"].values():
        block.pop("distribution")
    write_json("generation-eval.json", generation)
    write_json("projection-fidelity.json", fidelity)

    summary = {
        "project": "Steel QC knowledge assistant (embedding RAG)",
        "corpus": {"articles": len(articles), "passages": len(chunks), "topics": len(topics), "license": LICENSE},
        "embedding_model": EMBEDDERS[selected]["name"], "dimensions": int(vectors.shape[1]),
        "vector_db": "Chroma (HNSW, cosine)", "chroma_exact_agreement_top_k": r4(chroma_agreement),
        "generator": generator_name, "nli_model": NLI_MODEL,
        "retrieval": {k: best[k] for k in ("hit_at_1", "hit_at_4", "mrr_at_10", "curated_article_hit_at_5")},
        "tfidf_baseline": {k: benchmark[0][k] for k in ("hit_at_1", "hit_at_4", "mrr_at_10", "curated_article_hit_at_5")},
        "faithfulness": {"rag": generation["rag"]["mean_faithfulness"],
                         "closed_book": generation["closed_book"]["mean_faithfulness"],
                         "rag_abstention_rate": generation["rag"]["abstention_rate"],
                         "selected_prompt": selected_prompt,
                         "selected_config": selected_config,
                         "by_config": {c: sm["mean_faithfulness"] for c, sm in config_summaries.items()}},
        "projection": {"trustworthiness_3d": fidelity["trustworthiness"]["umap_3d"],
                       "mean_query_overlap_3d": fidelity["query_neighbor_overlap"]["mean_overlap_3d"], "k": TOP_K},
        "eval_questions": {"generated": len(eval_items), "curated": len(curated)},
    }
    write_json("summary.json", summary)
    log("done")


if __name__ == "__main__":
    main()
