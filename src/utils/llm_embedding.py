from typing import Dict, Tuple, Optional, List, Union
import threading
import time
import math
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

_MODEL_CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}
_MODEL_LOCK = threading.Lock()
_DEFAULT_KEY: Optional[Tuple[str, str]] = None

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_model(model_id: str, device: Optional[str] = None) -> SentenceTransformer:
    device = device or pick_device()
    key = (model_id, device)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = SentenceTransformer(model_id, device=device)
        return _MODEL_CACHE[key]

def preload_model(model_id: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None) -> SentenceTransformer:
    inst = get_model(model_id, device)
    dev = device or pick_device()
    with _MODEL_LOCK:
        global _DEFAULT_KEY
        _DEFAULT_KEY = (model_id, dev)
    return inst

def get_default_model() -> SentenceTransformer:
    with _MODEL_LOCK:
        if _DEFAULT_KEY is not None and _DEFAULT_KEY in _MODEL_CACHE:
            return _MODEL_CACHE[_DEFAULT_KEY]
    return preload_model()

def call_embedding(
    inputs: Union[str, List[str]],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> List[List[float]]:
    payload = [inputs] if isinstance(inputs, str) else inputs
    st_model = get_model(model)
    emb = st_model.encode(payload, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return emb.tolist()

def gen_texts(n: int, avg_chars: int) -> List[str]:
    base = "这是一个用于吞吐量测试的中文句子，包含多种常见词汇与标点。"
    texts: List[str] = []
    for _ in range(n):
        repeat = max(1, math.floor(avg_chars / len(base)))
        texts.append((base * repeat)[:avg_chars])
    return texts

def _count_tokens(tokenizer: AutoTokenizer, batch: List[str], max_length: Optional[int] = None) -> int:
    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return int(enc["attention_mask"].sum().item())

def measure_st_throughput(
    texts: List[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    max_length: int = 512,
) -> Tuple[float, int, float]:
    st_model = get_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    total_tokens = 0
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = _count_tokens(tokenizer, batch, max_length=max_length)
        total_tokens += toks
        st_model.encode(batch, batch_size=batch_size, convert_to_numpy=False, show_progress_bar=False)
    elapsed = time.time() - start
    return total_tokens / elapsed if elapsed > 0 else 0.0, total_tokens, elapsed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_texts", type=int, default=64)
    parser.add_argument("--avg_chars", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    texts = gen_texts(args.num_texts, args.avg_chars)
    tps, tokens, sec = measure_st_throughput(texts, model=args.model, batch_size=args.batch_size, max_length=args.max_length)
    print(f"device={pick_device()} model={args.model}")
    print(f"tokens/s={tps:.2f} tokens={tokens} time={sec:.2f}s")
