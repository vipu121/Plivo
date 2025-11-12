from typing import List
import numpy as np
import re

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None


class PseudoLikelihoodRanker:
    def __init__(self, model_name="distilbert-base-uncased",
                 onnx_path=None, device="cpu", max_length=48):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.onnx = None
        self.torch_model = None

        if onnx_path and ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.onnx = ort.InferenceSession(
                onnx_path, sess_options=opts,
                providers=["CPUExecutionProvider"])
        elif AutoModelForMaskedLM is not None:
            self.torch_model = AutoModelForMaskedLM.from_pretrained(
                model_name).to(device).eval()
        else:
            raise RuntimeError("No ONNXRuntime or Transformers available")

    # -------------------------------------------------
    # Utility: very light log-softmax in NumPy
    # -------------------------------------------------
    @staticmethod
    def _log_softmax_np(x: np.ndarray) -> np.ndarray:
        m = np.max(x, axis=-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

    # -------------------------------------------------
    # Core scoring (single text to keep shape [1,L])
    # -------------------------------------------------
    def _score_one_onnx(self, text: str) -> float:
        toks = self.tokenizer(
            text, return_tensors="np", truncation=True, max_length=self.max_length)
        ort_inputs = {
            "input_ids": toks["input_ids"].astype(np.int64),
            "attention_mask": toks["attention_mask"].astype(np.int64)
        }
        logits = self.onnx.run(None, ort_inputs)[0]  # [1,L,V]
        attn = toks["attention_mask"][0]
        seq = toks["input_ids"][0]
        L = int(attn.sum())
        lp = self._log_softmax_np(logits[0, :L, :])
        return float(lp[np.arange(L), seq[:L]].mean())

    # -------------------------------------------------
    # Torch fallback (rarely used if ONNX present)
    # -------------------------------------------------
    def _score_one_torch(self, text: str) -> float:
        toks = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            out = self.torch_model(**toks).logits
            lp = torch.log_softmax(out, dim=-1)
            seq = toks["input_ids"][0]
            attn = toks["attention_mask"][0]
            L = int(attn.sum())
            val = lp[0, :L, :].gather(-1, seq[:L, None]).squeeze(-1)
            return float((val * attn[:L]).sum() / attn[:L].sum())

    # -------------------------------------------------
    # Fast selective scoring
    # -------------------------------------------------
    def score(self, candidates: List[str]) -> List[float]:
        """Score minimal subset of candidates."""
        scores = []
        for c in candidates:
            if self.onnx is not None:
                scores.append(self._score_one_onnx(c))
            else:
                scores.append(self._score_one_torch(c))
        return scores

    def choose_best(self, candidates: List[str]) -> str:
        """Return best candidate, short-circuiting obvious ones."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        # 🚀 Short-circuit rules (cheap and 80% of cases)
        for cand in candidates:
            # valid email
            if re.search(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b", cand):
                return cand
            # contains ₹ and digits (currency)
            if "₹" in cand and re.search(r"\d", cand):
                return cand
            # proper punctuation and reasonable length → good enough
            if cand.endswith((".", "?", "!")) and len(cand) < 80:
                return cand

        # Fallback to ONNX scoring for rare ambiguous ones
        sc = self.score(candidates)
        return candidates[int(np.argmax(sc))]
