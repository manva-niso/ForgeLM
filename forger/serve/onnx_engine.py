"""ONNX export + ONNX Runtime engine.

Exports the GPT's full forward (prefill/recompute decode) to a static graph;
the ORT engine generates with a growing sequence (no KV-cache in the graph -
correct and portable; the torch Engine keeps the cached path for comparison).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from forger.model.checkpoint import load_model_from_checkpoint
from forger.model.gpt import GPT
from forger.tokenizer.bpe import BPETokenizer


def export_onnx(
    model: GPT,
    output_path: str | Path,
    context_length: int | None = None,
    opset: int = 17,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = context_length or model.config.context_length
    model.eval()
    x = torch.zeros((1, ctx), dtype=torch.long)
    torch.onnx.export(
        model,
        (x,),
        str(out),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return out


def quantize_onnx_int8(model_path: str | Path, output_path: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(output_path)
    quantize_dynamic(
        str(model_path),
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    return out


class ORTEngine:
    def __init__(self, session_path: str | Path, tokenizer: BPETokenizer) -> None:
        self.session = ort.InferenceSession(
            str(session_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizer
        self.vocab = session_path

    @torch.inference_mode()
    def _forward(self, ids: list[int]) -> np.ndarray:
        x = np.array([ids], dtype=np.int64)
        out = self.session.run(["logits"], {"input_ids": x})[0]
        return out[0]  # (T, V)

    def sample(self, logits_row: np.ndarray, top_k: int | None, temperature: float, rng: np.random.Generator) -> int:
        logits = logits_row.astype(np.float64)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size)
            idx = np.argpartition(logits, -k)[-k:]
            keep = np.full_like(logits, -np.inf)
            keep[idx] = logits[idx]
            logits = keep
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return int(rng.choice(probs.size, p=probs))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        top_k: int = 50,
        temperature: float = 0.8,
        stop_id: int = 0,
        seed: int | None = None,
    ) -> tuple[str, list[int], dict[str, float]]:
        rng = np.random.default_rng(seed)
        ids = self.tokenizer.encode(prompt)
        start = time.monotonic()
        while len(ids) - len(self.tokenizer.encode(prompt)) < max_tokens:
            logits = self._forward(ids)
            next_id = self.sample(logits[-1], top_k, temperature, rng)
            if stop_id is not None and next_id == stop_id:
                break
            ids.append(next_id)
        elapsed = time.monotonic() - start
        generated = len(ids) - len(self.tokenizer.encode(prompt))
        return (
            self.tokenizer.decode(ids),
            ids,
            {"generated_tokens": generated, "tokens_per_sec": generated / elapsed if elapsed else 0.0},
        )

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        export_dir: str,
        tokenizer_dir: str = "artifacts/tokenizer",
        int8: bool = False,
    ) -> ORTEngine:
        model = load_model_from_checkpoint(ckpt_dir)
        export_path = Path(export_dir)
        onnx_path = export_path / "model.onnx"
        final_path = export_path / ("model_int8.onnx" if int8 else "model.onnx")
        if not final_path.exists():
            export_onnx(model, onnx_path)
            if int8:
                quantize_onnx_int8(onnx_path, final_path)
        return cls(final_path, BPETokenizer.load(tokenizer_dir))