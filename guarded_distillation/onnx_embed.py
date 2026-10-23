"""MiniLM sentence embeddings through ONNX Runtime: the same vectors as sentence-transformers
(mean pooling, L2 norm) without torch, so the service fits in a 512 MB container.
Enabled with GD_EMBED=onnx. Files are fetched once from the Hugging Face Hub."""
from __future__ import annotations

import numpy as np

REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 256


class OnnxEncoder:
    def __init__(self, repo: str = REPO):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False      # keeps resident memory low in a 512 MB container
        opts.enable_mem_pattern = False
        self.sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for a in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[a:a + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
        return np.vstack(out).astype(np.float32)
