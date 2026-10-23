# Triage service. No torch: MiniLM runs through ONNX Runtime, so the image stays small and fits 512 MB.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GD_EMBED=onnx GD_BUNDLE=models/v1 HF_HOME=/opt/hf
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY guarded_distillation guarded_distillation
RUN pip install --no-deps -e .
# fetch the encoder at build time so a cold start does not download it
RUN python -c "from guarded_distillation.onnx_embed import OnnxEncoder; OnnxEncoder()"
COPY models models
COPY labels/claude-haiku-5-5.jsonl labels/claude-haiku-5-5.jsonl
COPY data/fewshot.csv data/fewshot.csv
EXPOSE 8000
CMD ["sh", "-c", "gdistill serve --port ${PORT:-8000}"]
