# model-graph demo deployment: full UI + OpenAI-compatible API on the mock
# backend. Deliberately torch-free and model-free — the image stays ~60 MB
# and serves tiny gzipped responses to keep PaaS egress (cost) minimal.
# Browser engines still do real WebGPU inference client-side, with weights
# fetched from huggingface.co (zero egress from this service).
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir aiohttp websockets

COPY api_server.py server.py curator.py ./
COPY web ./web
COPY vault ./vault
COPY suites ./suites
COPY CONTRIBUTING.md ./

ENV PORT=8080 HOST=0.0.0.0
EXPOSE 8080
CMD ["python", "api_server.py", "--mock", "--no-curate"]
