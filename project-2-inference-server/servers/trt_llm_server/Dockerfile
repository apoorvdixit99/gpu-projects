FROM nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY server.py /app/server.py
WORKDIR /app
