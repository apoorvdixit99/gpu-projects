FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN pip install --no-cache-dir \
    "torch==2.5.1+cu124" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124

COPY model_repo /models
