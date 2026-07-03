import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from tensorrt_llm import LLM, SamplingParams

llm: LLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # First run compiles GPT-2 into TRT engines (~2-3 min); cached on subsequent runs.
    llm = LLM(model="gpt2", dtype="float16", tokenizer=tokenizer)
    print("TRT-LLM engine ready.")
    yield
    del llm


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class GenerateResponse(BaseModel):
    generated_text: str
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    params = SamplingParams(
        max_new_tokens=req.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    t0 = time.perf_counter()
    outputs = llm.generate([req.prompt], params)
    latency_ms = (time.perf_counter() - t0) * 1000

    generated_tokens = outputs[0].outputs[0].text
    return GenerateResponse(
        generated_text=req.prompt + generated_tokens,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
