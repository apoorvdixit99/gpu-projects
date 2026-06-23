import time
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer: GPT2Tokenizer | None = None
model: GPT2LMHeadModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16)
    model = model.to("cuda")
    model.eval()
    print("Model loaded on GPU.")
    yield
    del model
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


class GenerateResponse(BaseModel):
    generated_text: str
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return GenerateResponse(generated_text=generated, latency_ms=latency_ms)


@app.get("/health")
async def health():
    return {"status": "ok"}
