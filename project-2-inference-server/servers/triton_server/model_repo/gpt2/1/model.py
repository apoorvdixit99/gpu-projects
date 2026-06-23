import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda"
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(
            "gpt2", torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt_np = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
            prompt = prompt_np[0].decode("utf-8")

            max_new_tokens_np = pb_utils.get_input_tensor_by_name(
                request, "max_new_tokens"
            ).as_numpy()
            max_new_tokens = int(max_new_tokens_np[0])

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )

            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

            out_tensor = pb_utils.Tensor(
                "generated_text",
                np.array([generated.encode("utf-8")], dtype=object),
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        del self.model
        torch.cuda.empty_cache()
