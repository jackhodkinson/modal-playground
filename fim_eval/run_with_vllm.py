import time

import modal

from fim_eval.constants import MODEL_VOLUME, MODELS_DIR
from fim_eval.app import app

volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "transformers", "accelerate", "torch", "vllm"
)

with image.imports():
    from vllm import LLM, SamplingParams


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),
    # gpu=modal.gpu.L40S(count=1),
    volumes={MODELS_DIR: volume},
)
def run_with_vllm(model_name: str, prompts: list[str]):
    print(f"Running {model_name}")
    t0 = time.time()

    num_prompts = len(prompts)
    print(f"Running {num_prompts} prompts")

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512)
    model_path = MODELS_DIR + "/" + model_name
    llm = LLM(model=model_path, trust_remote_code=True)
    t1 = time.time()
    print(f"Model loaded in {t1 - t0} seconds")

    outputs = llm.generate(prompts, sampling_params)
    # for i, output in enumerate(outputs):
    #     output_text = output.outputs[0].text
    #     print(f"Completion [{i}/{num_prompts}]: ===\n{output_text}\n===")

    t2 = time.time()
    print(f"Inference complete in {t2 - t1} seconds")

    t3 = time.time()
    print(f"Total time taken to run {model_name}: {t3 - t0} seconds")

    return [o.outputs[0].text for o in outputs]
