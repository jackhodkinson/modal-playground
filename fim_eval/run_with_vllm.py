import time
import modal

from fim_eval.constants import VOLUME_NAME, MODELS_DIR
from fim_eval.app import app

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "transformers", "accelerate", "torch", "vllm"
)

with image.imports():
    from vllm import LLM, SamplingParams


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),
    volumes={MODELS_DIR: volume},
)
def run_with_vllm(model_name: str, prompt: str):
    print(f"Running {model_name}")
    t0 = time.time()

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    model_path = MODELS_DIR + "/" + model_name
    llm = LLM(model=model_path)
    t1 = time.time()
    print(f"Model loaded in {t1 - t0} seconds")

    outputs = llm.generate([prompt], sampling_params)
    print(outputs[0].outputs[0].text)

    t2 = time.time()
    print(f"Inference complete in {t2 - t1} seconds")

    t3 = time.time()
    print(f"Total time taken to run {model_name}: {t3 - t0} seconds")
