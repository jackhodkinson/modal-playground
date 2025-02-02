import time
import modal
import os

VOLUME_NAME = "deepseek"
MODELS_DIR = "/models"
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
DOWNLOAD_TIMEOUT = 4 * 60 * 60  # 4 hours (in seconds)

app = modal.App("deepseek-fim")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with download_image.imports():
    from huggingface_hub import snapshot_download


@app.function(
    volumes={MODELS_DIR: volume},
    timeout=DOWNLOAD_TIMEOUT,
    image=download_image,
)
def download_model():
    volume.reload()
    snapshot_download(
        MODEL_NAME,
        local_dir=MODELS_DIR + "/" + MODEL_NAME,
    )
    volume.commit()


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
def run_model():
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    model_path = MODELS_DIR + "/" + MODEL_NAME
    llm = LLM(model=model_path)
    input_text = """<｜fim▁begin｜>def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = []
        right = []
    <｜fim▁hole｜>
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>"""

    outputs = llm.generate([input_text], sampling_params)
    print(outputs[0].outputs[0].text)


@app.function(image=image, volumes={MODELS_DIR: volume})
def check_model():
    # ls
    print(os.listdir(MODELS_DIR))
    print(os.listdir(MODELS_DIR + "/" + MODEL_NAME))


@app.local_entrypoint()
def main():
    t0 = time.time()
    print("Downloading deepseek-coder-1.3b-base")
    # download_model.remote()
    t1 = time.time()
    print(f"Downloading deepseek-coder-1.3b-base complete in {t1 - t0} seconds")

    print("Running deepseek-coder-1.3b-base")
    run_model.remote()
    t2 = time.time()
    print(f"Running deepseek-coder-1.3b-base complete in {t2 - t1} seconds")
    # Running deepseek-coder-1.3b-base complete in 75.16418719291687 seconds

    print(f"Total time taken: {t2 - t0} seconds")
