import time
import modal

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


image = modal.Image.debian_slim().pip_install("transformers", "accelerate", "torch")

with image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),
    volumes={MODELS_DIR: volume},
)
def run_model():
    model_path = MODELS_DIR + "/" + MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).cuda()
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
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(input_text) :])


@app.local_entrypoint()
def main():
    t0 = time.time()
    print("Downloading deepseek-coder-1.3b-base")
    download_model.remote()
    t1 = time.time()
    print(f"Downloading deepseek-coder-1.3b-base complete in {t1 - t0} seconds")

    print("Running deepseek-coder-1.3b-base")
    run_model.remote()
    t2 = time.time()
    print(f"Running deepseek-coder-1.3b-base complete in {t2 - t1} seconds")
    # Running deepseek-coder-1.3b-base complete in 26.479429960250854 seconds

    print(f"Total time taken: {t2 - t0} seconds")
