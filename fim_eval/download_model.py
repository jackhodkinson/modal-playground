import modal
from fim_eval.app import app
from fim_eval.constants import MODEL_VOLUME, MODELS_DIR
import time

DOWNLOAD_TIMEOUT = 4 * 60 * 60  # 4 hours (in seconds)


volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)

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
def download_model(model_name: str):
    t0 = time.time()
    volume.reload()
    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
    )
    volume.commit()
    t1 = time.time()
    print(f"Downloading {model_name} complete in {t1 - t0} seconds")
