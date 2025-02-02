import os

import modal

from fim_eval.app import app
from fim_eval.constants import DATA_DIR, EVAL_VOLUME

vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)

image = modal.Image.debian_slim().pip_install("requests", "pydantic")

with image.imports():
    import requests


@app.function(volumes={DATA_DIR: vol}, image=image)
def download_eval():
    url = "https://raw.githubusercontent.com/openai/human-eval-infilling/88062ff9859c875d04db115b698ed4b0f0395170/data/HumanEval-SingleLineInfilling.jsonl.gz"
    remote_path = "HumanEval-SingleLineInfilling.jsonl.gz"

    full_path = os.path.join("/data", remote_path)

    # Check if file exists using regular file operations
    if os.path.exists(full_path):
        print(f"File {remote_path} already exists in volume")
        return

    print(f"Downloading {url} to volume...")
    response = requests.get(url)
    response.raise_for_status()

    with open(full_path, "wb") as f:
        f.write(response.content)
    print("Download complete")

    vol.commit()
