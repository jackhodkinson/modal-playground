import random
import time

import modal

from fim_eval.app import app
from fim_eval.download_eval import download_eval
from fim_eval.download_model import download_model
from fim_eval.run_with_transformers import run_with_transformers
from fim_eval.run_with_vllm import run_with_vllm
from fim_eval.constants import DATA_DIR, EVAL_VOLUME
from fim_eval.load_problems import load_problems, Problem

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"

prompt = """<｜fim▁begin｜>def quick_sort(arr):
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


@app.local_entrypoint()
def main():
    load_and_solve_problems.remote()
    # download_eval.remote()
    # download_model.remote(MODEL_NAME)
    # run_with_transformers.remote(MODEL_NAME, prompt)
    # run_with_vllm.remote(MODEL_NAME, prompt)


image = modal.Image.debian_slim().pip_install("requests", "pydantic")

vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)


@app.function(image=image, volumes={DATA_DIR: vol})
def load_and_solve_problems() -> list[Problem]:
    t0 = time.time()
    problems = load_problems()

    for problem in problems[:10]:
        run_with_transformers.remote(MODEL_NAME, problem.prompt)

    tf = time.time()
    print(f"Time taken: {tf - t0}")
