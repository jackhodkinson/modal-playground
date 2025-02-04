import os
import time
import json

import modal

from fim_eval.app import app
from fim_eval.download_eval import download_eval
from fim_eval.download_model import download_model
from fim_eval.constants import DATA_DIR, EVAL_VOLUME
from fim_eval.load_problems import load_problems, Problem
from fim_eval.result import Result
from fim_eval.run_with_vllm import run_with_vllm
# from fim_eval.run_with_transformers import run_with_transformers

SAMPLES_PER_PROBLEM = 5
# MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"

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
    download_eval.remote()
    download_model.remote(MODEL_NAME)

    results = load_and_solve_problems.remote()

    # Write to a local file
    path = os.path.join(os.getcwd(), "data", "results.jsonl")
    with open(path, "w") as f:
        for result in results:
            line = json.dumps(result.model_dump())
            f.write(line + "\n")


image = (
    modal.Image.debian_slim()
    .pip_install("requests", "pydantic")
    .add_local_python_source("fim_eval")
)

vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)


def construct_prompt(problem: Problem) -> str:
    return f"<｜fim▁begin｜>{problem.prompt}<｜fim▁hole｜>{problem.suffix}<｜fim▁end｜>"


@app.function(image=image, volumes={DATA_DIR: vol}, timeout=1200)
def load_and_solve_problems() -> list[Result]:
    t0 = time.time()
    problems: list[Problem] = load_problems()

    prompts = [construct_prompt(problem) for problem in problems]

    # multiple samples per prompt to account for temperature effects
    prompts = [prompt for prompt in prompts for _ in range(SAMPLES_PER_PROBLEM)]
    repeated_problems = [
        problem for problem in problems for _ in range(SAMPLES_PER_PROBLEM)
    ]

    # Running with vanilla transformers is too slow
    # completions = run_with_transformers.remote(MODEL_NAME, prompts)
    # 23.75 seconds (10 problems)
    # 123.13 seconds (50 problems)
    # Timed out at 300s (100 problems)

    # Running with vllm is faster
    completions = run_with_vllm.remote(MODEL_NAME, prompts)
    # 35.16 seconds (10 problems)
    # 35.31 seconds (100 problems)

    results = [
        Result(task_id=problem.task_id, completion=completion)
        for problem, completion in zip(repeated_problems, completions)
    ]

    # Write to a volume
    with open(f"{DATA_DIR}/results.jsonl", "w") as f:
        for result in results:
            line = json.dumps(result.model_dump())
            f.write(line + "\n")

    tf = time.time()
    print(f"Time taken: {tf - t0}")

    return results
