import gzip
import json
import os
import random
import time

import modal
from pydantic import BaseModel

app = modal.App("fim-eval-data")

vol = modal.Volume.from_name("fim-eval-data", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("requests", "pydantic")

with image.imports():
    import requests


@app.function(volumes={"/data": vol}, image=image)
def download_to_volume():
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


class Problem(BaseModel):
    task_id: str
    prompt: str
    suffix: str
    canonical_solution: str
    test: str
    entry_point: str


def load_problems() -> list[Problem]:
    problems: list[Problem] = []

    with gzip.open("/data/HumanEval-SingleLineInfilling.jsonl.gz", "rb") as f:
        for line in f:
            problems.append(Problem(**json.loads(line)))

    return problems


def solve_problem(problem: Problem) -> Problem:
    print("Solving problem: ", problem.task_id)

    # Artificial delay
    time.sleep(random.randint(0, 10))

    return problem


@app.function(image=image, volumes={"/data": vol})
def load_and_solve_problems() -> list[Problem]:
    problems = load_problems()
    # TODO: Parallelize this
    results = [solve_problem(problem) for problem in problems]
    print(f"Solved all problems, got {len(results)} results")


@app.local_entrypoint()
def main():
    download_to_volume.remote()
    load_and_solve_problems.remote()
