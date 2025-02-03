import gzip
import json
import os

from fim_eval.execution import check_correctness
from fim_eval.load_problems import Problem
from fim_eval.result import Result as Sample

import requests
from rich.console import Console

console = Console()


def download_eval():
    url = "https://raw.githubusercontent.com/openai/human-eval-infilling/88062ff9859c875d04db115b698ed4b0f0395170/data/HumanEval-SingleLineInfilling.jsonl.gz"

    full_path = os.path.join(
        os.getcwd(), "data", "HumanEval-SingleLineInfilling.jsonl.gz"
    )

    # Check if file exists using regular file operations
    if os.path.exists(full_path):
        print(f"File {full_path} already exists")
        return

    response = requests.get(url)
    response.raise_for_status()

    with open(full_path, "wb") as f:
        f.write(response.content)


def load_eval() -> list[Problem]:
    problems: list[Problem] = []

    with gzip.open(
        os.path.join(os.getcwd(), "data", "HumanEval-SingleLineInfilling.jsonl.gz"),
        "rb",
    ) as f:
        for line in f:
            problems.append(Problem(**json.loads(line)))

    return problems


def load_samples() -> list[Sample]:
    samples: list[Sample] = []

    with open(os.path.join(os.getcwd(), "data", "results.jsonl"), "r") as f:
        for line in f:
            samples.append(Sample(**json.loads(line)))

    return samples


def evaluate_results(samples: list[Sample], problems: list[Problem]):
    print(f"Evaluating {len(samples)} samples and {len(problems)} problems...")


if __name__ == "__main__":
    download_eval()
    problems = load_eval()
    samples = load_samples()
    problem_by_id = {problem.task_id: problem for problem in problems}

    results = []
    for sample in samples:
        problem = problem_by_id[sample.task_id]
        result = check_correctness(problem.model_dump(), sample.completion, 10)
        results.append(result)

        if not result["passed"]:
            console.print(
                "-" * 30,
                "\n",
                f"{problem.prompt}[yellow on grey23]{sample.completion}[/yellow on grey23]{problem.suffix}",
                "\n",
                "-" * 30,
            )

    print(f"Accuracy: {sum(result['passed'] for result in results) / len(results)}")
