import gzip
import itertools
import json
import os
import tqdm
from typing import Union, List
from collections import defaultdict

import requests
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from fim_eval.execution import check_correctness
from fim_eval.load_problems import Problem
from fim_eval.result import Result as Sample

MAX_WORKERS = 16
K = [1, 3, 5]

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


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


if __name__ == "__main__":
    download_eval()
    problems = load_eval()
    samples = load_samples()
    problem_by_id = {problem.task_id: problem for problem in problems}

    results_by_id = defaultdict(list)
    flat_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, sample in enumerate(samples):
            future = executor.submit(
                check_correctness,
                problem_by_id[sample.task_id].model_dump(),
                sample.completion,
                10,
                completion_id=i,
            )
            futures.append(future)

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            task_id = future.result()["task_id"]
            results_by_id[task_id].append(future.result())
            flat_results.append(future.result())

    passed = sum(result["passed"] for result in flat_results)
    print(f"Accuracy: {passed / len(flat_results)} = {passed} / {len(flat_results)}")

    # Debug some number of failed results
    # num_failed = 0
    # for result in flat_results:
    #     if not result["passed"]:
    #         sample = samples[result["completion_id"]]
    #         print(f"task id failed: {result['task_id']}")
    #         num_failed += 1
    #         problem = problem_by_id[result["task_id"]]
    #         console.print(
    #             "-" * 30,
    #             "\n",
    #             f"{problem.prompt}[yellow on grey23]{sample.completion}[/yellow on grey23]{problem.suffix}",
    #             "\n",
    #             "-" * 30,
    #         )
    #     if num_failed >= 10:
    #         break

    num_attempts_per_problem = [len(results) for results in results_by_id.values()]
    num_successes_per_problem = [
        sum(result["passed"] for result in results)
        for results in results_by_id.values()
    ]
    for k in K:
        pass_at_k = estimate_pass_at_k(
            num_attempts_per_problem,
            num_successes_per_problem,
            k=k,
        ).mean()
        print(f"Pass@{k}: {pass_at_k}")
