import gzip
import json
import os

import modal
from pydantic import BaseModel

from fim_eval.constants import DATA_DIR, EVAL_VOLUME

image = modal.Image.debian_slim().pip_install("requests", "pydantic")

vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)


class Problem(BaseModel):
    task_id: str
    prompt: str
    suffix: str
    canonical_solution: str
    test: str
    entry_point: str


def load_problems() -> list[Problem]:
    problems: list[Problem] = []

    with gzip.open(
        os.path.join(DATA_DIR, "HumanEval-SingleLineInfilling.jsonl.gz"), "rb"
    ) as f:
        for line in f:
            problems.append(Problem(**json.loads(line)))

    return problems
