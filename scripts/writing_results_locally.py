import modal
import os
from pydantic import BaseModel

image = (
    modal.Image.debian_slim()
    .add_local_dir("./assets", remote_path="/root/assets", copy=True)
    .pip_install("pydantic")
)

app = modal.App("example-hello-world")


class Task(BaseModel):
    id: int
    value: int


class Result(BaseModel):
    id: int
    value: int


@app.function(image=image)
def f(task: Task) -> Result:
    print(task)
    return Result(id=task.id, value=task.id * task.value)


@app.local_entrypoint()
def main():
    task = Task(id=1, value=1)
    f.local(task)
    print("Finished local execution")

    # run the function remotely on Modal
    task = Task(id=2, value=2)
    f.remote(task)
    print("Finished remote execution")

    tasks = []
    with open("./assets/example.jsonl", "r") as file:
        for line in file:
            task = Task.model_validate_json(line)
            tasks.append(task)

    results = f.map(tasks)
    print("Finished remote map execution")

    # if ./data directory does not exist, create it
    if not os.path.exists("./data"):
        os.makedirs("./data")

    with open("./data/results.jsonl", "w") as file:
        for result in results:
            print(result)
            file.write(result.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
