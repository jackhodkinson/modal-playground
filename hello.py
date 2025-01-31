#!/usr/bin/env -S uvx --with-requirements pyproject.toml modal run

import sys
import os

import modal

# An example of installing the entire pyproject.toml dependencies
# Probably a bad idea since you don't need `modal` for remote execution.
# This defeats the purpose of dependency isolation.
image = (
    modal.Image.debian_slim()
    .add_local_file("./pyproject.toml", remote_path="/root/", copy=True)
    .add_local_dir("./assets", remote_path="/root/assets", copy=True)
    .pip_install("uv")
    .run_commands(
        "cd root; uv pip compile pyproject.toml -o requirements.txt; pip install -r requirements.txt"
    )
)

# Note: we make this availiable locally via `pyproject.toml`
with image.imports():
    import numpy as np

app = modal.App("example-hello-world")


@app.function(image=image)
def f(i):
    locally_or_remotely = "locally" if modal.is_local() else "remotely"
    print(f"Running {locally_or_remotely} from cwd: {os.getcwd()}")

    # Print ls
    print(os.listdir("."))

    # Print contents of hello.txt
    with open("./assets/hello.txt", "r") as f:
        print(f"Contents of hello.txt: {f.read()}")

    arr = np.repeat(i, 10)
    if i % 2 == 0:
        print("hello", arr)
    else:
        print("world", arr, file=sys.stderr)

    return i * i


@app.local_entrypoint()
def main():
    # run the function locally
    total = 10
    total += f.local(total)
    print("Finished local execution: ", total)

    # run the function remotely on Modal
    total += f.remote(total)
    print("Finished remote execution: ", total)

    # run the function in parallel and remotely on Modal
    # total = 0
    # for ret in f.map(range(200)):
    #     total += ret

    print(total)


if __name__ == "__main__":
    main()
