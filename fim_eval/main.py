from fim_eval.app import app
from fim_eval.download_model import download_model
from fim_eval.run_with_transformers import run_with_transformers
from fim_eval.run_with_vllm import run_with_vllm

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
    download_model.remote(MODEL_NAME)
    run_with_transformers.remote(MODEL_NAME, prompt)
    run_with_vllm.remote(MODEL_NAME, prompt)
