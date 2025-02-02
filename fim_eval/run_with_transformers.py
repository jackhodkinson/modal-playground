import time
import modal
from fim_eval.constants import MODEL_VOLUME, MODELS_DIR
from fim_eval.app import app

volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)


image = modal.Image.debian_slim().pip_install("transformers", "accelerate", "torch")

with image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),
    volumes={MODELS_DIR: volume},
)
def run_with_transformers(model_name: str, prompt: str):
    print(f"Running {model_name}")
    t0 = time.time()

    model_path = MODELS_DIR + "/" + model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    t1 = time.time()
    print(f"Tokenizer loaded in {t1 - t0} seconds")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).cuda()
    t2 = time.time()
    print(f"Model loaded in {t2 - t1} seconds")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=1024)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = decoded_output[len(prompt) :]
    print(f"Completion: ===\n{completion}\n===")
    t3 = time.time()
    print(f"Inference complete in {t3 - t2} seconds")

    t4 = time.time()
    print(f"Total time taken to run {model_name}: {t4 - t0} seconds")
