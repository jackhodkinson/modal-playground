import modal

app = modal.App("demo-volumes")

vol = modal.Volume.from_name("demo-volumes", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("requests")

with image.imports():
    import requests


@app.function(volumes={"/data": vol}, image=image)
def run():
    result = requests.get("https://www.modal.com")
    with open("/data/xyz.txt", "w") as f:
        f.write(result.text)
    vol.commit()  # Needed to make sure all changes are persisted before exit


@app.local_entrypoint()
def main():
    run.remote()
