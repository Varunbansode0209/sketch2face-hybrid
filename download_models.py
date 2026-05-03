import urllib.request
import os

os.makedirs("models/onnx", exist_ok=True)

models = {
    "retinaface.onnx": "https://raw.githubusercontent.com/deepinsight/insightface/master/model_zoo/det_10g.onnx",
    "arcface_r100.onnx": "https://raw.githubusercontent.com/deepinsight/insightface/master/model_zoo/arcface_r100.onnx",
}

for name, url in models.items():
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, f"models/onnx/{name}")
    print(f"{name} downloaded")
