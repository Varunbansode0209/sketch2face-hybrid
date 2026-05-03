import os 
import urllib.request

os.makedirs("models/onnx", exist_ok=True)

models = {
    "retinaface.onnx": "https://insightface.ai/files/models/det_10g.onnx",
    "arcface_r100.onnx": "https://insightface.ai/files/models/arcface_r100.onnx",
}



for name, url in models.items():
	print(f"Downloading {name} from InsightFace CDN...")
	urllib.request.urlretrieve(url, f"models/onnx/{name}")
	print(f"{name} downloaded succussfully")