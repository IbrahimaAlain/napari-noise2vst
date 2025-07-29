import os
import urllib.request

# Points to the folder containing this script, i.e., pretrained_weights/
folder = os.path.dirname(__file__)
url = "https://github.com/cszn/KAIR/releases/download/v1.0/"
models = ["ffdnet_gray.pth", "ffdnet_color.pth", "drunet_gray.pth", "drunet_color.pth"]

os.makedirs(folder, exist_ok=True)

for model in models:
    path = os.path.join(folder, model)
    if not os.path.isfile(path):
        print(f"Downloading {model}...")
        urllib.request.urlretrieve(url + model, path)
        print(f"{model} downloaded.")
    else:
        print(f"{model} already exists.")
