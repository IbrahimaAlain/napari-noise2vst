import os
import urllib.request

# Point vers le dossier contenant ce script, donc pretrained_weights/
folder = os.path.dirname(__file__)
url = "https://github.com/cszn/KAIR/releases/download/v1.0/"
models = ["ffdnet_gray.pth", "ffdnet_color.pth", "drunet_gray.pth", "drunet_color.pth"]

# Crée le dossier s'il n'existe pas (sécurité)
os.makedirs(folder, exist_ok=True)

for model in models:
    path = os.path.join(folder, model)
    if not os.path.isfile(path):
        print(f"Téléchargement de {model}...")
        urllib.request.urlretrieve(url + model, path)
        print(f"{model} téléchargé.")
    else:
        print(f"{model} déjà présent.")
