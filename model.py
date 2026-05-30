from google import genai
import yaml
from pathlib import Path

config_dir = Path(__file__).parent
config_file = config_dir / "setup.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

client = genai.Client(api_key=config["Gemini_api_key"])

EXCLUDE_PREFIXES = ("embedding", "text-embedding", "gemini-embedding", "imagen", "veo", "aqa", "deep-research")
EXCLUDE_SUBSTRINGS = ("-tts", "image-generation", "-image", "native-audio", "robotics", "computer-use", "nano-banana")

try:
    models = client.models.list()
    for model in models:
        base = model.name.replace("models/", "", 1).lower()
        if any(base.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(s in base for s in EXCLUDE_SUBSTRINGS):
            continue
        print(model.name)
finally:
    client.close()
