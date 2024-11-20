from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def load_and_save_models_from_huggingface(model_names=None):
    if model_names is None:
        model_names = ["meta-llama/llama-3.2-1B", "meta-llama/llama-3.2-3B"]
    print("Downloading models ", model_names)



    for model_name in model_names:
        if os.path.exists(f"models/{model_name}") and os.path.exists(f"tokenizers/{model_name}"):
            print(f"Model {model_name} already exists. Skipping download.")
            continue
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        os.makedirs(f"models/{model_name}", exist_ok=True)
        os.makedirs(f"tokenizers/{model_name}", exist_ok=True)
        model.save_pretrained(f"models/{model_name}")
        tokenizer.save_pretrained(f"tokenizers/{model_name}")
    print(f"Finished downloading models {model_names}")


