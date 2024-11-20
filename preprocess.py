from preprocess_fns.get_llms import load_and_save_models_from_huggingface
from preprocess_fns.save_and_process_wikitext import save_wikitext
from preprocess_fns.process_wikitext2 import save_chunked_wikitext

def preprocess():
    load_and_save_models_from_huggingface()
    save_wikitext()
    save_chunked_wikitext()


if __name__ == "__main__":
    preprocess()
