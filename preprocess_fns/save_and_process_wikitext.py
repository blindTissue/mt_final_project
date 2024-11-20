from datasets import load_dataset, load_from_disk


def save_wikitext():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset.save_to_disk("data/wikitext-2")
