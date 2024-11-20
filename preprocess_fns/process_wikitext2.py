from datasets import load_from_disk

import pickle
import os
def load_wikitext():
    wikitext2 = load_from_disk('../data/wikitext-2')
    return wikitext2

def chunk_by_topic(wikitext_data):
    outputs = []
    string_stream = ""
    for item in wikitext_data['text']:
        if validate_single_equals(item):
            outputs.append(string_stream)
            string_stream = item
        else:
            string_stream += item

    outputs.append(string_stream)
    return outputs[1:]


# wikitext data is seperated by topic with = Topic =, = = Subtopic == ...
# TBH The model should be better when we remove the excessive \n and init.
def validate_single_equals(text):
    text = text.strip()
    # Check if string starts and ends with =
    starts_with_single = text.startswith('=') and not text.startswith('= =')
    ends_with_single = text.endswith('=') and not text.endswith('= =')

    return starts_with_single and ends_with_single

def save_chunked_wikitext():
    wikitext2 = load_wikitext()
    for split in wikitext2:
        chunked = chunk_by_topic(wikitext2[split])
        if not os.path.exists("../data/chunked_wikitext2"):
            os.makedirs("../data/chunked_wikitext2")

        with open(f"data/chunked_wikitext2/{split}.pkl", "wb") as f:
            pickle.dump(chunked, f)

if __name__ == "__main__":
    save_chunked_wikitext()