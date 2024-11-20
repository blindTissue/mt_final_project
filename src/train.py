from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
from encoder_decoder import VectorEncoderDecoder
from torch import nn
import torch

llama_1b = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-1B")
llama_3b = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-3B")
llama_1b_trunc = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-1B")
llama_tokenizer = AutoTokenizer.from_pretrained("tokenizers/meta-llama/llama-3.2-1B")
wikitext_chunked = pickle.load(open("data/chunked_wikitext2/train.pkl", "rb"))
device = 'mps' # change to whatever device you are using
max_len = 1000
loss_sum = 0

class inner_translation_model(nn.Module):
    def __init__(self, src_model, tgt_model, translation_model, tgt_layer, src_layer):
        super().__init__()
        tgt_model.model.layers = tgt_model.model.layers[tgt_layer:]

        self.src_model = src_model
        self.tgt_model = tgt_model
        self.translation_model = translation_model
        self.src_layer = src_layer
        # freeze source, target models
        for param in self.src_model.parameters():
            param.requires_grad = False
        for param in self.tgt_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        model_outs = self.src_model(x, output_hidden_states=True)
        model_outs_src_layer = model_outs.hidden_states[self.src_layer]
        model_outs_src_layer = model_outs_src_layer.squeeze()
        #model_outs_src_layer = model_outs_src_layer.permute(1,0,2)
        #print(model_outs_src_layer.shape)
        translation_outs = self.translation_model(model_outs_src_layer)
        translation_outs = translation_outs.unsqueeze(0)
        logits = self.tgt_model(inputs_embeds = translation_outs, use_cache = False).logits
        return translation_outs, logits


def loss_fn(intermediate_pred, intermediate_tgt, logits_pred, logits_tgt):
    cosine_loss = nn.CosineEmbeddingLoss()
    cosine_loss_out = cosine_loss(
        intermediate_pred.squeeze(),
        intermediate_tgt.squeeze(),
        torch.ones(intermediate_pred.shape[1]).to(device)
    )

    kl_div = nn.KLDivLoss()
    kl_div_out = kl_div(
        logits_pred.log_softmax(dim=-1),
        logits_tgt.softmax(dim=-1)
    )
    return cosine_loss_out + kl_div_out

ved = VectorEncoderDecoder(3072, [1024, 512], 256, 2048)
itm = inner_translation_model(llama_3b, llama_1b_trunc, ved, 10, 18)


def train_model():
    epochs = 2
    optimizer = torch.optim.Adam(itm.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, text in enumerate(wikitext_chunked):
            input = llama_tokenizer(text, max_length=max_len, return_tensors="pt", truncation=True)

            # input = llama_tokenizer(text, return_tensors="pt")
            input.to(device)
            optimizer.zero_grad()
            intermediate_pred, logits_pred = itm(input["input_ids"])
            true_out = llama_1b(input["input_ids"], output_hidden_states=True)
            intermediate_tgt = true_out.hidden_states[10]
            logits_tgt = true_out.logits
            loss = loss_fn(intermediate_pred, intermediate_tgt, logits_pred, logits_tgt)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss_sum / 10}")
                loss_sum = 0
            if i % 100 == 0:
                torch.save(itm, f"models/inner_translation_model_{epoch}_{i}.pth")

if __name__ == "__main__":
    train_model()