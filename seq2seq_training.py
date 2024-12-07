from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from encoder_decoder import VectorEncoderDecoder
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Setup device and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_models(device):
    llama_1b = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-1B").to(device)
    llama_3b = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-3B").to(device)
    llama_1b_trunc = AutoModelForCausalLM.from_pretrained("models/meta-llama/llama-3.2-1B")
    llama_tokenizer = AutoTokenizer.from_pretrained("tokenizers/meta-llama/llama-3.2-1B")

    # Set padding token
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_1b.config.pad_token_id = llama_tokenizer.eos_token_id
    llama_3b.config.pad_token_id = llama_tokenizer.eos_token_id
    llama_1b_trunc.config.pad_token_id = llama_tokenizer.eos_token_id

    return llama_1b, llama_3b, llama_1b_trunc, llama_tokenizer


def load_data(max_length=512):
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = dataset['train']['text']

    filtered_texts = [text for text in train_texts if text.strip() and len(text.split()) <= max_length]
    print(f"Processed {len(filtered_texts)} samples after filtering")

    return filtered_texts


class inner_translation_model(nn.Module):
    def __init__(self, src_model, tgt_model, translation_model, tgt_layer, src_layer):
        super().__init__()
        tgt_model.model.layers = tgt_model.model.layers[tgt_layer:]

        self.src_model = src_model
        self.tgt_model = tgt_model
        self.translation_model = translation_model
        self.src_layer = src_layer

        # Freeze source and target models
        for param in self.src_model.parameters():
            param.requires_grad = False
        for param in self.tgt_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        model_outs = self.src_model(x, output_hidden_states=True)
        model_outs_src_layer = model_outs.hidden_states[self.src_layer]
        batch_size, seq_len, hidden_dim = model_outs_src_layer.shape

        flattened = model_outs_src_layer.reshape(-1, hidden_dim)
        translation_outs = self.translation_model(flattened)
        translation_outs = translation_outs.reshape(batch_size, seq_len, -1)

        logits = self.tgt_model(inputs_embeds=translation_outs, use_cache=False).logits
        return translation_outs, logits


def loss_fn(intermediate_pred, intermediate_tgt, logits_pred, logits_tgt):
    cosine_loss = nn.CosineEmbeddingLoss()
    batch_size, seq_len, hidden_dim = intermediate_pred.shape

    cosine_loss_out = cosine_loss(
        intermediate_pred.reshape(-1, hidden_dim),
        intermediate_tgt.reshape(-1, hidden_dim),
        torch.ones(batch_size * seq_len).to(intermediate_pred.device)
    )

    kl_div = nn.KLDivLoss(reduction='batchmean')
    kl_div_out = kl_div(
        logits_pred.log_softmax(dim=-1),
        logits_tgt.softmax(dim=-1)
    )
    return cosine_loss_out + kl_div_out, cosine_loss_out.item(), kl_div_out.item()


class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.losses = {
            'total_loss': [],
            'cosine_loss': [],
            'kl_div_loss': [],
            'epoch': [],
            'iteration': []
        }

    def log_loss(self, total_loss, cosine_loss, kl_div_loss, epoch, iteration):
        self.losses['total_loss'].append(total_loss)
        self.losses['cosine_loss'].append(cosine_loss)
        self.losses['kl_div_loss'].append(kl_div_loss)
        self.losses['epoch'].append(epoch)
        self.losses['iteration'].append(iteration)

    def save_plots(self):
        df = pd.DataFrame(self.losses)

        # Plot total loss
        plt.figure(figsize=(10, 6))
        plt.plot(df['total_loss'])
        plt.title('Total Loss vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.log_dir, 'total_loss.pdf'))
        plt.close()

        # Plot component losses
        plt.figure(figsize=(10, 6))
        plt.plot(df['cosine_loss'], label='Cosine Loss')
        plt.plot(df['kl_div_loss'], label='KL Divergence Loss')
        plt.title('Component Losses vs Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, 'component_losses.pdf'))
        plt.close()

        # Save loss data
        df.to_csv(os.path.join(self.log_dir, 'training_losses.csv'), index=False)


def train_model(itm, llama_tokenizer, llama_1b, texts, device, max_len=512):
    epochs = 2
    optimizer = torch.optim.Adam(itm.parameters(), lr=0.001)
    logger = TrainingLogger()

    print(f"Starting training with {len(texts)} samples")
    print(f"Tokenizer pad token: {llama_tokenizer.pad_token}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, text in enumerate(texts):
            if not text.strip():
                continue

            # Tokenize with length limit
            input = llama_tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            input = {k: v.to(device) for k, v in input.items()}

            optimizer.zero_grad()
            intermediate_pred, logits_pred = itm(input["input_ids"])

            true_out = llama_1b(input["input_ids"], output_hidden_states=True)
            intermediate_tgt = true_out.hidden_states[10]
            logits_tgt = true_out.logits

            total_loss, cosine_loss, kl_div_loss = loss_fn(
                intermediate_pred, intermediate_tgt, logits_pred, logits_tgt
            )

            logger.log_loss(
                total_loss.item(),
                cosine_loss,
                kl_div_loss,
                epoch,
                i
            )

            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {total_loss.item():.4f}")
                print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            if i % 100 == 0:
                save_path = os.path.join(logger.log_dir, f"model_epoch_{epoch}_iter_{i}.pth")
                torch.save({
                    'model_state_dict': itm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'iteration': i,
                    'loss': total_loss.item()
                }, save_path)
                logger.save_plots()


def main():
    print("Starting main function...")

    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    llama_1b, llama_3b, llama_1b_trunc, llama_tokenizer = load_models(device)
    print(f"GPU memory after loading models: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print('Models loaded successfully')

    texts = load_data(max_length=256)
    print('Data loaded successfully')

    print("Creating translation model...")
    ved = VectorEncoderDecoder(3072, [1024, 512], 256, 2048)
    print("VectorEncoderDecoder created")

    print("Initializing inner translation model...")
    itm = inner_translation_model(llama_3b, llama_1b_trunc, ved, 10, 18).to(device)
    print("Inner translation model created and moved to device")

    print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    print("Starting training...")
    train_model(itm, llama_tokenizer, llama_1b, texts, device)


if __name__ == "__main__":
    main()