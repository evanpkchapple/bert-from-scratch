#Imports

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from transformers import BertTokenizerFast, AdamW
import glob
from tqdm import tqdm
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# This is the code to define the BERT model itself

class BertModel(nn.Module):
    def __init__(self, vocab_size, num_hidden_layers, hidden_size, num_attention_heads, max_positional_embeddings, num_segment_embeddings, dropout_prob):
        super(BertModel, self).__init__()
        self.vocab_size = vocab_size # Vocab size (from the tokenzizer)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads # hidden_size must be divisible cleanly by num_attention_heads
        self.max_positional_embeddings = max_positional_embeddings # Max sequence length to train positional embeddings to
        self.num_segment_embeddings = num_segment_embeddings # Set to 2 as normal BERT
        self.dropout_prob = dropout_prob

        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size) # Trainable word embedding layer
        self.positional_embedding = nn.Embedding(self.max_positional_embeddings, self.hidden_size) # Trainable positional embedding layer
        self.segment_embedding = nn.Embedding(self.num_segment_embeddings, self.hidden_size) # Trainable segmental embedding layer

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Stack of num_hidden_layers Transformer layers
        self.encoder_stack = nn.ModuleList([TransformerModule(self.hidden_size, self.num_attention_heads, self.dropout_prob) for _ in range(self.num_hidden_layers)])

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_length = input_ids.size(1)
        positional_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device) # Create a tensor of length sequence length for positional embedding
        positional_ids = positional_ids.unsqueeze(0).expand_as(input_ids) # Transform shape to match input ids

        # Embed all ids
        word_embeddings = self.word_embedding(input_ids)
        positional_embeddings = self.positional_embedding(positional_ids)
        segmental_embeddings = self.segment_embedding(token_type_ids)

        combined_ids = word_embeddings + positional_embeddings + segmental_embeddings # Combine embeddings
        normed_combined_ids = self.layer_norm(combined_ids)
        hidden_states = self.dropout(normed_combined_ids)

        attention_mask = attention_mask.float().unsqueeze(1).unsqueeze(2) # Match attention_mask to shape for MM later
        attention_mask = (1.0 - attention_mask) * -10000.0 # Fun math to make the attention_mask multiply masked token positions by very negative number so they are basically ignored during softmax

        # Loop through all encoder layers
        for layer in self.encoder_stack:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states

class TransformerModule(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(TransformerModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob

        self.attention = MultiHeadSelfAttention(self.hidden_size, self.num_attention_heads, self.dropout_prob) # Use multihead self attention for attention of this transformer block
        self.attention_output_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.attention_output_dropout = nn.Dropout(self.dropout_prob)

        self.feed_forward = FeedForward(self.hidden_size, self.dropout_prob) # Feed forward network to combine the attention output with residual stream
        self.feed_forward_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.feed_forward_dropout = nn.Dropout(self.dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)

        hidden_states = self.attention_output_norm(hidden_states + self.attention_output_dropout(attention_output))

        feed_forward_output = self.feed_forward(hidden_states)

        return self.feed_forward_norm(hidden_states + self.attention_output_dropout(feed_forward_output))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.head_dim = hidden_size // num_attention_heads # This is why we need to be confident of the sizes before

        # QKV attention
        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.projection = nn.Linear(self.hidden_size, self.hidden_size) # Projection layer from the attention

    def forward(self, hidden_states, attention_mask):
        batch_size, sequence_length, hidden_size = hidden_states.size()

        query = self.Q(hidden_states)
        key = self.K(hidden_states)
        value = self.V(hidden_states)

        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Scaled QKV product attention
        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = attention / math.sqrt(self.head_dim)

        # Apply the masking
        attention = attention + attention_mask

        # Apply softmax over the scaled QK output
        attention_probs = F.softmax(attention, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Multiply the softmaxed attention over the values to get the context
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)

        return self.projection(context)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(FeedForward, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size*4)
        self.GELU = nn.GELU() # GELU because ReLU is too basic for BERT
        self.linear2 = nn.Linear(self.hidden_size*4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        # Simple FF Network consisting of two Linear projections, an activation function, and dropout
        z = self.linear1(x)
        z = self.GELU(z)
        z = self.linear2(z)
        return self.dropout(z)

# This is the code to train the BERT model

class BertPretrainModel(nn.Module):
    def __init__(self, bert_encoder, hidden_size, vocab_size):
        super(BertPretrainModel, self).__init__()
        self.bert_encoder = bert_encoder # The BERT encoder to train
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size # This could be grabbed from the encoder...

        # This head is used for the masked language modelling task
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.GELU = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlm_out = nn.Linear(hidden_size, vocab_size)

        # This ties the encoder word embedding layer weights to the weights of the output for mlm, this saves space and is commonly done
        self.mlm_out.weight = self.bert_encoder.word_embedding.weight

        # This head is for the next sentence prediction
        self.nsp_out = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        hs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) # Get the outputs of the BERT encoder

        # Do masked language prediction for all token positions
        mlm_hs = self.linear1(hs)
        mlm_hs = self.GELU(mlm_hs)
        mlm_hs = self.layer_norm(mlm_hs)
        mlm_logits = self.mlm_out(mlm_hs)

        # Do NSP using the [CLS] token
        cls_embeddings = hs[:, 0, :] # Extract just the [CLS] token
        nsp_logits = self.nsp_out(cls_embeddings)

        return mlm_logits, nsp_logits

class BertPretrainDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.sent_pairs = [] # Collect all sentence pairs from all documents in the directory
        self.all_sents = [] # Collect all sentences from the documents (used to create false pairs)
        for file in tqdm(glob.glob(data_dir)):
            with open(file, 'r') as f:
                previous_sentence = '' # Set the previous sentence to the empty string
                for line in f.readlines():
                    curr_sent = line.strip()
                    self.all_sents.append(curr_sent)
                    if previous_sentence:
                        self.sent_pairs.append((previous_sentence, curr_sent)) # If we had a previous sentence in this document, add the pair to the sent_pairs
                    previous_sentence = curr_sent
        self.num_sents = len(self.all_sents) # Store total length of all_sents list
        self.special_tokens = set(self.tokenizer.all_special_ids) # Collect list of special tokens since we don't want to mask them
    def __len__(self):
        return len(self.sent_pairs)
    def __getitem__(self, idx):
        sent1, sent2 = self.sent_pairs[idx]
        nsp_label = True # Set nsp_label to True (changed to false when necessary)
        if random.random() < 0.5: # Make the NSP task target False 50% of the time
            rand_idx = random.randint(0, self.num_sents - 1)
            sent2 = self.all_sents[rand_idx] # This has a non-zero probability of being the correct sentence, but this is unlikely enough to ignore
            nsp_label = False

        # Encode the input via the BERT tokenizer
        encoding = self.tokenizer.encode_plus(sent1, sent2, add_special_tokens=True, max_length=32, truncation=True, padding="max_length", return_token_type_ids=True)
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        input_ids_len = len(input_ids)

        mlm_labels = [-100] * input_ids_len # Create a target tensor where first all positions are marked to ignore (-100)
        num_masked = int(round(input_ids_len * 0.15)) # Get the target number of tokens to mask

        possible_masked = [i for i, token in enumerate(input_ids) if token not in self.special_tokens] # Get all token positions that are not special tokens therefore can be masked
        random.shuffle(possible_masked) # Shuffle the list to get a random ordering
        possible_masked = possible_masked[:num_masked] # Select the first num_masked tokens from the shuffled ordering

        for i in possible_masked:
            mlm_labels[i] = input_ids[i] # Set the target tokens ids to their actual ids

            r = random.random()
            if r < 0.8: # 80% of the time, mask the input_id
                input_ids[i] = self.tokenizer.mask_token_id
            if r < 0.9: # 10% of the time, replace the token with a random token
                input_ids[i] = random.randint(0, self.vocab_size - 1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
            "nsp_label": torch.tensor(nsp_label, dtype=torch.long, device=device),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long, device=device)
        }

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

DATA_DIR = 'path/to/dataDirectory/*'
SAVE_DIR = 'path/to/saveDirectory'

# Define the hyperparameters for the BERT pretraining
BATCH_SIZE = 1024
VOCAB_SIZE = tokenizer.vocab_size
NUM_HIDDEN_LAYERS = 6
HIDDEN_SIZE = 384
NUM_ATTENTION_HEADS = 6
MAX_POSITIONAL_EMBEDDINGS = 256
NUM_SEGMENTAL_EMBEDDINGS = 2
DROPOUT = 0.1
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

# Use the BertTokenizerFast and create a dataset from the specified directory and a dataloader

dataset = BertPretrainDataset(DATA_DIR, tokenizer)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def pretrain(model, dataloader, loss_fn, num_epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr) # Adam optimizer with weight decay
    model.train()

    # The original BERT uses a scheduler, but this implementation will not
    for epoch in range(num_epochs):
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            token_type_ids = batch['token_type_ids']
            attention_mask = batch['attention_mask']
            nsp_labels = batch['nsp_label']
            mlm_labels = batch['mlm_labels']

            # Get the logits from passing the model the inputs_ids
            mlm_logits, nsp_logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

            # Calculate loss using CCE loss
            mlm_loss = loss_fn(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            nsp_loss = loss_fn(nsp_logits, nsp_labels)

            # Sum the two losses
            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        print(f"Epoch {epoch}:\t{str(total_loss / len(dataloader))}")
        os.makedirs(SAVE_DIR, exist_ok=True)
        checkpoint_path = os.path.join(SAVE_DIR, f"bert_pretrain_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(dataloader),
        }, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

base_model = BertModel(VOCAB_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_SIZE, NUM_ATTENTION_HEADS, MAX_POSITIONAL_EMBEDDINGS, NUM_SEGMENTAL_EMBEDDINGS, DROPOUT) # Define the base BERT model
base_model.to(device)
pretrain_model = BertPretrainModel(base_model, HIDDEN_SIZE, tokenizer.vocab_size) # Define the Pretraining model
pretrain_model.to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(device) # Define the CCE loss function
pretrain(pretrain_model, dataloader, loss_fn, NUM_EPOCHS, LEARNING_RATE) # Run the pretraining function