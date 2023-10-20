import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer

class SequenceConcatenationModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mlp_input_size, mlp_hidden_sizes):
        super(SequenceConcatenationModule, self).__init__()
        
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        
        layers = [nn.Linear(mlp_input_size, mlp_hidden_sizes[0]), nn.ReLU()]
        for i in range(1, len(mlp_hidden_sizes)):
            layers.extend([nn.Linear(mlp_hidden_sizes[i - 1], mlp_hidden_sizes[i]), nn.ReLU()])
        layers.extend([nn.Linear(mlp_hidden_sizes[-1], embedding_dim), nn.ReLU()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, test_number_seq_input, test_vec_seq_input, test_mask):
        embedded_numbers = self.embedding_layer(test_number_seq_input)
        masked_embedded_numbers = embedded_numbers * test_mask.unsqueeze(-1).float()

        mlp_output = self.mlp(test_vec_seq_input)
        masked_mlp_output = mlp_output * (1 - test_mask.unsqueeze(-1).float())

        output = masked_embedded_numbers + masked_mlp_output

        return output

# Define the Transformer-based model
class TransformerSeq2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerSeq2Vec, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_encoder_layer = TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc = nn.Linear(embed_size, 1)  # Output a single value
        
    def forward(self, input_seq):
        embedded_seq = self.embedding(input_seq)
        transformer_output = self.transformer_encoder(embedded_seq)
        # Take the mean of transformer output across all positions in the sequence
        seq_vector = torch.mean(transformer_output, dim=1)
        output = self.fc(seq_vector)
        return output

# Define hyperparameters
vocab_size = 10000  # Size of the vocabulary
embed_size = 256   # Size of word embeddings
hidden_size = 512  # Size of the hidden layer in the Transformer
num_layers = 2     # Number of Transformer layers
num_heads = 4      # Number of attention heads in multi-head attention
dropout = 0.2      # Dropout rate

# Create the model
model = TransformerSeq2Vec(vocab_size, embed_size, hidden_size, num_layers, num_heads, dropout)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample input sequence (replace this with your own data)
input_sequence = torch.randint(0, vocab_size, (10,))  # Sequence of length 10

# Forward pass
output = model(input_sequence.unsqueeze(0))  # Add a batch dimension

# Print the output
print("Output:", output)