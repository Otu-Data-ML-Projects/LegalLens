import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Bahdanau Attention Mechanism (or similar Context-Aware Attention).
    Selects which output (hidden states) the LSTM will focus on.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Layer that will learn the attention weights
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        
        # 1. Calculate Scores (Energy)
        # Produces a score for each word -> [batch, seq_len, 1]
        attn_scores = self.attn(lstm_output) 
        
        # 2. Distribute Weights (Softmax)
        # Convert scores to probabilities (sum to 1) -> [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 3. Context Vector (Weighted Sum)
        # Multiply word vectors by their weights and sum
        # Result: A single vector representing the entire sentence -> [batch, hidden_dim]
        context_vector = torch.sum(lstm_output * attn_weights, dim=1)
        
        return context_vector, attn_weights

class LegalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, dropout=0.3):
        super(LegalLSTM, self).__init__()
        
        # 1. Embedding Layer: Converts word IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. Bi-Directional LSTM
        # batch_first=True -> Input shape: (batch, seq, feature)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout, 
                            batch_first=True)
        
        # 3. Attention Layer
        # Since it's a Bi-LSTM, the input dimension is hidden_dim * 2
        self.attention = Attention(hidden_dim * 2)
        
        # 4. Classifier (Is it risky or not?)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, max_len] -> Word IDs
        
        embedded = self.embedding(x) # -> [batch, seq, emb_dim]
        
        # LSTM Output
        # output: Hidden states for all steps
        # (h_n, c_n): Hidden state of the last step (We won't use it, we'll use Attention)
        lstm_output, (h_n, c_n) = self.lstm(embedded) # -> [batch, seq, hidden*2]
        
        # Attention Mechanism
        context_vector, attn_weights = self.attention(lstm_output)
        
        # Classification
        out = self.dropout(context_vector)
        prediction = self.fc(out) # -> [batch, 1] (Returns logits, Sigmoid applied later)
        
        return prediction, attn_weights