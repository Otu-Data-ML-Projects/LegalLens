import torch
import torch.nn as nn
import torch.nn.functional as F

class LegalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, n_layers=2, dropout=0.3):
        super(LegalLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.encoder = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=True, 
                            dropout=dropout, 
                            batch_first=True)
        
        self.W_s1 = nn.Linear(2 * hidden_dim, 256)
        self.W_s2 = nn.Linear(256, 1)
        
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def attention_net(self, lstm_output):
        
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output))) 
        
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        
        return torch.bmm(attn_weight_matrix, lstm_output).squeeze(1), attn_weight_matrix

    def forward(self, x):
        embedded = self.embedding(x)
        
        output, _ = self.encoder(embedded)
        
        context_vector, attn_weights = self.attention_net(output)
        
        out = self.dropout(context_vector)
        prediction = self.fc(out)
        
        return prediction, attn_weights