import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Bahdanau Attention Mechanism (veya benzeri Context-Aware Attention).
    LSTM'in tüm çıktılar (hidden states) arasından hangisine odaklanacağını seçer.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Attention ağırlıklarını öğrenecek katman
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        
        # 1. Skor Hesapla (Enerji)
        # Her kelime için bir skor üretir -> [batch, seq_len, 1]
        attn_scores = self.attn(lstm_output) 
        
        # 2. Ağırlıkları Dağıt (Softmax)
        # Skorları olasılığa çevir (Toplamları 1 olsun) -> [batch, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 3. Context Vector (Ağırlıklı Toplam)
        # Kelime vektörlerini ağırlıklarıyla çarpıp topla
        # Sonuç: Cümlenin tamamını temsil eden tek bir vektör -> [batch, hidden_dim]
        context_vector = torch.sum(lstm_output * attn_weights, dim=1)
        
        return context_vector, attn_weights

class LegalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, dropout=0.3):
        super(LegalLSTM, self).__init__()
        
        # 1. Embedding Layer: Kelime ID'lerini vektöre çevirir
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
        # Bi-LSTM olduğu için hidden_dim * 2 boyutu girer
        self.attention = Attention(hidden_dim * 2)
        
        # 4. Classifier (Riskli mi değil mi?)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, max_len] -> Kelime ID'leri
        
        embedded = self.embedding(x) # -> [batch, seq, emb_dim]
        
        # LSTM Çıktısı
        # output: Tüm adımlardaki hidden state'ler
        # (h_n, c_n): Son adımın state'i (Kullanmayacağız, Attention kullanacağız)
        lstm_output, (h_n, c_n) = self.lstm(embedded) # -> [batch, seq, hidden*2]
        
        # Attention Mekanizması
        context_vector, attn_weights = self.attention(lstm_output)
        
        # Sınıflandırma
        out = self.dropout(context_vector)
        prediction = self.fc(out) # -> [batch, 1] (Logits döner, Sigmoid sonra uygulanır)
        
        return prediction, attn_weights