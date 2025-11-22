import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb
from sklearn.metrics import f1_score, accuracy_score

# Modülleri içe aktar (Dosya yollarına dikkat)
from architecture import LegalLSTM
from dataset import LegalDataset

# --- AYARLAR (Config) ---
CONFIG = {
    "vocab_size": 10002, # 10k kelime + 2 özel token
    "embedding_dim": 100,
    "hidden_dim": 64,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "pos_weight": 7.8 # <-- KRİTİK AYAR (Hesapladığımız değer)
}

def train():
    # 1. W&B Başlat
    wandb.init(project="LegalLens", config=CONFIG)
    config = wandb.config

    # 2. Cihaz Seçimi (GPU var mı?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Cihaz: {device}")

    # 3. Veriyi Hazırla
    print("⏳ Veri yükleniyor...")
    dataset = load_dataset("lex_glue", "unfair_tos")
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    # Dataset Class'ını başlat
    train_ds = LegalDataset(train_data['text'], train_data['labels'])
    val_ds = LegalDataset(val_data['text'], val_data['labels'], vocab=train_ds.vocab) # Aynı sözlüğü kullan!
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # 4. Model Kurulumu
    model = LegalLSTM(vocab_size=len(train_ds.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # LOSS FUNCTION: Class Imbalance için Ağırlıklı Loss
    pos_weight = torch.tensor([config.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Eğitim Döngüsü
    print("🔥 Eğitim Başlıyor...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].unsqueeze(1).to(device) # [batch] -> [batch, 1]
            
            optimizer.zero_grad()
            predictions, _ = model(inputs) # Attention ağırlıklarını şimdilik yoksay
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # --- Validation (Her epoch sonu test) ---
        val_acc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | F1-Score: {val_f1:.4f}")
        
        # W&B'ye Raporla
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "f1_score": val_f1, "accuracy": val_acc})

    # Modeli Kaydet
    torch.save(model.state_dict(), "legal_lstm_model.pth")
    print("💾 Model kaydedildi!")
    wandb.finish()

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            outputs, _ = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() # 0 ile 1 arasına sıkıştır
            preds = (preds > 0.5).astype(int) # 0.5 üstü 1 olsun
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1

if __name__ == "__main__":
    train()