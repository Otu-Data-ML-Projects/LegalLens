import pandas as pd
from datasets import load_dataset
import sys

def analyze_dataset():
    print("⏳ Veri seti indiriliyor: LexGLUE (Unfair ToS)...")
    try:
        # Veri setini indir
        dataset = load_dataset("lex_glue", "unfair_tos")
    except Exception as e:
        print(f"❌ İndirme Hatası: {e}")
        print("İnternet bağlantınızı kontrol edin veya 'pip install datasets' yaptığınızdan emin olun.")
        sys.exit(1)

    print("✅ İndirme Başarılı!\n")

    # Eğitim (Train) setini analize alalım
    df = pd.DataFrame(dataset['train'])

    # --- 1. Veri Boyutu ---
    total_sentences = len(df)
    print(f"📊 Toplam Cümle Sayısı (Train): {total_sentences}")

    # --- 2. Risk Analizi (Label Kontrolü) ---
    # LexGLUE formatı: labels = [] (Boş liste) -> Güvenli
    #                  labels = [1, 3] (Dolu liste) -> Riskli (Madde ihlali var)
    
    df['is_risky'] = df['labels'].apply(lambda x: 1 if len(x) > 0 else 0)
    
    risky_count = df['is_risky'].sum()
    safe_count = total_sentences - risky_count
    
    ratio = (risky_count / total_sentences) * 100

    print("\n--- 🚨 KRİTİK RAPOR ---")
    print(f"✅ Güvenli Cümleler: {safe_count}")
    print(f"❌ Riskli Cümleler : {risky_count}")
    print(f"📉 Riskli Oranı    : %{ratio:.2f}")
    
    print("-" * 30)
    if ratio < 15:
        print("⚠️ SONUÇ: Veri Çok Dengesiz (Imbalanced)!")
        print("👉 Model eğitirken 'Class Weights' kullanmak ZORUNDAYIZ.")
    else:
        print("👍 SONUÇ: Veri dengesi kabul edilebilir.")

    # --- 3. Örnek Veri Gösterimi ---
    print("\n--- Örnek Riskli Cümle ---")
    risky_example = df[df['is_risky'] == 1].iloc[0]
    print(f"Text: {risky_example['text']}")
    print(f"Label: {risky_example['labels']}")

if __name__ == "__main__":
    analyze_dataset()