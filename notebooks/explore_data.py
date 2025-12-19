import pandas as pd
from datasets import load_dataset
import sys

def analyze_dataset():
    print("⏳ Dataset is downloading: LexGLUE (Unfair ToS)...")
    try:
        # Download the LexGLUE Unfair ToS dataset
        dataset = load_dataset("lex_glue", "unfair_tos")
    except Exception as e:
        print(f"❌ Download Error: {e}")
        print("Check your internet connection or ensure you have run 'pip install datasets'.")
        sys.exit(1)

    print("✅ Download Successful!\n")
    # Let's analyze the training set
    df = pd.DataFrame(dataset['train'])

    # --- 1. Data Size ---
    total_sentences = len(df)
    print(f"📊 Total Sentences (Train): {total_sentences}")

    # --- 2. Risk Analysis (Label Check) ---
    # LexGLUE format: labels = [] (Empty list) -> Safe
    #                  labels = [1, 3] (Non-empty list) -> Risky (Clause violation present)
    
    df['is_risky'] = df['labels'].apply(lambda x: 1 if len(x) > 0 else 0)
    
    risky_count = df['is_risky'].sum()
    safe_count = total_sentences - risky_count
    
    ratio = (risky_count / total_sentences) * 100

    print("\n--- 🚨 CRITICAL REPORT ---")
    print(f"✅ Safe Sentences: {safe_count}")
    print(f"❌ Risky Sentences: {risky_count}")
    print(f"📉 Risk Ratio    : %{ratio:.2f}")
    
    print("-" * 30)
    if ratio < 15:
        print("⚠️ RESULT: Data is Highly Imbalanced!")
        print("👉 We MUST use 'Class Weights' when training the model.")
    else:
        print("👍 RESULT: Data balance is acceptable.")

    # --- 3. Sample Data Display ---
    print("\n--- Sample Risky Sentence ---")
    risky_example = df[df['is_risky'] == 1].iloc[0]
    print(f"Text: {risky_example['text']}")
    print(f"Label: {risky_example['labels']}")

    print("\n--- 💾 SAVING DATASET TO CSV ---")
    # Loop through all dataset splits (Train, Validation, Test) and save them
    for split in dataset.keys():
        # Convert to Pandas DataFrame
        split_df = dataset[split].to_pandas()
        
        # Create filename (e.g., lexglue_train.csv)
        filename = f"lexglue_{split}.csv"
        
        # Save as CSV
        split_df.to_csv(filename, index=False)
        print(f"✅ {filename} successfully created!")

if __name__ == "__main__":
    analyze_dataset()