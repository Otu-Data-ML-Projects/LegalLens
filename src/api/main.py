import sys
import os
import torch
import pickle
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

# --- Configuration ---
# we import settings from config.py
from .config import settings

# --- Path Settings (Architecture) ---
# Add src folder to path so we can import model files
# This ensures that no matter where the script runs, it can find src
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/api
parent_dir = os.path.dirname(current_dir) # src
sys.path.append(parent_dir)

from model.architecture import LegalLSTM

# --- Global Variables ---
ml_models = {}
vocab = {}
DEVICE = torch.device("cpu") # CPU is sufficient and stable for server/API side

# --- Helper: Dynamic File Path Resolver ---
def get_abs_path(relative_path: str):
    # Find the project's root directory (src/api -> src -> root)
    root_dir = os.path.dirname(parent_dir)
    return os.path.join(root_dir, relative_path)

# --- Helper: Text Processing ---
def text_to_indices(text: str, vocab: dict, max_len: int):
    indices = []
    for word in text.split():
        indices.append(vocab.get(word.lower(), vocab.get("<UNK>", 1)))
    
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
        
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

# --- Lifespan (Application Lifespan) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Resolve File Paths
    vocab_full_path = get_abs_path(settings.vocab_path)
    model_full_path = get_abs_path(settings.model_path)
    
    print(f"📂 Loading Vocabulary from: {vocab_full_path}")
    
    try:
        with open(vocab_full_path, "rb") as f:
            global vocab
            vocab = pickle.load(f)
        
        print(f"🧠 Loading Model form: {model_full_path}")
        model = LegalLSTM(vocab_size=len(vocab)) 
        model.load_state_dict(torch.load(model_full_path, map_location=DEVICE))
        model.eval()
        ml_models["legal_lens"] = model
        print("✅ System Ready & Secure!")
        
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: File not found. Check .env paths.\n{e}")
        # In a real-world scenario, consider logging this error properly.
        
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- SECURITY (CORS) ---
# Only allowed origins can access, similar to React
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"], # SADECE "POST" DEĞİL, TÜM METODLARA (OPTIONS DAHİL) İZİN VER
    allow_headers=["*"],
)

# --- Schemas ---
class AnalyzeRequest(BaseModel):
    sentences: List[str]

class AnalyzeResponse(BaseModel):
    risky_indices: List[int]
    scores: List[float]

# --- Endpoint ---
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentences(payload: AnalyzeRequest):
    model = ml_models.get("legal_lens")
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded properly.")
    
    risky_indices = []
    scores = []
    
    with torch.no_grad():
        for idx, sentence in enumerate(payload.sentences):
            if not sentence.strip():
                continue
                
            tensor_input = text_to_indices(sentence, vocab, settings.max_len).to(DEVICE)
            prediction, _ = model(tensor_input)
            
            score = torch.sigmoid(prediction).item()
            scores.append(score)
            
            if score > 0.5:
                risky_indices.append(idx)
    
    return {"risky_indices": risky_indices, "scores": scores}