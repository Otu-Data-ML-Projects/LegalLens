import sys
import os
import torch
import pickle
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Optional

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

class SummarizeRequest(BaseModel):
    texts: List[str]
    scores: Optional[List[float]] = None

class SummarizeResponse(BaseModel):
    summaries: List[str]
    overall_summary: str

# --- Gemini API Helper ---
async def summarize_with_gemini(text: str, score: float = 0.0) -> str:
    """Tek bir riskli maddeyi Gemini ile özetle"""
    if not settings.google_api_key:
        return "API key bulunamadı"
    
    prompt = f"""Sen bir hukuk uzmanısın. Aşağıdaki Terms & Conditions maddesini Türkçe olarak kısa ve anlaşılır şekilde özetle. 
Kullanıcının dikkat etmesi gereken riskleri vurgula. Maksimum 2-3 cümle kullan.

Risk Skoru: %{int(score * 100)}

Madde:
{text[:1000]}

Kısa Özet:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.google_api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 150
                    }
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return f"Özet oluşturulamadı (Hata: {response.status_code})"
    except Exception as e:
        return f"Özet hatası: {str(e)}"

async def create_overall_summary(texts: List[str], scores: List[float]) -> str:
    """Tüm riskli maddelerin genel özetini oluştur"""
    if not settings.google_api_key:
        return "API key bulunamadı"
    
    # En riskli 5 maddeyi al
    combined = list(zip(texts, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    top_risks = combined[:5]
    
    risk_texts = "\n\n".join([f"[Risk %{int(s*100)}]: {t[:300]}..." for t, s in top_risks])
    
    prompt = f"""Sen bir hukuk uzmanısın. Aşağıdaki Terms & Conditions'daki riskli maddeleri analiz et ve kullanıcıya Türkçe olarak genel bir özet sun.

Toplam {len(texts)} riskli madde bulundu. En riskli maddeler:

{risk_texts}

Kullanıcıya hitap ederek:
1. Bu sözleşmede dikkat edilmesi gereken ana riskler neler?
2. Kabul etmeden önce ne yapmalı?

Maksimum 4-5 cümle ile özetle:"""

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.google_api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 300
                    }
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return f"Genel özet oluşturulamadı (Hata: {response.status_code})"
    except Exception as e:
        return f"Özet hatası: {str(e)}"

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

# --- Summarize Endpoint ---
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_risks(payload: SummarizeRequest):
    """Riskli maddeleri Gemini ile özetle"""
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Google API key yapılandırılmamış")
    
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Özetlenecek metin bulunamadı")
    
    scores = payload.scores or [0.5] * len(payload.texts)
    
    # Her madde için özet oluştur (paralel)
    import asyncio
    summary_tasks = [
        summarize_with_gemini(text, score) 
        for text, score in zip(payload.texts, scores)
    ]
    summaries = await asyncio.gather(*summary_tasks)
    
    # Genel özet oluştur
    overall_summary = await create_overall_summary(payload.texts, scores)
    
    return {"summaries": list(summaries), "overall_summary": overall_summary}