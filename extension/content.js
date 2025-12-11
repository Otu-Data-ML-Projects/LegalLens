// --- Ayarlar ---
const API_URL = "http://127.0.0.1:8000/analyze";

// --- Durum ---
let isAnalyzed = false;

// 1. Butonu Oluştur
function createFloatingButton() {
    if (document.getElementById("legallens-floating-btn")) return;

    const btn = document.createElement("div");
    btn.id = "legallens-floating-btn";
    btn.innerText = "Legal\nLens"; 
    btn.title = "Sayfayı Analiz Et";
    
    btn.onclick = async () => {
        if (isAnalyzed) return; 
        
        const originalText = btn.innerText;
        btn.innerHTML = '<div class="legallens-loading"></div>';
        
        try {
            await analyzePage();
            btn.innerHTML = "✅"; // Başarılı
            btn.style.backgroundColor = "#27ae60";
            isAnalyzed = true;
            
            setTimeout(() => {
                btn.innerHTML = "Legal\nLens"; 
                btn.style.backgroundColor = "#2c3e50";
                isAnalyzed = false;
            }, 5000);

        } catch (error) {
            console.error(error);
            btn.innerHTML = "❌"; 
            btn.style.backgroundColor = "#c0392b";
            alert("Hata: API çalışmıyor olabilir! Terminali kontrol et.");
            
            setTimeout(() => {
                btn.innerText = originalText;
                btn.style.backgroundColor = "#2c3e50";
            }, 3000);
        }
    };
    
    document.body.appendChild(btn);
}

// 2. Analiz Fonksiyonu
async function analyzePage() {
    // Paragraf ve listeleri al
    const elements = Array.from(document.querySelectorAll('p, li'));
    
    // Filtrele: 30 karakterden uzun cümleleri al
    const candidates = elements
        .map(el => ({ element: el, text: el.innerText.trim() }))
        .filter(item => item.text.length > 30);
    
    if (candidates.length === 0) {
        alert("Yeterli uzunlukta metin bulunamadı.");
        return;
    }

    // Sadece metinleri gönder
    const sentences = candidates.map(c => c.text);

    // API'ye İstek At
    const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentences: sentences })
    });

    if (!response.ok) throw new Error("API Hatası");
    
    const data = await response.json();
    
    // Boyama İşlemi
    let riskCount = 0;
    data.risky_indices.forEach((idx) => {
        const item = candidates[idx];
        if (item && item.element) {
            item.element.classList.add("legallens-risky-highlight");
            item.element.title = `Risk Skoru: %${(data.scores[idx] * 100).toFixed(0)}`;
            riskCount++;
        }
    });

    if (riskCount > 0) console.log(`${riskCount} riskli madde bulundu.`);
    else alert("Riskli madde bulunamadı! Temiz.");
}

// Başlat
createFloatingButton();