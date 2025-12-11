// --- Ayarlar ---
const API_URL = "http://127.0.0.1:8000/analyze";
const SUMMARIZE_URL = "http://127.0.0.1:8000/summarize";

// --- Durum ---
let isAnalyzed = false;
let riskyElements = [];
let riskyTexts = [];
let riskyScores = [];
let summaries = [];
let overallSummary = "";
let termsLink = null; // Terms sayfasının linki

// Terms and Services ile ilgili tüm olası başlık varyasyonları
const TERMS_KEYWORDS = [
    'terms of service', 'terms of use', 'terms and conditions',
    'conditions of use', 'user agreement', 'legal terms',
    'service agreement', 'usage terms', 'terms & conditions',
    'kullanım şartları', 'kullanım koşulları', 'hizmet şartları',
    'end user agreement', 'eula', 'license agreement',
    'privacy policy', 'legal notice', 'disclaimer'
];

// URL'de aranacak anahtar kelimeler
const URL_KEYWORDS = [
    'terms', 'conditions', 'tos', 'legal', 'privacy', 'policy',
    'agreement', 'eula', 'license', 'disclaimer', 'notice',
    'kullanim', 'sartlar', 'kosullar', 'gizlilik'
];

// URL'de terms/legal içerik var mı kontrol et
function isLegalPage() {
    const url = window.location.href.toLowerCase();
    const title = document.title.toLowerCase();
    
    // URL kontrolü
    for (const keyword of URL_KEYWORDS) {
        if (url.includes(keyword)) {
            return true;
        }
    }
    
    // Sayfa başlığı kontrolü
    for (const keyword of TERMS_KEYWORDS) {
        if (title.includes(keyword)) {
            return true;
        }
    }
    
    return false;
}

// Terms başlığını bul
function findTermsHeader() {
    // Önce link'leri kontrol et (a tag'leri) - bunlar genellikle Terms sayfasına yönlendirir
    const links = document.querySelectorAll('a');
    for (const el of links) {
        const text = el.innerText?.toLowerCase().trim() || '';
        const href = el.href || '';
        for (const keyword of TERMS_KEYWORDS) {
            if (text.includes(keyword) && text.length < 100) {
                // Bu bir link, href'i kaydet
                if (href && !href.startsWith('javascript:')) {
                    termsLink = href;
                }
                return el;
            }
        }
    }
    
    // Başlık elementlerini kontrol et (h1, h2, h3, h4, span, div)
    const selectors = ['h1', 'h2', 'h3', 'h4', 'span', 'div', 'header'];
    
    for (const selector of selectors) {
        const elements = document.querySelectorAll(selector);
        for (const el of elements) {
            const text = el.innerText?.toLowerCase().trim() || '';
            for (const keyword of TERMS_KEYWORDS) {
                if (text.includes(keyword) && text.length < 100) {
                    return el;
                }
            }
        }
    }
    
    // Sayfa başlığını kontrol et
    const pageTitle = document.title.toLowerCase();
    for (const keyword of TERMS_KEYWORDS) {
        if (pageTitle.includes(keyword)) {
            // İlk h1 veya h2'yi döndür
            return document.querySelector('h1') || document.querySelector('h2');
        }
    }
    
    return null;
}

// Sayfada yeterli içerik var mı kontrol et
function hasEnoughContent() {
    const elements = Array.from(document.querySelectorAll('p, li'));
    const candidates = elements.filter(el => el.innerText.trim().length > 30);
    return candidates.length >= 5; // En az 5 paragraf/liste item'ı olmalı
}

// 1. Logo/Butonu Oluştur (Sağ alt köşeye sabit)
function createLegalLensLogo() {
    if (document.getElementById("legallens-logo-container")) return;
    
    // Sadece legal/terms sayfalarında çalış
    if (!isLegalPage()) {
        console.log("LegalLens: Bu sayfa bir Terms/Legal sayfası değil, logo eklenmedi.");
        return;
    }

    // Logo container (floating - sağ alt)
    const logoContainer = document.createElement("div");
    logoContainer.id = "legallens-logo-container";
    
    // Logo
    const logo = document.createElement("div");
    logo.id = "legallens-logo";
    logo.innerHTML = "⚖️";
    logo.title = "LegalLens - Analiz Et";
    
    // Logo text
    const logoText = document.createElement("span");
    logoText.id = "legallens-logo-text";
    logoText.innerText = "LegalLens";
    
    // Tooltip
    const tooltip = document.createElement("div");
    tooltip.id = "legallens-tooltip";
    tooltip.innerHTML = `
        <div class="legallens-tooltip-header">⚖️ LegalLens</div>
        <div class="legallens-tooltip-content">
            <p>Bu dokümandaki riskli maddeleri analiz etmek için tıklayın.</p>
            <p class="legallens-tooltip-info">Kırmızı ile işaretlenen bölümler, dikkat etmeniz gereken şartlardır.</p>
        </div>
        <div class="legallens-tooltip-footer" id="legallens-risk-summary"></div>
        <div class="legallens-tooltip-actions" id="legallens-actions"></div>
    `;
    
    // Summary Modal oluştur
    createSummaryModal();
    
    logoContainer.appendChild(logo);
    logoContainer.appendChild(logoText);
    logoContainer.appendChild(tooltip);
    
    // Body'ye ekle (floating)
    document.body.appendChild(logoContainer);
    
    // Logo tıklama olayı
    logoContainer.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        if (isAnalyzed && riskyElements.length > 0) {
            // Zaten analiz edilmişse, modal'ı aç
            openSummaryModal();
            return;
        }
        
        // Sayfada yeterli içerik yoksa ve terms linki varsa, yönlendir
        if (!hasEnoughContent() && termsLink) {
            const userChoice = confirm(
                "Bu sayfada analiz edilecek yeterli içerik yok.\n\n" +
                "Terms & Conditions sayfasına gitmek ister misiniz?\n" +
                "Orada LegalLens otomatik olarak analiz yapacaktır."
            );
            
            if (userChoice) {
                window.location.href = termsLink;
            }
            return;
        }
        
        logo.innerHTML = '<span class="legallens-spinner"></span>';
        logoText.innerText = "Analiz ediliyor...";
        
        try {
            await analyzePage();
            logo.innerHTML = "✅";
            logoText.innerText = "LegalLens";
            isAnalyzed = true;
            updateTooltipSummary();
            
            // Analiz bitince direkt modal aç
            if (riskyElements.length > 0) {
                setTimeout(() => openSummaryModal(), 300);
            }
            
        } catch (error) {
            console.error(error);
            logo.innerHTML = "❌";
            logoText.innerText = "Hata!";
            alert("Hata: API çalışmıyor olabilir!");
            
            setTimeout(() => {
                logo.innerHTML = "⚖️";
                logoText.innerText = "LegalLens";
            }, 3000);
        }
    };
}

// Summary Modal oluştur
function createSummaryModal() {
    if (document.getElementById("legallens-modal")) return;
    
    const modal = document.createElement("div");
    modal.id = "legallens-modal";
    modal.innerHTML = `
        <div class="legallens-modal-content">
            <div class="legallens-modal-header">
                <span>📋 Tüm Riskli Maddeler Özeti</span>
                <button class="legallens-modal-close" id="legallens-modal-close">✕</button>
            </div>
            <div class="legallens-modal-body" id="legallens-modal-body">
                <p class="legallens-modal-placeholder">Henüz analiz yapılmadı. Önce logoya tıklayarak analiz başlatın.</p>
            </div>
            <div class="legallens-modal-footer">
                <span id="legallens-modal-count"></span>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Modal kapatma
    document.getElementById("legallens-modal-close").onclick = () => {
        modal.classList.remove("legallens-modal-open");
    };
    
    // Dışarı tıklayınca kapat
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.classList.remove("legallens-modal-open");
        }
    };
}

// Summary Modal'ı aç
async function openSummaryModal() {
    const modal = document.getElementById("legallens-modal");
    const body = document.getElementById("legallens-modal-body");
    const count = document.getElementById("legallens-modal-count");
    
    if (!modal || !body) return;
    
    if (riskyElements.length === 0) {
        body.innerHTML = `<p class="legallens-modal-placeholder">Henüz riskli madde bulunamadı.</p>`;
        count.textContent = "";
    } else {
        // Önce loading göster
        body.innerHTML = `
            <div class="legallens-loading-container">
                <span class="legallens-spinner-large"></span>
                <p>LegalLens sizin için şartları inceliyor...</p>
            </div>
        `;
        modal.classList.add("legallens-modal-open");
        
        // Eğer özetler henüz alınmadıysa, API'den al
        if (summaries.length === 0 && riskyTexts.length > 0) {
            try {
                const response = await fetch(SUMMARIZE_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        texts: riskyTexts,
                        scores: riskyScores
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    summaries = data.summaries;
                    overallSummary = data.overall_summary;
                }
            } catch (error) {
                console.error("Özet API hatası:", error);
            }
        }
        
        // Riskli maddelerin özetini oluştur
        let html = '';
        
        // Genel özet
        if (overallSummary) {
            html += `
                <div class="legallens-overall-summary">
                    <div class="legallens-overall-header">🤖 Genel Değerlendirme</div>
                    <p>${overallSummary}</p>
                </div>
            `;
        }
        
        html += '<div class="legallens-risk-list">';
        
        riskyElements.forEach((el, index) => {
            const score = parseFloat(el.dataset.riskScore || 0);
            const scorePercent = (score * 100).toFixed(0);
            const text = el.innerText.trim();
            const shortText = text.length > 150 ? text.substring(0, 150) + '...' : text;
            const riskLevel = score >= 0.8 ? 'high' : score >= 0.6 ? 'medium' : 'low';
            const summary = summaries[index] || "Özet yükleniyor...";
            
            html += `
                <div class="legallens-risk-item legallens-risk-${riskLevel}" data-index="${index}">
                    <div class="legallens-risk-item-header">
                        <span class="legallens-risk-badge">Risk: %${scorePercent}</span>
                        <span class="legallens-risk-number">#${index + 1}</span>
                    </div>
                    <div class="legallens-ai-summary">
                        <span class="legallens-ai-badge">🤖 AI Özet</span>
                        <p>${summary}</p>
                    </div>
                    <details class="legallens-original-text">
                        <summary>📄 Orijinal Metin</summary>
                        <p>${shortText}</p>
                    </details>
                    <button class="legallens-goto-btn" data-idx="${index}">📍 Bu maddeye git</button>
                </div>
            `;
        });
        
        html += '</div>';
        body.innerHTML = html;
        count.textContent = `Toplam ${riskyElements.length} riskli madde`;
        
        // "Bu maddeye git" butonlarına event ekle
        body.querySelectorAll('.legallens-goto-btn').forEach((btn) => {
            btn.onclick = () => {
                const idx = parseInt(btn.dataset.idx);
                modal.classList.remove("legallens-modal-open");
                scrollToRiskyElement(idx);
            };
        });
    }
    
    modal.classList.add("legallens-modal-open");
}

// Belirli bir riskli elemana scroll
function scrollToRiskyElement(index) {
    if (riskyElements.length === 0 || index < 0 || index >= riskyElements.length) return;
    
    // Önceki highlight'ı kaldır
    riskyElements.forEach(el => el.classList.remove('legallens-current-risky'));
    
    const targetEl = riskyElements[index];
    
    if (targetEl) {
        targetEl.classList.add('legallens-current-risky');
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Tooltip içeriğini güncelle
function updateTooltipSummary() {
    const summaryEl = document.getElementById("legallens-risk-summary");
    const actionsEl = document.getElementById("legallens-actions");
    
    if (summaryEl && riskyElements.length > 0) {
        summaryEl.innerHTML = `
            <strong>⚠️ ${riskyElements.length} riskli madde bulundu!</strong>
            <p>Kabul etmeden önce kırmızı bölümleri dikkatlice okuyun.</p>
        `;
        
        // Actions bölümünü temizle (buton yok artık)
        if (actionsEl) {
            actionsEl.innerHTML = '';
        }
    } else if (summaryEl) {
        summaryEl.innerHTML = `<strong>✅ Riskli madde bulunamadı.</strong>`;
        if (actionsEl) actionsEl.innerHTML = '';
    }
}

// Riskli elemanlara sırayla scroll
let currentRiskyIndex = 0;
function scrollToNextRisky() {
    if (riskyElements.length === 0) return;
    
    // Önceki highlight'ı kaldır
    riskyElements.forEach(el => el.classList.remove('legallens-current-risky'));
    
    // Sonraki elemana geç
    currentRiskyIndex = (currentRiskyIndex) % riskyElements.length;
    const targetEl = riskyElements[currentRiskyIndex];
    
    if (targetEl) {
        targetEl.classList.add('legallens-current-risky');
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        currentRiskyIndex++;
    }
}

// 2. Analiz Fonksiyonu
async function analyzePage() {
    // Paragraf ve listeleri al
    const elements = Array.from(document.querySelectorAll('p, li'));
    
    // Filtrele: 30 karakterden uzun cümleleri al
    const candidates = elements
        .map(el => ({ element: el, text: el.innerText.trim() }))
        .filter(item => item.text.length > 30);
    
    // Yeterli içerik yoksa ve terms linki varsa yönlendir
    if (candidates.length < 5 && termsLink) {
        const userChoice = confirm(
            "Bu sayfada analiz edilecek yeterli içerik yok.\n\n" +
            "Terms & Conditions sayfasına gitmek ister misiniz?\n" +
            "Orada LegalLens otomatik olarak analiz yapacaktır."
        );
        
        if (userChoice) {
            window.location.href = termsLink;
        }
        return;
    }
    
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
    
    // Riskli elementleri ve verileri sıfırla
    riskyElements = [];
    riskyTexts = [];
    riskyScores = [];
    summaries = [];
    overallSummary = "";
    
    // Boyama İşlemi
    data.risky_indices.forEach((idx) => {
        const item = candidates[idx];
        if (item && item.element) {
            item.element.classList.add("legallens-risky-highlight");
            item.element.title = `Risk Skoru: %${(data.scores[idx] * 100).toFixed(0)}`;
            item.element.dataset.riskScore = data.scores[idx];
            riskyElements.push(item.element);
            riskyTexts.push(item.text);
            riskyScores.push(data.scores[idx]);
        }
    });

    // Skorlara göre sırala (en riskli önce)
    const combined = riskyElements.map((el, i) => ({
        element: el,
        text: riskyTexts[i],
        score: riskyScores[i]
    }));
    combined.sort((a, b) => b.score - a.score);
    
    riskyElements = combined.map(c => c.element);
    riskyTexts = combined.map(c => c.text);
    riskyScores = combined.map(c => c.score);

    if (riskyElements.length > 0) {
        console.log(`LegalLens: ${riskyElements.length} riskli madde bulundu.`);
    } else {
        alert("Riskli madde bulunamadı! Temiz.");
    }
}

// Başlat
createLegalLensLogo();