# 🧠 AI Resume Analyzer  
### Yapay Zekâ Destekli Özgeçmiş Analiz Aracı  

**AI Resume Analyzer**, yapay zekâ (NLP) kullanarak özgeçmişleri analiz eden ve güçlü/zayıf yönleri belirleyen bir web uygulamasıdır.  
Kullanıcı PDF veya metin olarak CV yükler, sistem HuggingFace modeliyle içeriği analiz eder ve yüzdelik skor + geliştirme önerileri döner.  

---

## 🚀 Özellikler

✅ PDF veya metin olarak CV yükleme  
✅ distilBERT modeliyle NLP analizi (semantic + keyword tabanlı)  
✅ “Software”, “AI”, “Data Science”, “Soft Skills” skorları  
✅ Otomatik öneri listesi (örnek: *“Soft skill kısmını güçlendirebilirsin”*)  
✅ Hedef pozisyon seçimi (Backend / Data Scientist / AI Engineer / Frontend / Mobile)  
✅ Seçilen role göre **eksik anahtar kelime önerileri**  
✅ Basit görsel rapor (Chart.js bar chart)  
✅ JSON API endpoint (`/api/analyze`)  
✅ Flask + PyPDF2 + HuggingFace Transformers  
✅ GitHub Actions CI (otomatik test)

---

## 🏗️ Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|------------|
| Backend | Python, Flask |
| NLP | HuggingFace Transformers (distilbert-base-uncased) |
| Veri İşleme | PyTorch, PyPDF2 |
| Frontend | HTML, CSS, Chart.js |
| Test & CI | Pytest, GitHub Actions |

---

## 📂 Proje Yapısı

```bash
ai-resume-analyzer/
│
├─ app.py                     # Flask backend
├─ resume_analyzer.py         # NLP & analiz fonksiyonları
├─ requirements.txt           # Bağımlılıklar
├─ templates/
│   ├─ index.html             # Ana sayfa (upload formu)
│   └─ result.html            # Analiz sonuç sayfası
├─ static/
│   └─ style.css              # CSS stilleri
├─ tests/
│   └─ test_analyzer.py       # Basit unit test
└─ .github/
    └─ workflows/
        └─ python-app.yml     # GitHub Actions CI pipeline
