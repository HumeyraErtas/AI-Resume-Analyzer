# resume_analyzer.py
import re
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# 1) HuggingFace modelini yükle
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# 2) Kategori bazlı keyword listeleri
SKILL_KEYWORDS = {
    "software": [
        "python", "java", "c++", "c#", "kotlin", "javascript", "typescript",
        "react", "angular", "node.js", "flask", "django", "git", "rest",
        "api", "oop", "docker", "linux"
    ],
    "ai": [
        "machine learning", "deep learning", "neural network", "pytorch",
        "tensorflow", "transformers", "nlp", "computer vision", "cnn",
        "rnn", "lstm", "bert", "gpt", "huggingface"
    ],
    "data_science": [
        "pandas", "numpy", "matplotlib", "seaborn", "sql", "data analysis",
        "data visualization", "regression", "classification", "clustering",
        "scikit-learn", "feature engineering", "eda"
    ],
    "soft_skills": [
        "team", "teamwork", "communication", "leadership", "problem solving",
        "presentation", "collaboration", "mentor", "mentoring",
        "conflict resolution", "time management"
    ],
}

# 3) Hedef role göre beklenen (ideal) anahtar kelimeler
ROLE_EXPECTED_SKILLS = {
    "backend": [
        "python", "java", "c#", "rest", "api", "sql", "docker", "linux", "git",
        "flask", "django"
    ],
    "data_scientist": [
        "python", "pandas", "numpy", "matplotlib", "sql", "data analysis",
        "regression", "classification", "clustering", "scikit-learn",
        "data visualization", "eda"
    ],
    "ai_engineer": [
        "python", "pytorch", "tensorflow", "deep learning", "neural network",
        "nlp", "computer vision", "transformers", "bert", "huggingface"
    ],
    "frontend": [
        "javascript", "typescript", "react", "angular", "html", "css",
        "responsive", "ui", "ux"
    ],
    "mobile": [
        "kotlin", "java", "android", "android studio", "jetpack compose",
        "swift", "react native"
    ],
}

# 4) DistilBERT embedding fonksiyonları
def get_embedding(text: str) -> torch.Tensor:
    """
    Verilen metnin [CLS] token embedding'ini döner.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0)


CATEGORY_SENTENCES = {
    "software": "This resume is for a software engineer with strong programming skills.",
    "ai": "This resume is for an artificial intelligence engineer.",
    "data_science": "This resume is for a data scientist working with data analysis and machine learning.",
    "soft_skills": "This resume shows strong soft skills like teamwork and communication.",
}

CATEGORY_EMBEDDINGS = {
    label: get_embedding(sentence)
    for label, sentence in CATEGORY_SENTENCES.items()
}


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    return F.cosine_similarity(a, b).item()


# 5) Keyword analiz fonksiyonları
def count_keywords(text: str) -> Dict[str, int]:
    text_lower = text.lower()
    counts = {}

    for label, keywords in SKILL_KEYWORDS.items():
        c = 0
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            c += len(re.findall(pattern, text_lower))
        counts[label] = c

    return counts


def extract_present_keywords(text: str) -> List[str]:
    """
    CV'de geçen bütün keyword'leri (unique) liste olarak döner.
    """
    text_lower = text.lower()
    present = set()
    for keywords in SKILL_KEYWORDS.values():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower):
                present.add(kw)
    return sorted(present)


def compute_keyword_scores(text: str) -> Dict[str, int]:
    counts = count_keywords(text)
    max_count = max(counts.values()) if counts else 1
    scores = {}

    for label, c in counts.items():
        if max_count == 0:
            scores[label] = 0
        else:
            scores[label] = int((c / max_count) * 100)

    return scores


def compute_semantic_scores(text: str) -> Dict[str, int]:
    resume_emb = get_embedding(text)
    scores = {}

    for label, cat_emb in CATEGORY_EMBEDDINGS.items():
        sim = cosine_similarity(resume_emb, cat_emb)
        sim = max(0.0, sim)
        scores[label] = int(sim * 100)

    return scores


# 6) Role göre eksik skill önerisi
def recommend_missing_skills(present_keywords: List[str],
                             target_role: Optional[str]) -> List[str]:
    if not target_role:
        return []

    expected = ROLE_EXPECTED_SKILLS.get(target_role, [])
    # expected listte olup CV'de olmayanlar
    missing = [kw for kw in expected if kw not in present_keywords]
    return missing


# 7) Ana fonksiyon: tüm analiz + role göre öneriler
def analyze_resume(text: str, target_role: Optional[str] = None) -> Dict:
    """
    :param text: CV metni
    :param target_role: backend / data_scientist / ai_engineer / frontend / mobile
    """
    keyword_scores = compute_keyword_scores(text)
    semantic_scores = compute_semantic_scores(text)

    final_scores = {}
    for label in CATEGORY_SENTENCES.keys():
        k = keyword_scores.get(label, 0)
        s = semantic_scores.get(label, 0)
        final_scores[label] = int(0.6 * k + 0.4 * s)

    if final_scores:
        overall_score = int(sum(final_scores.values()) / len(final_scores))
    else:
        overall_score = 0

    present_keywords = extract_present_keywords(text)
    role_missing_skills = recommend_missing_skills(present_keywords, target_role)
    suggestions = build_suggestions(final_scores, target_role, role_missing_skills)

    return {
        "scores": final_scores,
        "overall_score": overall_score,
        "present_keywords": present_keywords,
        "suggestions": suggestions,
        "target_role": target_role,
        "recommended_skills": role_missing_skills,
    }


def build_suggestions(scores: Dict[str, int],
                      target_role: Optional[str],
                      missing_skills: List[str]) -> List[str]:
    suggestions = []

    # Genel teknik kısım
    if scores.get("software", 0) < 50:
        suggestions.append(
            "Teknik / yazılım kısmın biraz zayıf görünüyor. Daha fazla programlama dili, framework ve somut proje detayı ekleyebilirsin."
        )
    else:
        suggestions.append(
            "Teknik / yazılım kısmın iyi. Bunu desteklemek için projelerine ölçülebilir çıktılar ekleyebilirsin (örn. %X performans artışı)."
        )

    # AI
    if scores.get("ai", 0) < 40:
        suggestions.append(
            "AI ile ilgili kısım zayıf görünüyor. Makine öğrenmesi / derin öğrenme projelerini, kullandığın kütüphaneleri (PyTorch, TensorFlow vb.) ve metrikleri daha net yazabilirsin."
        )
    else:
        suggestions.append(
            "AI alanında güzel sinyaller var. Model türlerini, veri boyutunu ve değerlendirme metriklerini daha ayrıntılı yazman CV'ni güçlendirir."
        )

    # Data science
    if scores.get("data_science", 0) < 40:
        suggestions.append(
            "Veri bilimi tarafında daha çok vurgu yapabilirsin. SQL, veri analizi, görselleştirme, EDA gibi yeteneklerini projelerle göstermek iyi olur."
        )
    else:
        suggestions.append(
            "Veri bilimi kısmın fena değil. İş problemini, kullandığın yöntemleri ve elde ettiğin iş değerini daha net anlatabilirsin."
        )

    # Soft skills
    if scores.get("soft_skills", 0) < 40:
        suggestions.append(
            "Soft skill kısmı (iletişim, ekip çalışması vb.) zayıf görünüyor. Takım projeleri, rolün, sorumlulukların ve işbirliği örneklerini eklemeyi düşünebilirsin."
        )
    else:
        suggestions.append(
            "Soft skill'lerin iyi yansımış görünüyor. Liderlik, mentorluk veya çatışma çözümü gibi spesifik örnekler eklemek CV'ni daha da öne çıkarır."
        )

    # Role-specific öneri
    if target_role:
        readable = {
            "backend": "Backend Developer",
            "data_scientist": "Data Scientist",
            "ai_engineer": "AI Engineer",
            "frontend": "Frontend Developer",
            "mobile": "Mobile Developer",
        }.get(target_role, target_role)

        if missing_skills:
            suggestions.append(
                f"Seçtiğin pozisyon: {readable}. Bu rol için CV'ne ekleyebileceğin bazı eksik anahtar kelimeler: "
                + ", ".join(missing_skills)
            )
        else:
            suggestions.append(
                f"Seçtiğin pozisyon: {readable}. Bu rol için temel anahtar kelimelerin çoğu CV'nde görünüyor. Projelerini daha somut çıktılarla zenginleştirebilirsin."
            )

    return suggestions
