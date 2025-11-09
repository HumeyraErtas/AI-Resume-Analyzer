# app.py
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader

from resume_analyzer import analyze_resume

app = Flask(__name__)


def extract_text_from_pdf(file_storage) -> str:
    """
    Yüklenen PDF dosyasından tüm metni çıkarır.
    """
    reader = PdfReader(file_storage)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_text = ""

        uploaded_file = request.files.get("resume_file")
        pasted_text = request.form.get("resume_text", "").strip()
        target_role = request.form.get("target_role") or None

        if uploaded_file and uploaded_file.filename != "":
            filename = uploaded_file.filename.lower()
            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif pasted_text:
            resume_text = pasted_text

        if not resume_text:
            return render_template("index.html", error="Lütfen bir PDF yükleyin veya metin girin.")

        results = analyze_resume(resume_text, target_role)

        return render_template(
            "result.html",
            resume_text=resume_text,
            scores=results["scores"],
            overall_score=results["overall_score"],
            present_keywords=results["present_keywords"],
            suggestions=results["suggestions"],
            target_role=results["target_role"],
            recommended_skills=results["recommended_skills"],
        )

    return render_template("index.html")


# 🔹 Ekstra: JSON API endpoint
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    JSON body ile:
    {
      "text": "...cv metni...",
      "target_role": "backend"  # opsiyonel
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    resume_text = data.get("text", "").strip()
    target_role = data.get("target_role")

    if not resume_text:
        return jsonify({"error": "text alanı boş olamaz"}), 400

    results = analyze_resume(resume_text, target_role)
    return jsonify(results), 200


if __name__ == "__main__":
    app.run(debug=True)
