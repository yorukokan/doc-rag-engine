<div align="center">

# Doc-RAG Engine — Hybrid Search & Evaluation Pipeline

**Framework Bağımsız · Hibrit Arama (Dense + BM25 RRF) · Otomatik Halüsinasyon & Citation Doğrulama**

[![GitHub](https://img.shields.io/badge/GitHub-yorukokan-181717?style=for-the-badge&logo=github)](https://github.com/yorukokan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Okan_Yörük-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/okanyoruk)
[![Email](https://img.shields.io/badge/E--Posta-okanyoruk.tr%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:okanyoruk.tr@gmail.com)

</div>

---

## 📌 Mimari Genel Bakış

Bu proje; LangChain veya LlamaIndex gibi soyutlama katmanlarına bağımlı kalmadan **saf Python** ile sıfırdan geliştirilmiş, üretim (production) standartlarında ölçülebilir bir **Retrieval-Augmented Generation (RAG)** motorudur. Teknik Markdown dokümanlarını anlamsal bütünlüğü koruyarak parçalar, yerel vektör veritabanında indeksler, hibrit arama ile en alakalı bağlamı getirir ve zorunlu kaynak gösterimi (**Citation**) ile yanıt üretir.

```text
Markdown Dokümanları (data/ — 4 Dosya / 16 Chunk)
        │
        ▼
[1. Chunker]   ──► Başlık bazlı (#, ##, ###) parçalama + metadata (source, section)
        │
        ▼
[2. Indexer]   ──► FastEmbed (384-dim Multilingual) ──► Qdrant (Local Disk Storage)
        │
        ▼
[3. Retriever] ──► Dense Vector Search + BM25Okapi ──► RRF (Reciprocal Rank Fusion)
        │
        ▼
[4. Generator] ──► OpenAI-Uyumlu LLM API (temp=0.1) ──► Zorunlu [dosya - Bölüm] Citation
        │
        ▼
[5. Evaluator] ──► 20 Soruluk Otomatik Test Seti (Recall@2, Citation, Halüsinasyon Reddi)
```

### ✨ Temel Mühendislik Özellikleri

- **Yapısal Başlık Parçalama (`src/chunker.py`)**: Sabit karakter kesimi yerine Regex tabanlı başlık ayrıştırma (`#`, `##`, `###`) ve 800 karakter üst limitli paragraf koruma mantığı kullanır. Her parçaya kaynak dosya (`source`) ve alt başlık (`section`) metadatası gömülür.
- **Sıfır Docker Bağımlılıklı Yerel İndeksleme (`src/indexer.py`)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli (FastEmbed) ile üretilen 384 boyutlu vektörler, yerel disk tabanlı **Qdrant** üzerinde Cosine benzerlik metriğiyle saklanır.
- **Hibrit Arama ve RRF Sıralama (`src/retriever.py`)**: Yalnızca anlamsal vektör aramasının (Dense) spesifik teknik terimleri kaçırma riskini ortadan kaldırmak için **BM25Okapi** anahtar kelime skoru ile vektör sıralaması **Reciprocal Rank Fusion (RRF)** formülüyle birleştirilir.
- **Halüsinasyon Koruması ve Zorunlu Citation (`src/generator.py`)**: OpenAI-uyumlu API arayüzü üzerinden düşük sıcaklık (`temperature=0.1`) ve katı sistem kurallarıyla çalışır. Model yalnızca getirilen parçalara dayanır, her bilginin sonuna `[dosya_adi - Bölüm]` kaynağını ekler ve doküman dışı soruları reddeder.

---

## 📊 Değerlendirme & Metrik Sonuçları (Evaluation)

Sistemin arama ve üretim kalitesi `eval/run_eval.py` ile otomatik olarak ölçülmüştür. İlk aşamada (V1) yalnızca yoğun vektör araması kullanılırken kaçırılan teknik kavramlar (ör. *TTL önbellek süresi*), **BM25 + RRF Hibrit Arama (V2)** mimarisine geçildikten sonra tamamen yakalanmıştır.

**Test Kapsamı:** `4 Teknik Doküman` · `16 Bilgi Kartı (Chunk)` · `20 Soru (16 Doküman İçi + 4 Tuzak Soru)`

| Değerlendirme Metriği | V1 (Sadece Vektör) | V2 (Hybrid: Vektör + BM25 RRF) | Durum |
| :--- | :---: | :---: | :---: |
| **Recall@2 (Arama Başarısı)** | %75.0 | **%100.0 (16/16)** | ✅ Tam Başarı |
| **Citation (Kaynak Gösterme Uyumu)** | %100.0 | **%100.0 (16/16)** | ✅ Tam Başarı |
| **Tuzak Soru (Halüsinasyon Reddi)** | %100.0 | **%100.0 (4/4)** | ✅ Sıfır Halüsinasyon |

---

## 📂 Proje Yapısı

```text
doc-rag-engine/
├── data/               # Teknik Markdown dokümanları (DNS, Linux, Ağ Güvenliği, HTTP/API)
├── src/
│   ├── chunker.py      # Başlık bazlı parçalama ve metadata çıkarma motoru
│   ├── indexer.py      # FastEmbed vektörleştirme ve Qdrant yerel indeksleme
│   ├── retriever.py    # Dense Vector + BM25 + RRF Hibrit Arama motoru
│   └── generator.py    # OpenAI-uyumlu LLM ve zorunlu Citation katmanı
├── eval/
│   ├── dataset.json    # 20 soruluk test veri seti (tuzak sorular dahil)
│   └── run_eval.py     # Otomatik metrik hesaplama ve raporlama scripti
├── .env.example        # Ortam değişkenleri şablonu
├── .gitignore
└── requirements.txt
```

---

## 🚀 Kurulum ve Çalıştırma

1. Depoyu klonlayın ve izole Python sanal ortamını aktif edin:

```bash
git clone [https://github.com/yorukokan/doc-rag-engine.git](https://github.com/yorukokan/doc-rag-engine.git)
cd doc-rag-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturup OpenAI-uyumlu LLM uç nokta bilgilerinizi tanımlayın:

```bash
cp .env.example .env
# LLM_BASE_URL, LLM_API_KEY ve LLM_MODEL_NAME değerlerini girin
```

3. Dokümanları indeksleyin, örnek sorgu çalıştırın ve 20 soruluk değerlendirme testini koşturun:

```bash
python3 src/indexer.py      # Dokümanları vektörleştirip Qdrant'a kaydeder
python3 src/generator.py    # Örnek soruyu kaynak göstererek yanıtlar
python3 eval/run_eval.py    # 20 soruluk sınavı çalıştırıp sistem karnesini çıkarır
```
