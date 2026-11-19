# Doc-RAG Engine (Hybrid Search + Citation + Evaluation)

Kendi teknik Markdown dokümanları üzerinde çalışan, dış framework (LangChain vb.) kullanmadan saf Python ile geliştirilmiş, **Hybrid Search** ve **otomatik değerlendirme (evaluation)** destekli RAG motoru.

## Özellikler

* **Başlık Bazlı Parçalama (Chunking):** Markdown dokümanlarını `#`, `##`, `###` başlık yapılarını bozmadan böler ve her parçaya kaynak (`source`) ile bölüm (`section`) metadatası ekler.
* **Yerel Vektör İndeksleme:** Docker gerektirmeden yerel diskte çalışan Qdrant ve Türkçe destekli FastEmbed modeli ile tamamen lokal indeksleme yapar.
* **Hybrid Search (Dense + BM25 + RRF):** Anlamsal vektör aramasını BM25 anahtar kelime aramasıyla birleştirip Reciprocal Rank Fusion (RRF) ile sıralar.
* **Zorunlu Kaynak Gösterimi (Citation) ve Halüsinasyon Kontrolü:** LLM yalnızca getirilen parçaları kullanarak cevap verir, her bilginin sonuna `[dosya_adi - Bölüm]` kaynağını ekler ve dokümanda olmayan soruları reddeder.
* **Otomatik Değerlendirme (Evaluation):** Recall@K, kaynak uyumu ve tuzak soru (halüsinasyon) testlerini otomatik olarak ölçer.

## Mimari Akış

```text
Markdown Dokümanları (data/)
        │
        ▼
[1. Chunker] ──► Başlık bazlı parçalama + metadata (source, section)
        │
        ▼
[2. Indexer] ──► Local Embedding (FastEmbed) ──► Qdrant (Local Storage)
        │
        ▼
[3. Retriever] ─► Dense Vector Search + BM25 ──► RRF Birleştirme
        │
        ▼
[4. Generator] ─► OpenAI-uyumlu LLM API ──► Zorunlu Kaynak (Citation) + Halüsinasyon Kontrolü
```

## Proje Yapısı

```text
doc-rag-engine/
├── data/               # Teknik Markdown dokümanları (4 dosya / 16 chunk)
├── src/
│   ├── chunker.py      # Başlık bazlı parçalama ve metadata çıkarma
│   ├── indexer.py      # FastEmbed ile vektörleştirme ve Qdrant indeksleme
│   ├── retriever.py    # Hybrid Search (Vektör + BM25 + RRF) motoru
│   └── generator.py    # OpenAI-uyumlu LLM ve Citation katmanı
├── eval/
│   ├── dataset.json    # 20 soruluk değerlendirme seti (16 soru + 4 tuzak soru)
│   └── run_eval.py     # Otomatik metrik hesaplama scripti
├── .env.example        # Örnek ortam değişkenleri
├── .gitignore
└── requirements.txt
```

## Ölçüm Sonuçları (Evaluation)

Başlangıç aşamasında (V1) yalnızca yoğun vektör araması (Dense Vector Search) kullanıldığında bazı spesifik teknik kavramlar (ör. TTL) ilk sıralarda kaçırılırken, BM25 + RRF tabanlı **Hybrid Search (V2)** mimarisine geçildikten sonra tüm test setinde tam başarı elde edilmiştir.

**Test Kapsamı:** 4 doküman, 16 chunk, 20 soru (16 doküman içi teknik soru + 4 doküman dışı tuzak soru)

| Metrik | Sonuç (Hybrid V2) |
| :--- | :---: |
| **Recall@2 (Arama Başarısı)** | **%100.0 (16/16)** |
| **Citation (Kaynak Gösterme Uyumu)** | **%100.0 (16/16)** |
| **Tuzak Soru (Halüsinasyon Reddi)** | **%100.0 (4/4)** |

## Kurulum ve Çalıştırma

1. Sanal ortamı kurun ve bağımlılıkları yükleyin:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. `.env` dosyasını oluşturup kendi OpenAI-uyumlu LLM bilgilerinizi girin:

```bash
cp .env.example .env
```

3. Dokümanları indeksleyin, örnek sorgu çalıştırın ve değerlendirme testini koşturun:

```bash
python3 src/indexer.py      # Dokümanları vektörleştirip Qdrant'a yazar
python3 src/generator.py    # Örnek soruyu RAG ile yanıtlar
python3 eval/run_eval.py    # 20 soruluk sınavı koşturup karne çıkarır
```
