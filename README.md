<div align="center">

# Doc-RAG Engine — Two-Stage Hybrid Search, Reranking & Evaluation Pipeline

**Framework Bağımsız · Hibrit Arama (Dense + BM25 RRF) · Multilingual Cross-Encoder Reranker · LLM-as-a-Judge & Observability**

[![GitHub](https://img.shields.io/badge/GitHub-yorukokan-181717?style=for-the-badge&logo=github)](https://github.com/yorukokan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Okan_Yörük-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/okanyoruk)
[![Email](https://img.shields.io/badge/E--Posta-okanyoruk.tr%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:okanyoruk.tr@gmail.com)

</div>

---

## 📌 Mimari Genel Bakış

Bu proje; LangChain veya LlamaIndex gibi soyutlama katmanlarına bağımlı kalmadan **saf Python** ile sıfırdan geliştirilmiş, üretim (production) standartlarında ölçülebilir bir **Retrieval-Augmented Generation (RAG)** motorudur. Markdown, düz metin ve PDF dokümanlarını anlamsal bütünlüğü koruyarak (**Chunk Overlap**) parçalar, yerel vektör veritabanında indeksler, çok turlu sohbetlerde **Sliding Window Hafıza** ve **Query Rewriting** ile takip sorularını bağımsız arama sorgularına dönüştürür, iki aşamalı arama (**Hybrid Search + Cross-Encoder Reranking**) ile en alakalı bağlamı getirir ve zorunlu kaynak gösterimi (**Citation**) ile yanıt üretir.

```text
Dokümanlar (data/ — .md, .txt, .pdf)
        │
        ▼
[1. Chunker]           ──► Başlık/Paragraf bazlı parçalama + Chunk Overlap + Metadata (source, section)
        │
        ▼
[2. Indexer]           ──► FastEmbed (384-dim Multilingual) ──► Qdrant (Local Disk Storage)
        │
        ▼
[3. Memory & Rewriter] ──► Sliding Window Hafıza (max_turns=5) + Takip Sorusu Yeniden Yazma
        │                  (1. turda 0.0 ms atlama; takip turlarında zamir/bağlam çözme)
        ▼
[4. Retriever]         ──► Stage 1: Dense Vector Search + BM25Okapi ──► RRF (Reciprocal Rank Fusion)
        │
        ▼
[5. Reranker]          ──► Stage 2: Multilingual Cross-Encoder (bge-reranker-v2-m3-int8) ──► Hassas Sıralama
        │
        ▼
[6. Generator]         ──► OpenAI-Uyumlu LLM API (temp=0.1, geçmiş farkında) ──► Zorunlu [dosya - Bölüm] Citation + Süre Ölçümü
        │
        ▼
[7. CLI / Eval]        ──► Etkileşimli Röntgen CLI (--debug, --rerank, --no-rewrite, --max-turns) & 20 Soruluk LLM-as-a-Judge Karnesi
```

### ✨ Temel Mühendislik Özellikleri

- **Çoklu Format ve Chunk Overlap (`src/chunker.py`)**: `.md`, `.txt` ve `.pdf` (`pypdf`) dosyalarını destekler. Başlık (`#`, `##`, `###`) ve paragraf sınırlarını koruyarak 800 karakterlik parçalara ayırır; uzun metinlerde bağlam kopmasını önlemek için **120 karakterlik Chunk Overlap** uygular.
- **Sıfır Docker Bağımlılıklı Yerel İndeksleme (`src/indexer.py`)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli (FastEmbed) ile üretilen 384 boyutlu vektörler, yerel disk tabanlı **Qdrant** üzerinde Cosine benzerlik metriğiyle saklanır.
- **Sliding Window Sohbet Hafızası (`src/memory.py`)**: `ConversationMemory` sınıfı son `N` sohbet turunu (varsayılan `max_turns=5`) kayan pencere ile saklar; sınır aşıldığında en eski tur düşer ve `get_messages()` dış değişikliklere karşı izole edilmiş mesaj kopyası döndürür.
- **Bağlam Farkında Query Rewriting (`src/rewriter.py`)**: Takip sorularındaki zamirleri (`o`, `bunu`, `oradaki`, `peki bu` vb.) ve eksik bağlamı sohbet geçmişini kullanarak `temperature=0.0` ile tek satırlık bağımsız (standalone) arama sorgusuna dönüştürür. 1. turda (geçmiş boşken) LLM çağrısı yapmadan **0.0 ms** gecikmeyle atlanır; boş çıktı veya API hatasında uyarı vererek orijinal sorguya geri döner.
- **İki Aşamalı Hibrit Arama + Reranking (`src/retriever.py`, `src/reranker.py`)**:
  - **1. Aşama (Hızlı Eleme):** Anlamsal vektör araması (Dense) ile anahtar kelime araması (**BM25Okapi**) sonuçları **Reciprocal Rank Fusion (RRF)** formülüyle birleştirilir.
  - **2. Aşama (Hassas Sıralama):** Aday kartlar Türkçe destekli yerel Cross-Encoder (`BAAI/bge-reranker-v2-m3-int8`) ile soruyla yan yana okunarak yeniden puanlanır.
- **Halüsinasyon Koruması, Citation ve Süre İzleme (`src/llm_client.py`, `src/generator.py`)**: Ortak `.env` yapılandırmalı OpenAI-uyumlu istemci üzerinden düşük sıcaklık (`temperature=0.1`) ve katı sistem kurallarıyla çalışır. Sohbet geçmişini yalnızca konuşma akışını anlamak için kullanır; tüm olgusal bilgileri o turda getirilen `DOKÜMANLAR` bölümünden alır, her bilginin sonuna `[dosya_adi - Bölüm]` kaynağı ekler, doküman dışı soruları reddeder ve `rewrite_ms`, `retrieval_ms`, `generation_ms` sürelerini ölçer.
- **Etkileşimli Röntgen CLI (`src/cli.py`)**: `--debug`, `--search-only`, `--rerank`, `--top-k`, `--query`, `--no-rewrite` ve `--max-turns` bayraklarını destekler; her soruda (varsa) yeniden yazılan sorguyu, Vektör (cosine), BM25, RRF, Rerank skorlarını ve Rewrite/Arama/Üretim gecikmelerini canlı gösterir.
- **Otomatik Değerlendirme ve LLM-as-a-Judge (`eval/run_eval.py`)**: `Recall@1`, `Recall@2`, Citation uyumu, tuzak soru reddi, ortalama latency ve hakem LLM üzerinden **Sadakat (Faithfulness)** & **Cevap Alakası (Relevance)** metriklerini ölçer.

---

## 📊 Değerlendirme & Metrik Sonuçları (Evaluation)

Sistemin arama ve üretim kalitesi `eval/run_eval.py --rerank --judge` ile otomatik olarak ölçülmüştür.

**Test Kapsamı:** `20 Soru (16 Doküman İçi + 4 Tuzak Soru)`

| Değerlendirme Metriği | V1 (Sadece Vektör) | V2 (Hybrid: Vektör + BM25 RRF) | V3 (Hybrid + Cross-Encoder Rerank) | Durum |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1 (Tam 1. Sırada Bulma)** | — | — | **%100.0 (16/16)** | ✅ Tam İsabet |
| **Recall@2 (İlk 2'de Bulma)** | %75.0 | %100.0 (16/16) | **%100.0 (16/16)** | ✅ Tam Başarı |
| **Citation (Kaynak Gösterme Uyumu)** | %100.0 | %100.0 (16/16) | **%100.0 (16/16)** | ✅ Tam Başarı |
| **Tuzak Soru (Halüsinasyon Reddi)** | %100.0 | %100.0 (4/4) | **%100.0 (4/4)** | ✅ Sıfır Halüsinasyon |
| **LLM-as-a-Judge: Sadakat (Faithfulness)** | — | — | **5.00 / 5.00** | ✅ Tam Sadakat |
| **LLM-as-a-Judge: Cevap Alakası** | — | — | **5.00 / 5.00** | ✅ Tam Alaka |

---

## 📂 Proje Yapısı

```text
doc-rag-engine/
├── data/                              # Teknik dokümanlar (.md, .txt, .pdf)
├── src/
│   ├── chunker.py                     # Çoklu format (.md/.txt/.pdf) + Chunk Overlap motoru
│   ├── indexer.py                     # FastEmbed vektörleştirme ve Qdrant yerel indeksleme
│   ├── llm_client.py                  # .env tabanlı ortak OpenAI-uyumlu LLM istemcisi
│   ├── memory.py                      # Sliding window tabanlı sohbet hafızası (ConversationMemory)
│   ├── rewriter.py                    # Bağlam farkında takip sorusu yeniden yazma (Query Rewriter)
│   ├── retriever.py                   # Dense Vector + BM25 + RRF Hibrit Arama motoru
│   ├── reranker.py                    # Multilingual Cross-Encoder 2. aşama sıralayıcı
│   ├── generator.py                   # OpenAI-uyumlu LLM, Citation ve süre ölçüm katmanı
│   └── cli.py                         # Etkileşimli sohbet ve Observability (Röntgen) CLI
├── eval/
│   ├── dataset.json                   # 20 soruluk test veri seti (16 normal + 4 tuzak soru)
│   └── run_eval.py                    # Recall@1/2, Citation, Latency ve LLM-as-a-Judge scripti
├── tests/
│   └── test_memory_and_rewriter.py    # Sohbet hafızası, query rewriter ve generator birim testleri
├── .env.example                       # Ortam değişkenleri şablonu
├── .gitignore
└── requirements.txt
```

---

## 🚀 Kurulum ve Çalıştırma

1. Depoyu klonlayın ve izole Python sanal ortamını aktif edin:

```bash
git clone https://github.com/yorukokan/doc-rag-engine.git
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

3. Dokümanları indeksleyin, etkileşimli CLI ile röntgen çekin ve değerlendirme testini çalıştırın:

```bash
python3 src/indexer.py                                        # Dokümanları vektörleştirip Qdrant'a kaydeder
python3 src/cli.py --debug --rerank                           # Etkileşimli sohbet + Röntgen + Reranker (varsayılan: rewrite açık, max-turns=5)
python3 src/cli.py --debug --rerank --max-turns 3             # Sohbet hafızasını son 3 turla sınırlar
python3 src/cli.py --debug --rerank --no-rewrite              # Query rewriting'i devre dışı bırakır
python3 src/cli.py --search-only --rerank                     # LLM çağırmadan sadece arama röntgeni
python3 eval/run_eval.py --rerank --judge                     # 20 soruluk tam değerlendirme ve hakem karnesi
```

4. **CLI Seçenekleri ve Etkileşimli Komutlar:**
- `--no-rewrite`: Takip sorularında query rewriting adımını kapatır; sorgu yeniden yazılmadan aranır, sohbet geçmişi yalnızca yanıtlama aşamasına aktarılır.
- `--max-turns <N>`: Sohbet hafızasında tutulacak maksimum tur sayısını belirler (varsayılan: `5`, en az `1`).
- `/history`: Hafızadaki sohbet geçmişini tur bazında özetleyerek listeler.
- `/clear` veya `/reset`: Aktif oturumdaki sohbet hafızasını sıfırlar.
- `q`, `exit` veya `quit`: Etkileşimli sohbetten çıkış yapar.

5. **Birim Testlerini Çalıştırma:**

```bash
.venv/bin/python -m unittest discover -s tests -v
# veya aktif sanal ortam içinde:
python3 -m unittest discover -s tests -v
```
