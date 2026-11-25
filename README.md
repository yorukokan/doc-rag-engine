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
[2. Indexer]           ──► FastEmbed (384-dim Multilingual, get_embedder) ──► Qdrant (Local Disk + Artımlı İndeksleme)
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
[6. Agentic RAG (Ops.)]──► (--agent / use_agent=True) Planla → Ara → Değerlendir Sınırlı Döngüsü (max_steps)
        │                  Aksiyonlar: search | answer | direct_chat | fallback
        │                  seen_queries tekrar koruması + chunk_id tekilleştirme + agent_steps izleme
        ▼
[7. Guardrail]         ──► Erken Reddetme Kapısı (rerank < -3.0 veya dense < 0.45 & bm25 < 7.0)
        │                  (Eşik altı: LLM atlanır, 0.0 ms ile sabit reddetme; hafızaya yazılmaz)
        ▼
[8. Generator]         ──► OpenAI-Uyumlu LLM API (temp=0.1, geçmiş farkında, tam yanıt & SSE akışı) ──► Zorunlu [dosya - Bölüm] Citation + Süre Ölçümü
        │
        ▼
[9. CLI / API / UI / Eval]  ──► Etkileşimli Röntgen CLI (--debug, --rerank, --agent, --max-agent-steps, --no-rewrite, --no-guardrail, --max-turns)
                                FastAPI REST API (src/api.py: /health, /ask, /ask/stream [SSE: meta → token → done / error], /search, /sessions/{session_id}, /documents, Swagger /docs)
                                Görsel Web Arayüzü (src/ui.py + src/ui_client.py -> Streamlit @ http://localhost:8501, ⚡ Canlı Akış & 📚 Bilgi Bankası)
                                20 Soruluk LLM-as-a-Judge Karnesi (eval/run_eval.py)
                                Çok Turlu & Multi-Hop Agentic Değerlendirme Karnesi (eval/run_advanced_eval.py)
```

### ✨ Temel Mühendislik Özellikleri

- **Çoklu Format ve Chunk Overlap (`src/chunker.py`)**: `.md`, `.txt` ve `.pdf` (`pypdf`) dosyalarını destekler. Başlık (`#`, `##`, `###`) ve paragraf sınırlarını koruyarak 800 karakterlik parçalara ayırır; uzun metinlerde bağlam kopmasını önlemek için **120 karakterlik Chunk Overlap** uygular.
- **Sıfır Docker Bağımlılıklı Yerel İndeksleme (`src/indexer.py`)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli (FastEmbed) ile üretilen 384 boyutlu vektörler, yerel disk tabanlı **Qdrant** üzerinde Cosine benzerlik metriğiyle saklanır.
- **Dinamik Doküman Yükleme & Canlı İndeksleme (`src/indexer.py`, `src/api.py`, `src/ui.py`)**: Yeniden tam indeksleme gerektirmeden çalışma zamanında `.md`, `.txt` ve `.pdf` dokümanlarını (maks. **10 MB**, `MAX_UPLOAD_SIZE_BYTES`) yükler, parçalar ve canlı olarak Qdrant koleksiyonuna ekler veya kaldırır (`python-multipart` ile `multipart/form-data` desteği).
  - `validate_filename` ile dosya adı uzunluğu (maks. 120 karakter), yol geçişi (`..`, `/`, `\`), kontrol karakteri, gizli dosya (`.`) ve izin verilen uzantı kontrolleri uygulanır.
  - `make_point_id` her parça için `uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}")` tabanlı **deterministik UUID** üretir; `index_single_file` ve `delete_document` paylaşılan `get_embedder()` singleton'ı ve `_qdrant_lock` kilidi altında yalnızca ilgili `source` kaydına ait noktaları günceller veya siler.
  - `save_and_index_document` içeriği geçici dosyada `chunk_file` ile ön doğrulamadan geçirip boş/okunamayan dosyaları reddeder, `overwrite=False` iken mevcut dosya çakışmalarını engeller ve indeksleme başarısız olursa diski önceki durumuna geri alır (**rollback**).
- **Sliding Window Sohbet Hafızası (`src/memory.py`)**: `ConversationMemory` sınıfı son `N` sohbet turunu (varsayılan `max_turns=5`) kayan pencere ile saklar; sınır aşıldığında en eski tur düşer ve `get_messages()` dış değişikliklere karşı izole edilmiş mesaj kopyası döndürür.
- **Bağlam Farkında Query Rewriting (`src/rewriter.py`)**: Takip sorularındaki zamirleri (`o`, `bunu`, `oradaki`, `peki bu` vb.) ve eksik bağlamı sohbet geçmişini kullanarak `temperature=0.0` ile tek satırlık bağımsız (standalone) arama sorgusuna dönüştürür. 1. turda (geçmiş boşken) LLM çağrısı yapmadan **0.0 ms** gecikmeyle atlanır; boş çıktı veya API hatasında uyarı vererek orijinal sorguya geri döner.
- **İki Aşamalı Hibrit Arama + Reranking (`src/retriever.py`, `src/reranker.py`)**:
  - **1. Aşama (Hızlı Eleme):** Anlamsal vektör araması (Dense) ile anahtar kelime araması (**BM25Okapi**) sonuçları **Reciprocal Rank Fusion (RRF)** formülüyle birleştirilir.
  - **2. Aşama (Hassas Sıralama):** Aday kartlar Türkçe destekli yerel Cross-Encoder (`BAAI/bge-reranker-v2-m3-int8`) ile soruyla yan yana okunarak yeniden puanlanır.
- **Erken Reddetme Kapısı / Guardrail (`src/guardrail.py`)**: Reranker/Retriever ile Generator arasında konumlanır. Parça dönmezse veya en yüksek skorlar belirlenen eşiklerin altında kalırsa (`--rerank` açıkken `rerank_score < -3.0`; kapalıyken veya `rerank_score` yokken `dense_score < 0.45` ve `bm25_score < 7.0`) LLM'i hiç çağırmadan **0.0 ms** üretim süresiyle sabit reddetme mesajı (`"Verilen dokümanlarda bu bilgi bulunmamaktadır."`) döndürür. Erken reddedilen sorular `ConversationMemory` sohbet hafızasına yazılmaz.
- **Çok Adımlı Agentic RAG (`src/agent.py`)**: `use_agent=True` (`--agent`) etkinleştirildiğinde `run_agent_rag` yürütücüsü saf Python ile sınırlı bir **Planla → Ara → Değerlendir** döngüsü (varsayılan `max_steps=3`) çalıştırır. `temperature=0.0` ile çalışan JSON planlayıcı her adımda dört aksiyon yolundan birini izler:
  - `search`: Çok parçalı veya karşılaştırmalı soruları odaklanmış alt sorgulara böler; `seen_queries` normalizasyonuyla aynı sorgunun tekrar aranmasını engeller ve sonuçları `chunk_id` (yoksa kaynak/bölüm/metin özeti) üzerinden tekilleştirerek (`merge_and_deduplicate_chunks`) en yüksek skorlu kayıtları biriktirir (birden fazla arama yapıldığında final bağlam sınırı `top_k * 2`, tek aramada `top_k`).
  - `answer`: Toplanan doküman parçaları yeterli görüldüğünde döngüyü sonlandırıp sentez aşamasına geçer.
  - `direct_chat`: Yalnızca 1. adımda ve henüz hiç arama yapılmamışken selamlaşma veya gündelik sohbet mesajlarında arama yapmadan doğrudan yanıt üretir (`temperature=0.3`).
  - `fallback`: Planlayıcı LLM çağrısı hata verirse, geçerli JSON döndürmezse veya henüz parça yokken yeni alt sorgu üretilemezse yeniden yazılan/orijinal sorgu ile güvenli geri dönüş araması yapar (parça varsa mevcut bağlamla senteze geçer).
  - Sentez öncesinde (direct chat hariç) **Guardrail** kontrolü uygulanır ve final yanıt `temperature=0.1` ile zorunlu `[dosya_adi - Bölüm]` citation kuralıyla üretilir; her karar adımı (`step`, `thought`, `action`, `query`, `chunks_found`, `duration_ms`) `agent_steps` listesiyle izlenebilir.
- **Halüsinasyon Koruması, Citation ve Süre İzleme (`src/llm_client.py`, `src/generator.py`)**: Ortak `.env` yapılandırmalı OpenAI-uyumlu istemci üzerinden düşük sıcaklık (`temperature=0.1`) ve katı sistem kurallarıyla çalışır. Sohbet geçmişini yalnızca konuşma akışını anlamak için kullanır; tüm olgusal bilgileri o turda getirilen `DOKÜMANLAR` bölümünden alır, her bilginin sonuna `[dosya_adi - Bölüm]` kaynağı ekler, doküman dışı soruları reddeder ve `rewrite_ms`, `retrieval_ms`, `generation_ms` sürelerini ölçer (`use_agent=True` olduğunda `run_agent_rag` akışına yönlendirir).
- **Etkileşimli Röntgen CLI (`src/cli.py`)**: `--debug`, `--search-only`, `--rerank`, `--top-k`, `--query`, `--no-rewrite`, `--no-guardrail`, `--min-rerank-score`, `--min-dense-score`, `--min-bm25-score`, `--max-turns`, `--agent` ve `--max-agent-steps` bayraklarını destekler; her soruda (varsa) yeniden yazılan sorguyu, ajan karar adımlarını (`Ajan Adımları`), Guardrail erken reddetme bilgisini, Vektör (cosine), BM25, RRF, Rerank skorlarını ve Rewrite/Arama/Üretim gecikmelerini canlı gösterir.
- **REST API & Oturum Yönetimi (`src/api.py`)**: FastAPI tabanlı üretim uyumlu HTTP katmanı sunar; otomatik OpenAPI/Swagger (`/docs`) dokümantasyonu üretir. Bellek içi `SessionStore` (`threading.Lock` + `OrderedDict`, varsayılan `max_sessions=100`, `default_max_turns=5`) ile thread-safe ve LRU tahliyeli çok turlu oturum yönetimi sağlar (erken reddedilen sorular oturuma kaydedilmez). `src/retriever.py` içindeki `_qdrant_lock` ve `finally: client.close()` yapısı ile yerel disk tabanlı Qdrant erişiminde eşzamanlı kilit çakışmasını önler. Pydantic v2 doğrulaması (`question`/`query` boşluk kontrolü, `top_k` `1..20`, `candidate_k` `1..50`, `rrf_k >= 1`, `max_agent_steps` `1..10`, `session_id` `^[A-Za-z0-9_-]{1,128}$`) uygular; `POST /ask` `use_agent` ve `max_agent_steps` parametrelerini destekleyip yanıtta `agent_steps` döndürür. `POST /ask/stream` uç noktası aynı `AskRequest` şemasıyla **Server-Sent Events (`text/event-stream`)** üzerinden canlı akış sağlar (`meta` → `token` → `done` / `error`). Ayrıca `GET /documents`, `POST /documents` (`multipart/form-data`, `?overwrite=true`, `201` / `400` / `409` / `413` / `503`) ve `DELETE /documents/{filename}` (`400` / `404` / `503`) uç noktalarıyla canlı doküman yönetimi sunar. `/health`, `502`/`503` ve SSE `error` yanıtlarında `LLM_API_KEY`, `LLM_BASE_URL` veya iç istisna detaylarını dışarı sızdırmaz.
- **Görsel Web Arayüzü (`src/ui.py`, `src/ui_client.py`)**: Streamlit tabanlı web arayüzü, `httpx` ile çalışan `src/ui_client.py` istemcisi üzerinden FastAPI arka ucuna (`http://127.0.0.1:8000`) bağlanır. Kenar çubuğunda canlı `/health` durum rozeti (Qdrant koleksiyonu, indekslenen parça sayısı, LLM modeli), **📚 Bilgi Bankası (Dokümanlar)** paneli (sürükle-bırak `.md`/`.txt`/`.pdf` yükleme, `Aynı isimde varsa üzerine yaz` seçeneği, `📤 Yükle ve İndeksle`, yüklü doküman/parça özeti, `🔄 Yenile` ve `🗑️` ile tek tıkla silme), **Tam RAG Sohbet** (`/ask` ve `/ask/stream`) ve **Sadece Arama (Röntgen)** (`/search`) çalışma modları, `top_k` (`1..10`) / Rerank / Rewrite / Guardrail anahtarları, **🤖 Agentic RAG (Çok Adımlı Ajan)** anahtarı (`Maksimum Ajan Adımı` `1..5` kaydırıcısı), varsayılan açık **⚡ Canlı Akış (Streaming)** anahtarı (`ask_question_stream` üzerinden kelime/token bazlı canlı yanıt akışı), gelişmiş Guardrail eşik ayarları (`min_rerank_score`, `min_dense_score`, `min_bm25_score`) ve sunucu oturumunu sıfırlayan sohbet temizleme kontrolü (`DELETE /sessions/{session_id}`) sunar. Ana ekranda başlangıç için 4 adet tıklanabilir örnek soru kartı (çok parçalı ajan testi ve tuzak soru dahil) ve her yanıtın altında Rewrite/Arama/LLM sürelerini, **🤖 Ajan Karar Adımları** kartlarını, yeniden yazılan sorguyu, Guardrail gerekçesini ve Rerank/RRF/Vektör/BM25 skorlu kaynak parçalarını gösteren katlanabilir **🔬 Röntgen (Observability)** paneli bulunur.
- **Otomatik Değerlendirme ve LLM-as-a-Judge (`eval/run_eval.py`)**: `Recall@1`, `Recall@2`, Citation uyumu, tuzak soru reddi, Guardrail erken reddetme / yanlış pozitif sayıları, ortalama latency ve hakem LLM üzerinden **Sadakat (Faithfulness)** & **Cevap Alakası (Relevance)** metriklerini ölçer (`--no-guardrail` ile kapısız karşılaştırmayı destekler).
- **Çok Turlu (Multi-Turn) & Multi-Hop Agentic Değerlendirme Karnesi (`eval/run_advanced_eval.py`, `eval/metrics.py`, `eval/advanced_dataset.json`)**:
  - **Çok Turlu Senaryolar (`multi_turn`)**: `ConversationMemory(max_turns=5)` ve `Query Rewriter` akışıyla **4 diyalog / 12 tur** (8 cevaplanabilir + 4 tuzak takip sorusu) üzerinde 1. turda sıfır rewrite uyumunu (`turn1_zero_rewrite_ok`), takip turlarında zamir çözümleme anahtar kelime oranını (`rewrite_keyword_rate`), bölüm/kaynak kapsamını, erken reddedilen turların hafızaya eklenmemesini ve Guardrail tuzak yakalama / yanlış pozitif oranlarını ölçer.
  - **Multi-Hop Karşılaştırma (`multi_hop`)**: **8 multi-hop soru** (5 `cross_source` + 1 `multi_section` çoklu-parça ve 2 `trap` tuzak soru) üzerinde `Classic RAG` (`use_agent=False`) ile `Agentic RAG` (`use_agent=True`) modlarını yan yana çalıştırır; çoklu kaynak citation uyumunu (`compute_multi_source_citation`), tam/ortalama bölüm & kaynak kapsamını, ortalama ajan/arama adımlarını ve `compute_mode_delta` ile **`Agentic - Classic`** delta karnesini hesaplar.
  - **Hakem & JSON Rapor**: Opsiyonel `--judge` ile cevaplanabilir sorularda `faithfulness`, `relevance`, `completeness` (1–5) ve tuzak sorularda `refused_properly` puanlaması yapar; `--output` parametresiyle meta, özet, delta ve detaylı kayıtları UTF-8 **JSON rapor** (`save_report`) olarak dışa aktarır.

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

### Çok Turlu (Multi-Turn) & Multi-Hop Agentic Değerlendirme Karnesi

`eval/run_advanced_eval.py` ve `eval/advanced_dataset.json` ile iki ek boyutta kapsamlı değerlendirme yapılır:
- **Çok Turlu Sohbet Karnesi (`--mode multi-turn`)**: 4 diyalog / 12 tur (zamir çözümlemeli takip soruları + tuzak takip soruları) üzerinden rewrite başarısı, bölüm/kaynak kapsamı, citation, Guardrail tuzak yakalama ve gecikme dağılımını raporlar.
- **Multi-Hop Karşılaştırma Karnesi (`--mode multi-hop`)**: 8 multi-hop soruda (6 çoklu-parça + 2 tuzak) **Classic RAG** vs **Agentic RAG** sonuçlarını ve **`Agentic - Classic`** delta farklarını (`full_section_coverage_delta`, `full_source_coverage_delta`, `citation_full_delta`, `total_latency_ms_delta`, `completeness_delta` vb.) üretir; `--output` ile JSON rapor kaydeder.

---

## 📂 Proje Yapısı

```text
doc-rag-engine/
├── data/                              # Teknik dokümanlar (.md, .txt, .pdf)
├── src/
│   ├── chunker.py                     # Çoklu format (.md/.txt/.pdf) + Chunk Overlap motoru
│   ├── indexer.py                     # FastEmbed vektörleştirme, Qdrant yerel ve artımlı doküman indeksleme/silme
│   ├── llm_client.py                  # .env tabanlı ortak OpenAI-uyumlu LLM istemcisi
│   ├── memory.py                      # Sliding window tabanlı sohbet hafızası (ConversationMemory)
│   ├── rewriter.py                    # Bağlam farkında takip sorusu yeniden yazma (Query Rewriter)
│   ├── retriever.py                   # Dense Vector + BM25 + RRF Hibrit Arama motoru
│   ├── reranker.py                    # Multilingual Cross-Encoder 2. aşama sıralayıcı
│   ├── guardrail.py                   # Skor eşiği tabanlı Erken Reddetme Kapısı (Guardrail)
│   ├── agent.py                       # Planla → Ara → Değerlendir döngülü Agentic RAG katmanı
│   ├── generator.py                   # OpenAI-uyumlu LLM, Citation ve süre ölçüm katmanı
│   ├── cli.py                         # Etkileşimli sohbet ve Observability (Röntgen) CLI
│   ├── api.py                         # FastAPI REST API (/ask, /search, /sessions, /documents), Pydantic v2 ve LRU SessionStore
│   ├── ui_client.py                   # Streamlit arayüzü için httpx tabanlı FastAPI istemcisi
│   └── ui.py                          # Streamlit Görsel Web Arayüzü, 📚 Bilgi Bankası ve 🔬 Röntgen paneli
├── eval/
│   ├── advanced_dataset.json          # 4 diyalog (12 tur) + 8 multi-hop soruluk gelişmiş değerlendirme seti
│   ├── dataset.json                   # 20 soruluk test veri seti (16 normal + 4 tuzak soru)
│   ├── metrics.py                     # Veri seti şema doğrulama, kapsam/citation metrikleri ve delta hesaplama
│   ├── run_advanced_eval.py           # Çok turlu & Multi-Hop Classic vs Agentic değerlendirme ve JSON rapor scripti
│   └── run_eval.py                    # Recall@1/2, Citation, Latency ve LLM-as-a-Judge scripti
├── tests/
│   ├── test_agent.py                  # Agentic RAG planlayıcı, tekilleştirme, fallback ve entegrasyon testleri
│   ├── test_api.py                    # REST API endpoint, doğrulama, LRU oturum ve sır izolasyonu testleri
│   ├── test_documents.py              # Dinamik doküman doğrulama, artımlı indeksleme, /documents API ve ui_client testleri
│   ├── test_eval.py                   # Gelişmiş eval veri seti tutarlılığı, metrik, hakem, çok turlu ve delta testleri
│   ├── test_guardrail.py              # Erken reddetme kapısı, generator ve CLI hafıza testleri
│   ├── test_memory_and_rewriter.py    # Sohbet hafızası, query rewriter ve generator birim testleri
│   ├── test_streaming.py              # Generator/Agent akışı, POST /ask/stream (SSE) ve ui_client akış testleri
│   └── test_ui_client.py              # FastAPI istemcisi (ui_client) MockTransport birim testleri
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
python3 src/cli.py --debug --rerank                           # Etkileşimli sohbet + Röntgen + Reranker (varsayılan: rewrite ve guardrail açık, max-turns=5)
python3 src/cli.py --debug --rerank --agent --max-agent-steps 3  # Çok adımlı Agentic RAG + Röntgen + Reranker
python3 src/cli.py --debug --rerank --max-turns 3             # Sohbet hafızasını son 3 turla sınırlar
python3 src/cli.py --debug --rerank --no-rewrite              # Query rewriting'i devre dışı bırakır
python3 src/cli.py --debug --rerank --no-guardrail            # Erken reddetme kapısını (guardrail) devre dışı bırakır
python3 src/cli.py --debug --rerank --min-rerank-score -2.5   # Rerank guardrail eşiğini özelleştirir
python3 src/cli.py --debug --min-dense-score 0.5 --min-bm25-score 8.0  # Rerank kapalıyken dense/BM25 eşiklerini özelleştirir
python3 src/cli.py --search-only --rerank                     # LLM çağırmadan sadece arama röntgeni
python3 eval/run_eval.py --rerank --judge                     # 20 soruluk tam değerlendirme ve hakem karnesi
python3 eval/run_eval.py --rerank --no-guardrail              # Guardrail kapalıyken değerlendirme çalıştırır
python3 eval/run_advanced_eval.py --mode all --rerank --judge --output eval/report.json  # Çok turlu + Multi-Hop tam karne ve JSON çıktı
python3 eval/run_advanced_eval.py --mode multi-turn --rerank                              # Yalnızca 4 diyalog / 12 turluk çok turlu karne
python3 eval/run_advanced_eval.py --mode multi-hop --rerank --judge                       # Yalnızca 8 soruluk Classic vs Agentic karşılaştırması
```

4. **CLI Seçenekleri ve Etkileşimli Komutlar:**
- `--no-rewrite`: Takip sorularında query rewriting adımını kapatır; sorgu yeniden yazılmadan aranır, sohbet geçmişi yalnızca yanıtlama aşamasına aktarılır.
- `--no-guardrail`: Erken reddetme kapısını (guardrail) devre dışı bırakır; düşük skorlarda da bağlam LLM'e gönderilir (`eval/run_eval.py --no-guardrail` için de geçerlidir).
- `--min-rerank-score <SKOR>`: Reranker aktifken guardrail eşik değerini belirler (varsayılan: `-3.0`).
- `--min-dense-score <SKOR>`: Reranker kapalıyken (veya `rerank_score` yokken) dense cosine eşik değerini belirler (varsayılan: `0.45`).
- `--min-bm25-score <SKOR>`: Reranker kapalıyken (veya `rerank_score` yokken) BM25 eşik değerini belirler (varsayılan: `7.0`).
- `--max-turns <N>`: Sohbet hafızasında tutulacak maksimum tur sayısını belirler (varsayılan: `5`, en az `1`).
- `--agent`: Çok adımlı Agentic RAG modunu aktif eder; soruyu alt sorgularla araştırıp Röntgen çıktısında `Ajan Adımları:` bölümünü gösterir.
- `--max-agent-steps <N>`: Agentic RAG modunda atılacak maksimum planlama/arama adım sayısını belirler (varsayılan: `3`, en az `1`).
- `/history`: Hafızadaki sohbet geçmişini tur bazında özetleyerek listeler.
- `/clear` veya `/reset`: Aktif oturumdaki sohbet hafızasını sıfırlar.
- `q`, `exit` veya `quit`: Etkileşimli sohbetten çıkış yapar.

5. **FastAPI REST API Sunucusunu Başlatma ve Kullanım:**

```bash
.venv/bin/uvicorn api:app --app-dir src --reload
# veya aktif sanal ortam içinde:
uvicorn api:app --app-dir src --reload
```

Sunucu ayağa kalktıktan sonra etkileşimli Swagger/OpenAPI arayüzüne **`http://127.0.0.1:8000/docs`** adresinden erişebilirsiniz.

| Metot & Yol | Açıklama | Temel Parametre / Yanıt Özeti |
| :--- | :--- | :--- |
| `GET /health` | Servis sağlık ve konfigürasyon kontrolü | Qdrant durumu (`ok`/`degraded`), `points_count`, model adları, `llm_configured` |
| `POST /ask` | Çok turlu RAG soru-cevap + izleme | `question`, `session_id`, `top_k` (`1..20`, varsayılan `3`), `use_rerank`, `use_rewrite`, `use_guardrail`, `use_agent` (varsayılan `false`), `max_agent_steps` (`1..10`, varsayılan `3`), eşik skorları; yanıtta `agent_steps` listesi |
| `POST /ask/stream` | Canlı akışlı (SSE) RAG soru-cevap | `AskRequest` şemasıyla `text/event-stream` yanıtı; `meta` (`session_id`, `rewritten_query`, `early_rejected`, `chunks`, `agent_steps`, `timings`) → `token` (`content`) → `done` (`answer`, `timings`) / `error` (`detail`) olayları |
| `POST /search` | LLM çağırmadan hibrit arama (+ opsiyonel rerank) | `query`, `top_k` (`1..20`, varsayılan `3`), `use_rerank`, `rrf_k` (varsayılan `10`), `candidate_k` (`1..50`, varsayılan `6`) |
| `GET /sessions/{session_id}` | Oturum sohbet geçmişini getirme | `session_id`, `turn_count`, `max_turns`, `messages` (yoksa `404`) |
| `DELETE /sessions/{session_id}` | Oturum sohbet geçmişini silme | `session_id`, `deleted: true` (yoksa `404`) |
| `GET /documents` | Yüklü dokümanları ve Qdrant parça sayılarını listeleme | `documents` (`filename`, `size_bytes`, `extension`, `chunk_count`), `total_documents`, `total_chunks` (`503` hata) |
| `POST /documents` | Yeni doküman yükleme ve canlı indeksleme (`201`) | `multipart/form-data` (`file`: `.md`/`.txt`/`.pdf`, maks. 10 MB), `?overwrite=true` opsiyonel; yanıtta `filename`, `size_bytes`, `chunk_count`, `overwritten`, `message` · Hatalar: `400`, `409`, `413`, `503` |
| `DELETE /documents/{filename}` | Dokümanı Qdrant indeksinden ve diskten silme | `filename`, `deleted_chunks`, `deleted: true` · Hatalar: `400`, `404`, `503` |

Örnek `curl` istekleri:

```bash
# 1) Sağlık kontrolü ve çok turlu soru-cevap (/ask)
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Wildcard DNS nedir?", "session_id": "demo-1", "use_rerank": true}'

# 2) Canlı akışlı (SSE) soru-cevap (/ask/stream)
curl -N -s -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Wildcard DNS nedir?", "session_id": "demo-stream", "use_rerank": true}'

# 3) Agentic RAG ile çok adımlı soru-cevap (/ask)
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "SSH parola girişi nasıl kapatılır ve SSH hata logları nasıl incelenir?", "session_id": "agent-1", "use_rerank": true, "use_agent": true, "max_agent_steps": 3}'

# 4) LLM çağırmadan hibrit arama (/search)
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Wildcard DNS", "top_k": 3, "use_rerank": true}'

# 5) Doküman listeleme, yükleme ve silme (/documents)
curl -s http://127.0.0.1:8000/documents
curl -s -X POST http://127.0.0.1:8000/documents \
  -F "file=@data/ornek.md"
curl -s -X POST "http://127.0.0.1:8000/documents?overwrite=true" \
  -F "file=@data/ornek.md"
curl -s -X DELETE http://127.0.0.1:8000/documents/ornek.md
```

6. **Streamlit Görsel Web Arayüzünü Başlatma:**

Görsel arayüz FastAPI sunucusuna bağlanarak çalışır; bu nedenle iki ayrı terminalde aşağıdaki komutları başlatın:

```bash
# 1. Terminal — FastAPI REST API sunucusunu başlatın
uvicorn api:app --app-dir src --reload

# 2. Terminal — Streamlit web arayüzünü başlatın
streamlit run src/ui.py
```

Arayüze tarayıcınızdan **`http://localhost:8501`** adresinden erişebilir, sol kenar çubuğundan canlı `/health` durumunu, **📚 Bilgi Bankası (Dokümanlar)** bölümünü (sürükle-bırak `.md`/`.txt`/`.pdf` doküman yükleme, `Aynı isimde varsa üzerine yaz` seçeneğiyle `📤 Yükle ve İndeksle`, yüklü doküman/parça özeti, `🔄 Yenile` ve `🗑️` ile tek tıkla silme), çalışma modunu, **🤖 Agentic RAG (Çok Adımlı Ajan)** anahtarını (`Maksimum Ajan Adımı` `1..5`), **⚡ Canlı Akış (Streaming)** anahtarını (`POST /ask/stream` ile kelime/token bazlı canlı yanıt akışı; kapatılırsa `/ask` kullanılır) ve Guardrail eşiklerini yönetebilirsiniz. Ajan modu açıkken her yanıtın altındaki **🔬 Röntgen (Observability)** panelinde **🤖 Ajan Karar Adımları** (`search`, `answer`, `direct_chat`, `fallback` aksiyonları, düşünce gerekçesi, alt sorgu, bulunan parça sayısı ve adım süresi) görüntülenir.

7. **Birim Testlerini Çalıştırma (110 Test — `tests/test_eval.py` dahil):**

```bash
.venv/bin/python -m unittest discover -s tests -v
# veya aktif sanal ortam içinde:
python3 -m unittest discover -s tests -v
```
