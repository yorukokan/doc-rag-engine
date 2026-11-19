from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from chunker import load_and_chunk_folder

COLLECTION_NAME = "dokumanlar"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def index_documents():
    # 1. Önceki adımda yazdığımız chunker ile bilgi kartlarını al
    chunks = load_and_chunk_folder("data")
    if not chunks:
        print("data/ klasöründe hiç doküman bulunamadı!")
        return

    print(f"{len(chunks)} adet chunk bulundu. Vektöre çevriliyor...")

    # 2. Türkçe destekli hafif yerel embedding modelini yükle
    # (İlk çalışmada modeli indirir, sonraki çalışmalarda direkt bilgisayardan okur)
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    vectors = [v.tolist() for v in embedder.embed(texts)]

    # 3. Qdrant'ı Docker olmadan yerel klasör (qdrant_data/) olarak başlat
    client = QdrantClient(path="qdrant_data")

    # Aynı veriyi üst üste yazmamak için tablo varsa sıfırla, yeniden oluştur
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),            # Her vektörün sayı uzunluğu (384)
            distance=models.Distance.COSINE, # Benzerliği açısal yakınlıkla ölç
        ),
    )

    # 4. Vektörleri ve etiketleri (metadata) Qdrant'a kaydet
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
        points.append(
            models.PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "section": chunk["metadata"]["section"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Başarılı! {len(points)} kayıt Qdrant veritabanına (qdrant_data/) yazıldı.")


if __name__ == "__main__":
    index_documents()
