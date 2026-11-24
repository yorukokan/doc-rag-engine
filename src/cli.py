import argparse
import time
from retriever import search
from generator import ask_rag_with_trace


def print_xray(chunks: list[dict], retrieval_ms: float, generation_ms: float | None = None):
    print("\n--- RÖNTGEN (OBSERVABILITY) ---")
    if generation_ms is not None:
        print(f"Süre -> Arama: {retrieval_ms} ms | LLM Üretim: {generation_ms} ms")
    else:
        print(f"Süre -> Arama: {retrieval_ms} ms")

    for i, c in enumerate(chunks, start=1):
        rerank_info = ""
        if "rerank_score" in c:
            rerank_info = f"Rerank: {c['rerank_score']:.4f} (RRF #{c['rrf_rank']}'den geldi) | "
        print(
            f"  {i}. [{c['source']} - {c['section']}]\n"
            f"     {rerank_info}RRF: {c['score']:.4f} | "
            f"Vektör: #{c['dense_rank']} (cosine={c['dense_score']:.4f}) | "
            f"BM25: #{c['bm25_rank']} (skor={c['bm25_score']:.4f})"
        )
    print("-" * 50)


def handle_question(question: str, top_k: int, debug: bool, search_only: bool, use_rerank: bool):
    if search_only:
        t0 = time.perf_counter()
        chunks = search(question, top_k=top_k, use_rerank=use_rerank)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        print_xray(chunks, retrieval_ms)
        return

    trace = ask_rag_with_trace(question, top_k=top_k, use_rerank=use_rerank)
    if debug:
        print_xray(trace["chunks"], trace["retrieval_ms"], trace["generation_ms"])

    print(f"\nCEVAP:\n{trace['answer']}\n")


def main():
    parser = argparse.ArgumentParser(description="Doc-RAG Etkileşimli Sohbet & Röntgen Aracı")
    parser.add_argument("--debug", action="store_true", help="Vektör, BM25, RRF ve süre detaylarını göster")
    parser.add_argument("--search-only", action="store_true", help="LLM çağırmadan sadece arama röntgenini göster")
    parser.add_argument("--rerank", action="store_true", help="2. aşama Cross-Encoder Reranker'ı aktif et")
    parser.add_argument("--top-k", type=int, default=3, help="Getirilecek bilgi kartı sayısı (varsayılan: 3)")
    parser.add_argument("--query", type=str, help="Tek seferlik soru sor ve çık")
    args = parser.parse_args()

    if args.query:
        handle_question(args.query, args.top_k, args.debug, args.search_only, args.rerank)
        return

    print("Doc-RAG Sohbet Başladı! (Çıkmak için 'q' veya 'exit' yaz)")
    print(
        f"Ayarlar -> top_k={args.top_k} | debug={args.debug} | "
        f"search_only={args.search_only} | rerank={args.rerank}\n"
    )

    while True:
        try:
            question = input("Soru > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz!")
            break

        if not question:
            continue
        if question.lower() in {"q", "exit", "quit"}:
            print("Görüşürüz!")
            break

        handle_question(question, args.top_k, args.debug, args.search_only, args.rerank)


if __name__ == "__main__":
    main()
