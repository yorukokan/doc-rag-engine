import argparse
import time
from retriever import search
from generator import ask_rag_with_trace
from memory import ConversationMemory


def print_xray(
    chunks: list[dict],
    retrieval_ms: float,
    generation_ms: float | None = None,
    rewrite_ms: float | None = None,
    original_query: str | None = None,
    rewritten_query: str | None = None,
):
    print("\n--- RÖNTGEN (OBSERVABILITY) ---")
    if (
        rewritten_query
        and original_query is not None
        and rewritten_query.strip() != original_query.strip()
    ):
        print(f"Yeniden Yazılan Sorgu: {rewritten_query}")

    if generation_ms is not None:
        if rewrite_ms is not None:
            print(
                f"Süre -> Rewrite: {rewrite_ms} ms | Arama: {retrieval_ms} ms | "
                f"LLM Üretim: {generation_ms} ms"
            )
        else:
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


def _summarize_text(text: str, limit: int = 80) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def print_history(memory: ConversationMemory):
    if memory.is_empty():
        print("\nHafıza boş (0 tur).\n")
        return

    messages = memory.get_messages()
    print(f"\nHafızadaki sohbet geçmişi ({memory.turn_count}/{memory.max_turns} tur):")
    for idx in range(0, len(messages), 2):
        turn_no = idx // 2 + 1
        user_msg = _summarize_text(messages[idx]["content"])
        assistant_msg = _summarize_text(messages[idx + 1]["content"])
        print(f"  Tur {turn_no} | Soru : {user_msg}")
        print(f"        | Cevap: {assistant_msg}")
    print()


def handle_question(
    question: str,
    top_k: int,
    debug: bool,
    search_only: bool,
    use_rerank: bool,
    memory: ConversationMemory | None = None,
    use_rewrite: bool = True,
):
    if search_only:
        t0 = time.perf_counter()
        chunks = search(question, top_k=top_k, use_rerank=use_rerank)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        print_xray(chunks, retrieval_ms)
        return

    history = memory.get_messages() if memory is not None else None
    trace = ask_rag_with_trace(
        question,
        top_k=top_k,
        use_rerank=use_rerank,
        history=history,
        use_rewrite=use_rewrite,
    )
    if debug:
        print_xray(
            trace["chunks"],
            trace["retrieval_ms"],
            trace["generation_ms"],
            rewrite_ms=trace["rewrite_ms"],
            original_query=question,
            rewritten_query=trace["rewritten_query"],
        )

    print(f"\nCEVAP:\n{trace['answer']}\n")
    if memory is not None:
        memory.add_turn(question, trace["answer"])


def main():
    parser = argparse.ArgumentParser(description="Doc-RAG Etkileşimli Sohbet & Röntgen Aracı")
    parser.add_argument("--debug", action="store_true", help="Vektör, BM25, RRF ve süre detaylarını göster")
    parser.add_argument("--search-only", action="store_true", help="LLM çağırmadan sadece arama röntgenini göster")
    parser.add_argument("--rerank", action="store_true", help="2. aşama Cross-Encoder Reranker'ı aktif et")
    parser.add_argument("--top-k", type=int, default=3, help="Getirilecek bilgi kartı sayısı (varsayılan: 3)")
    parser.add_argument("--query", type=str, help="Tek seferlik soru sor ve çık")
    parser.add_argument("--no-rewrite", action="store_true", help="Takip sorularında query rewriting'i devre dışı bırak")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Sohbet hafızasında tutulacak maksimum tur sayısı (varsayılan: 5)",
    )
    args = parser.parse_args()

    if args.max_turns < 1:
        parser.error("--max-turns en az 1 olmalıdır.")

    use_rewrite = not args.no_rewrite

    if args.query:
        handle_question(
            args.query,
            args.top_k,
            args.debug,
            args.search_only,
            args.rerank,
            memory=None,
            use_rewrite=use_rewrite,
        )
        return

    memory = ConversationMemory(max_turns=args.max_turns)

    print(
        "Doc-RAG Sohbet Başladı! "
        "(Çıkmak için 'q'/'exit', hafıza için '/history', temizlemek için '/clear' veya '/reset')"
    )
    print(
        f"Ayarlar -> top_k={args.top_k} | debug={args.debug} | "
        f"search_only={args.search_only} | rerank={args.rerank} | "
        f"rewrite={use_rewrite} | max_turns={args.max_turns}\n"
    )

    while True:
        try:
            question = input("Soru > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGörüşürüz!")
            break

        if not question:
            continue
        lowered = question.lower()
        if lowered in {"q", "exit", "quit"}:
            print("Görüşürüz!")
            break
        if lowered in {"/clear", "/reset"}:
            memory.clear()
            print("\nSohbet hafızası temizlendi.\n")
            continue
        if lowered == "/history":
            print_history(memory)
            continue

        handle_question(
            question,
            args.top_k,
            args.debug,
            args.search_only,
            args.rerank,
            memory=memory,
            use_rewrite=use_rewrite,
        )


if __name__ == "__main__":
    main()
