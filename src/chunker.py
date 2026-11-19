import re
from pathlib import Path


def chunk_markdown(file_path: Path, max_chars: int = 800) -> list[dict]:
    # 1. Dosyayı oku
    text = file_path.read_text(encoding="utf-8")

    # 2. Metni #, ## veya ### başlıklarının başladığı yerlerden böl
    sections = re.split(r"(?m)^(?=#{1,3}\s+)", text)

    chunks = []
    chunk_index = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # İlk satırı başlık (heading), kalanını gövde (body) olarak ayır
        lines = section.splitlines()
        heading = lines[0].lstrip("#").strip() if lines[0].startswith("#") else "Genel"
        body = "\n".join(lines[1:]).strip() if lines[0].startswith("#") else section

        if not body:
            continue

        # Gövde 800 karakterden uzunsa paragraflara bölerek sığdır
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current_text = ""

        for para in paragraphs:
            if len(current_text) + len(para) + 2 <= max_chars:
                current_text = f"{current_text}\n\n{para}".strip()
            else:
                if current_text:
                    chunks.append({
                        "chunk_id": f"{file_path.stem}_{chunk_index}",
                        "text": f"{heading}\n{current_text}",
                        "metadata": {
                            "source": file_path.name,
                            "section": heading,
                        },
                    })
                    chunk_index += 1
                current_text = para

        if current_text:
            chunks.append({
                "chunk_id": f"{file_path.stem}_{chunk_index}",
                "text": f"{heading}\n{current_text}",
                "metadata": {
                    "source": file_path.name,
                    "section": heading,
                },
            })
            chunk_index += 1

    return chunks


def load_and_chunk_folder(data_dir: str = "data") -> list[dict]:
    # data/ klasöründeki tüm .md dosyalarını gez ve parçala
    all_chunks = []
    for md_file in sorted(Path(data_dir).glob("*.md")):
        all_chunks.extend(chunk_markdown(md_file))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_folder("data")
    print(f"Toplam {len(chunks)} chunk üretildi.\n")
    for c in chunks:
        print(f"ID: {c['chunk_id']} | Kaynak: {c['metadata']['source']} | Bölüm: {c['metadata']['section']}")
        print(f"Önizleme: {c['text'][:100]}...\n" + "-" * 50)
