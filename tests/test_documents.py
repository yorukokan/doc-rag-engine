import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import api
import chunker
import indexer
import retriever
from ui_client import (
    ApiClientError,
    delete_document as client_delete_document,
    list_documents as client_list_documents,
    upload_document as client_upload_document,
)


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeEmbedder:
    """Testler için deterministik 4 boyutlu sahte vektör üretici."""

    def embed(self, texts: list[str]):
        for idx, text in enumerate(texts):
            length = float(len(text) or 1)
            yield _FakeVector([0.1 * (idx + 1), 0.2, 0.3, min(1.0, length / 100.0)])


def _make_mock_http_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestValidateFilename(unittest.TestCase):
    def test_valid_filenames_pass_and_strip_whitespace(self):
        self.assertEqual(indexer.validate_filename("  rehber.md  "), "rehber.md")
        self.assertEqual(indexer.validate_filename("notlar.TXT"), "notlar.TXT")
        self.assertEqual(indexer.validate_filename("rapor_2026.pdf"), "rapor_2026.pdf")

    def test_rejects_traversal_hidden_control_null_and_long_names(self):
        invalid_names = [
            "",
            "   ",
            None,
            123,
            "../secret.md",
            "alt/dosya.md",
            "alt\\dosya.md",
            "dosya..md",
            ".gizli.md",
            "test\x00.md",
            "test\n.md",
            "test\t.md",
            "script.py",
            "resim.png",
            "uzantisiz",
            "a" * (indexer.MAX_FILENAME_LENGTH + 1) + ".md",
        ]
        for bad in invalid_names:
            with self.subTest(filename=bad):
                with self.assertRaises(ValueError):
                    indexer.validate_filename(bad)


class TestChunkFile(unittest.TestCase):
    def test_chunk_file_markdown_with_source_name_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file = Path(tmpdir) / "tmp_random_name.bin"
            tmp_file.write_text("# Giriş\nBu bir test paragrafıdır.\n", encoding="utf-8")

            chunks = chunker.chunk_file(str(tmp_file), source_name="ornek.md")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["metadata"]["source"], "ornek.md")
            self.assertEqual(chunks[0]["chunk_id"], "ornek_0")
            self.assertIn("Bu bir test paragrafıdır.", chunks[0]["text"])

    def test_chunk_file_txt_and_invalid_extension_and_bad_utf8(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "not.txt"
            txt_file.write_text("Birinci paragraf.\n\nİkinci paragraf.", encoding="utf-8")
            chunks = chunker.chunk_file(str(txt_file))
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["metadata"]["source"], "not.txt")

            bad_ext = Path(tmpdir) / "dosya.exe"
            bad_ext.write_text("merhaba", encoding="utf-8")
            with self.assertRaises(ValueError):
                chunker.chunk_file(str(bad_ext))

            bad_utf8 = Path(tmpdir) / "bozuk.txt"
            bad_utf8.write_bytes(b"\xff\xfe\xfa\xfb")
            with self.assertRaises(UnicodeDecodeError):
                chunker.chunk_file(str(bad_utf8))


class TestIndexerLifecycle(unittest.TestCase):
    def test_save_list_overwrite_delete_and_full_reindex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = str(Path(tmpdir) / "data")
            qdrant_path = str(Path(tmpdir) / "qdrant")

            # Olmayan klasör -> boş liste
            self.assertEqual(indexer.list_documents(data_dir, qdrant_path), [])

            with patch("indexer.get_embedder", return_value=_FakeEmbedder()):
                md_v1 = (
                    "# Bölüm 1\nİlk bölüm içeriği.\n\n"
                    "# Bölüm 2\nİkinci bölüm içeriği.\n"
                ).encode("utf-8")
                res1 = indexer.save_and_index_document(
                    "kilavuz.md",
                    md_v1,
                    overwrite=False,
                    data_dir=data_dir,
                    qdrant_path=qdrant_path,
                )
                self.assertEqual(res1["filename"], "kilavuz.md")
                self.assertEqual(res1["chunk_count"], 2)
                self.assertFalse(res1["overwritten"])

                docs = indexer.list_documents(data_dir, qdrant_path)
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0]["filename"], "kilavuz.md")
                self.assertEqual(docs[0]["chunk_count"], 2)
                self.assertEqual(docs[0]["extension"], ".md")

                # overwrite=False ile çakışma -> FileExistsError
                with self.assertRaises(FileExistsError):
                    indexer.save_and_index_document(
                        "kilavuz.md",
                        b"# Tek\nYeni metin.",
                        overwrite=False,
                        data_dir=data_dir,
                        qdrant_path=qdrant_path,
                    )

                # overwrite=True -> eski 2 chunk silinip 1 yeni chunk yazılmalı
                md_v2 = b"# Tek\nSadece tek parca icerik.\n"
                res2 = indexer.save_and_index_document(
                    "kilavuz.md",
                    md_v2,
                    overwrite=True,
                    data_dir=data_dir,
                    qdrant_path=qdrant_path,
                )
                self.assertTrue(res2["overwritten"])
                self.assertEqual(res2["chunk_count"], 1)

                docs_after = indexer.list_documents(data_dir, qdrant_path)
                self.assertEqual(docs_after[0]["chunk_count"], 1)

                # Qdrant içinde gerçekten sadece 1 point olduğunu doğrula
                qc = QdrantClient(path=qdrant_path)
                try:
                    pts = retriever.scroll_all_points(qc, retriever.COLLECTION_NAME)
                    self.assertEqual(len(pts), 1)
                    self.assertIn("Sadece tek parca icerik.", pts[0].payload["text"])
                finally:
                    qc.close()

                # Silme testi
                del_res = indexer.delete_document(
                    "kilavuz.md", data_dir=data_dir, qdrant_path=qdrant_path
                )
                self.assertEqual(del_res["deleted_chunks"], 1)
                self.assertTrue(del_res["deleted"])
                self.assertFalse((Path(data_dir) / "kilavuz.md").exists())
                self.assertEqual(indexer.list_documents(data_dir, qdrant_path), [])

                # Olmayan dosyayı silme -> FileNotFoundError
                with self.assertRaises(FileNotFoundError):
                    indexer.delete_document(
                        "yok.md", data_dir=data_dir, qdrant_path=qdrant_path
                    )

    def test_save_and_index_rollback_on_indexing_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = str(Path(tmpdir) / "data")
            qdrant_path = str(Path(tmpdir) / "qdrant")

            # 1. Yeni dosya yüklenirken indeksleme patlarsa dosya diskten geri alınmalı
            with patch(
                "indexer.index_single_file",
                side_effect=RuntimeError("Qdrant yazma hatası"),
            ):
                with self.assertRaises(RuntimeError):
                    indexer.save_and_index_document(
                        "yeni.txt",
                        b"Yeni dokuman metni",
                        overwrite=False,
                        data_dir=data_dir,
                        qdrant_path=qdrant_path,
                    )
            self.assertFalse((Path(data_dir) / "yeni.txt").exists())

            # 2. Var olan dosya overwrite edilirken indeksleme patlarsa eski içerik korunmalı
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            existing = Path(data_dir) / "eski.txt"
            existing.write_bytes(b"Orijinal icerik")

            with patch(
                "indexer.index_single_file",
                side_effect=RuntimeError("Qdrant yazma hatası"),
            ):
                with self.assertRaises(RuntimeError):
                    indexer.save_and_index_document(
                        "eski.txt",
                        b"Bozuk yeni icerik",
                        overwrite=True,
                        data_dir=data_dir,
                        qdrant_path=qdrant_path,
                    )
            self.assertEqual(existing.read_bytes(), b"Orijinal icerik")

    def test_save_rejects_empty_oversize_and_corrupt_utf8(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = str(Path(tmpdir) / "data")
            qdrant_path = str(Path(tmpdir) / "qdrant")

            with self.assertRaises(ValueError):
                indexer.save_and_index_document(
                    "bos.txt", b"", data_dir=data_dir, qdrant_path=qdrant_path
                )

            with self.assertRaises(OverflowError):
                indexer.save_and_index_document(
                    "buyuk.txt",
                    b"a" * (indexer.MAX_UPLOAD_SIZE_BYTES + 1),
                    data_dir=data_dir,
                    qdrant_path=qdrant_path,
                )

            with self.assertRaises(ValueError):
                indexer.save_and_index_document(
                    "bozuk.txt",
                    b"\xff\xfe\xfa",
                    data_dir=data_dir,
                    qdrant_path=qdrant_path,
                )
            self.assertFalse((Path(data_dir) / "bozuk.txt").exists())


class TestRetrieverScrollAndEmptyCollection(unittest.TestCase):
    def test_scroll_all_points_collects_multiple_pages(self):
        fake_client = MagicMock()
        p1 = SimpleNamespace(id="1", payload={"chunk_id": "a_0"})
        p2 = SimpleNamespace(id="2", payload={"chunk_id": "a_1"})
        p3 = SimpleNamespace(id="3", payload={"chunk_id": "a_2"})
        fake_client.scroll.side_effect = [
            ([p1], "cursor_1"),
            ([p2], "cursor_2"),
            ([p3], None),
        ]

        points = retriever.scroll_all_points(fake_client, batch_size=1)
        self.assertEqual(len(points), 3)
        self.assertEqual([p.id for p in points], ["1", "2", "3"])
        self.assertEqual(fake_client.scroll.call_count, 3)

    def test_search_returns_empty_when_collection_missing_and_closes_client(self):
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = False

        with (
            patch("retriever.get_embedder", return_value=_FakeEmbedder()),
            patch("retriever.QdrantClient", return_value=fake_client),
        ):
            results = retriever.search("test sorgusu")

        self.assertEqual(results, [])
        fake_client.close.assert_called_once()
        fake_client.query_points.assert_not_called()

    def test_search_returns_empty_when_collection_has_no_points(self):
        fake_client = MagicMock()
        fake_client.collection_exists.return_value = True
        fake_client.query_points.return_value = SimpleNamespace(points=[])
        fake_client.scroll.return_value = ([], None)

        with (
            patch("retriever.get_embedder", return_value=_FakeEmbedder()),
            patch("retriever.QdrantClient", return_value=fake_client),
        ):
            results = retriever.search("test sorgusu")

        self.assertEqual(results, [])
        fake_client.close.assert_called_once()


class TestDocumentsApi(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = str(Path(self._tmpdir.name) / "data")
        self.qdrant_path = str(Path(self._tmpdir.name) / "qdrant")
        self.client = TestClient(api.app)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _patched_context(self):
        return (
            patch("indexer.get_embedder", return_value=_FakeEmbedder()),
            patch(
                "api.indexer_list_documents",
                side_effect=lambda: indexer.list_documents(
                    data_dir=self.data_dir, qdrant_path=self.qdrant_path
                ),
            ),
            patch(
                "api.save_and_index_document",
                side_effect=lambda filename, content, overwrite=False: indexer.save_and_index_document(
                    filename=filename,
                    content=content,
                    overwrite=overwrite,
                    data_dir=self.data_dir,
                    qdrant_path=self.qdrant_path,
                ),
            ),
            patch(
                "api.indexer_delete_document",
                side_effect=lambda filename: indexer.delete_document(
                    filename=filename,
                    data_dir=self.data_dir,
                    qdrant_path=self.qdrant_path,
                ),
            ),
        )

    def test_full_crud_flow_and_error_codes(self):
        p1, p2, p3, p4 = self._patched_context()
        with p1, p2, p3, p4:
            # 1. Başlangıçta boş liste
            r_list = self.client.get("/documents")
            self.assertEqual(r_list.status_code, 200)
            self.assertEqual(
                r_list.json(),
                {"documents": [], "total_documents": 0, "total_chunks": 0},
            )

            # 2. Yeni yükleme -> 201
            r_up = self.client.post(
                "/documents",
                files={"file": ("notlar.md", b"# Baslik\nIcerik metni.", "text/markdown")},
            )
            self.assertEqual(r_up.status_code, 201)
            up_body = r_up.json()
            self.assertEqual(up_body["filename"], "notlar.md")
            self.assertEqual(up_body["chunk_count"], 1)
            self.assertFalse(up_body["overwritten"])

            # 3. Liste artık 1 doküman ve 1 chunk döndürmeli
            r_list2 = self.client.get("/documents")
            self.assertEqual(r_list2.status_code, 200)
            self.assertEqual(r_list2.json()["total_documents"], 1)
            self.assertEqual(r_list2.json()["total_chunks"], 1)

            # 4. Aynı dosya overwrite=false -> 409 Conflict
            r_conflict = self.client.post(
                "/documents",
                files={"file": ("notlar.md", b"# Baslik\nYeni icerik.", "text/markdown")},
            )
            self.assertEqual(r_conflict.status_code, 409)

            # 5. overwrite=true -> 201
            r_overwrite = self.client.post(
                "/documents?overwrite=true",
                files={
                    "file": (
                        "notlar.md",
                        b"# B1\nParca 1.\n\n# B2\nParca 2.",
                        "text/markdown",
                    )
                },
            )
            self.assertEqual(r_overwrite.status_code, 201)
            self.assertTrue(r_overwrite.json()["overwritten"])
            self.assertEqual(r_overwrite.json()["chunk_count"], 2)

            # 6. Geçersiz uzantı -> 400
            r_ext = self.client.post(
                "/documents",
                files={"file": ("zararli.sh", b"echo hi", "text/plain")},
            )
            self.assertEqual(r_ext.status_code, 400)

            # 7. Path traversal upload -> 400
            r_trav = self.client.post(
                "/documents",
                files={"file": ("../evil.md", b"# X\nY", "text/markdown")},
            )
            self.assertEqual(r_trav.status_code, 400)

            # 8. Boş dosya -> 400
            r_empty = self.client.post(
                "/documents",
                files={"file": ("bos.txt", b"", "text/plain")},
            )
            self.assertEqual(r_empty.status_code, 400)

            # 9. Silme -> 200, tekrar silme -> 404
            r_del = self.client.delete("/documents/notlar.md")
            self.assertEqual(r_del.status_code, 200)
            self.assertEqual(
                r_del.json(),
                {"filename": "notlar.md", "deleted_chunks": 2, "deleted": True},
            )

            r_del_404 = self.client.delete("/documents/notlar.md")
            self.assertEqual(r_del_404.status_code, 404)

            # 10. Path traversal delete -> 400
            r_del_trav = self.client.delete("/documents/..%2Fsecret.md")
            self.assertEqual(r_del_trav.status_code, 400)

    def test_upload_exceeding_size_limit_returns_413(self):
        with patch("api.MAX_UPLOAD_SIZE_BYTES", 32):
            resp = self.client.post(
                "/documents",
                files={"file": ("buyuk.txt", b"x" * 64, "text/plain")},
            )
        self.assertEqual(resp.status_code, 413)

    def test_upload_unexpected_error_returns_503_without_leaking_secret(self):
        with patch(
            "api.save_and_index_document",
            side_effect=RuntimeError("SECRET_INTERNAL_TOKEN_999"),
        ):
            resp = self.client.post(
                "/documents",
                files={"file": ("test.txt", b"Merhaba", "text/plain")},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertNotIn("SECRET_INTERNAL_TOKEN_999", resp.text)


class TestUiClientDocuments(unittest.TestCase):
    def test_list_documents_success_and_connect_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), "http://testserver/documents")
            return httpx.Response(
                200,
                json={"documents": [], "total_documents": 0, "total_chunks": 0},
            )

        with _make_mock_http_client(handler) as mock_client:
            data = client_list_documents(
                base_url="http://testserver/", client=mock_client
            )
        self.assertEqual(data["total_documents"], 0)

        def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        with _make_mock_http_client(fail_handler) as mock_client:
            with self.assertRaises(ApiClientError) as ctx:
                client_list_documents(base_url="http://testserver", client=mock_client)
        self.assertIsNone(ctx.exception.status_code)

    def test_upload_document_success_and_conflict_error(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["body"] = request.read()
            return httpx.Response(
                201,
                json={
                    "filename": "test.md",
                    "size_bytes": 12,
                    "chunk_count": 1,
                    "overwritten": True,
                    "message": "ok",
                },
            )

        with _make_mock_http_client(handler) as mock_client:
            res = client_upload_document(
                "test.md",
                b"# H\nMerhaba!",
                overwrite=True,
                base_url="http://testserver",
                client=mock_client,
            )
        self.assertEqual(captured["method"], "POST")
        self.assertIn("overwrite=true", captured["url"])
        self.assertIn(b"test.md", captured["body"])
        self.assertTrue(res["overwritten"])

        def conflict_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(409, json={"detail": "'test.md' zaten mevcut."})

        with _make_mock_http_client(conflict_handler) as mock_client:
            with self.assertRaises(ApiClientError) as ctx:
                client_upload_document(
                    "test.md",
                    b"# H\nMerhaba!",
                    base_url="http://testserver",
                    client=mock_client,
                )
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("zaten mevcut", str(ctx.exception))

    def test_delete_document_url_encoding_and_404(self):
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            return httpx.Response(
                200,
                json={"filename": "yeni rapor.md", "deleted_chunks": 3, "deleted": True},
            )

        with _make_mock_http_client(handler) as mock_client:
            res = client_delete_document(
                "yeni rapor.md",
                base_url="http://testserver",
                client=mock_client,
            )
        self.assertEqual(captured["method"], "DELETE")
        self.assertTrue(captured["url"].endswith("/documents/yeni%20rapor.md"))
        self.assertEqual(res["deleted_chunks"], 3)

        def not_found_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "'yok.md' bulunamadı."})

        with _make_mock_http_client(not_found_handler) as mock_client:
            with self.assertRaises(ApiClientError) as ctx:
                client_delete_document(
                    "yok.md", base_url="http://testserver", client=mock_client
                )
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
