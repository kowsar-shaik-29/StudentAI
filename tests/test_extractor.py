"""
Unit tests for PDF extraction module (src.extractor).
"""

import unittest
import tempfile
from pathlib import Path
import pymupdf
from src.extractor import extract_pdf, extract_pdf_pages


class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a synthetic 2-page PDF (Page 1 has text, Page 2 is blank)
        self.sample_pdf = self.temp_path / "test_notes.pdf"
        doc = pymupdf.open()
        page1 = doc.new_page()
        page1.insert_text((50, 72), "Database Management Systems Lecture 1: Relational Model.")
        # Page 2 is intentionally blank
        _ = doc.new_page()
        doc.save(str(self.sample_pdf))
        doc.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_pdf_pages_count_and_metadata(self):
        pages = extract_pdf(self.sample_pdf, include_empty=True)
        self.assertEqual(len(pages), 2)

        # Page 1 checks
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[0]["total_pages"], 2)
        self.assertEqual(pages[0]["file_name"], "test_notes.pdf")
        self.assertTrue(pages[0]["has_text"])
        self.assertIn("Relational Model", pages[0]["text"])

        # Page 2 checks (blank page)
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertEqual(pages[1]["total_pages"], 2)
        self.assertFalse(pages[1]["has_text"])
        self.assertEqual(pages[1]["text"], "")

    def test_extract_pdf_without_empty_pages(self):
        pages = extract_pdf_pages(self.sample_pdf)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_number"], 1)

    def test_nonexistent_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            extract_pdf(self.temp_path / "nonexistent.pdf")

    def test_directory_path_raises_isadirectoryerror(self):
        with self.assertRaises(IsADirectoryError):
            extract_pdf(self.temp_path)

    def test_zero_byte_file_raises_valueerror(self):
        empty_file = self.temp_path / "empty.pdf"
        empty_file.write_bytes(b"")
        with self.assertRaises(ValueError):
            extract_pdf(empty_file)


if __name__ == "__main__":
    unittest.main()
