import tempfile
import unittest
from pathlib import Path

from tools.audit_dataset import audit_directory, parse_labels


class DatasetAuditTests(unittest.TestCase):
    def test_label_parsing_and_normalization(self):
        self.assertEqual(parse_labels("['g06f/1', 'H04L']"), ["G06F", "H04L"])
        self.assertEqual(parse_labels("A01B,H04L"), ["A01B", "H04L"])
        self.assertEqual(parse_labels(""), [])

    def test_reports_duplicate_rows_and_cross_split_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            contents = {
                "train.tsv": "patent_id\ttitle\tabstract\tlabels\n1\tTool\tA device\t['A01B']\n1\tTool\tA device\t['A01B']\n",
                "dev.tsv": "patent_id\ttitle\tabstract\tlabels\n2\tTool\tA device\t['H04L']\n",
                "test.tsv": "patent_id\ttitle\tabstract\tlabels\n3\tOther\tA network\t['G06F']\n",
            }
            for name, content in contents.items():
                (root / name).write_text(content, encoding="utf-8")

            report = audit_directory(root)

        self.assertEqual(report["splits"]["train"]["rows"], 2)
        self.assertEqual(report["splits"]["train"]["duplicate_id_count"], 1)
        self.assertEqual(report["splits"]["train"]["duplicate_normalized_text_count"], 1)
        self.assertEqual(report["cross_split_overlaps"]["train__dev"]["exact_normalized_text_overlap"], 1)
        self.assertEqual(report["cross_split_overlaps"]["train__dev"]["id_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
