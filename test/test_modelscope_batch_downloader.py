import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import modelscope_batch_downloader as downloader


def record(owner: str, name: str) -> dict[str, object]:
    return {"Owner": owner, "Name": name, "StorageSize": 1}


class DatasetSelectionTests(unittest.TestCase):
    def test_dotenv_loads_token_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "MODELSCOPE_TOKEN=file-token\nEXTRA_SETTING='value'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MODELSCOPE_TOKEN": "process-token"}, clear=True):
                self.assertTrue(downloader.load_dotenv(path))
                self.assertEqual(os.environ["MODELSCOPE_TOKEN"], "process-token")
                self.assertEqual(os.environ["EXTRA_SETTING"], "value")

    def test_default_selection_excludes_giant_repositories(self):
        giant = record("Kpillow", "SceneVTG-Erase")
        regular = record("owner", "regular")
        with (
            patch.object(
                downloader,
                "load_or_collect",
                return_value={"datasets": [giant, regular]},
            ),
            patch.object(downloader, "classify", return_value="OCR"),
            patch.object(downloader, "should_download", return_value=True),
        ):
            targets, excluded, _ = downloader.selected_records()

        self.assertEqual([downloader.dataset_key(item) for item in targets], ["owner/regular"])
        self.assertEqual(
            [downloader.dataset_key(item) for item in excluded],
            ["Kpillow/SceneVTG-Erase"],
        )

    def test_explicit_selection_includes_giant_repository(self):
        giant = record("Kpillow", "SceneVTG-Erase")
        regular = record("owner", "regular")
        with (
            patch.object(
                downloader,
                "load_or_collect",
                return_value={"datasets": [giant, regular]},
            ),
            patch.object(downloader, "classify", return_value="OCR"),
            patch.object(downloader, "should_download", return_value=True),
        ):
            targets, excluded, _ = downloader.selected_records(
                ["Kpillow/SceneVTG-Erase"]
            )

        self.assertEqual(
            [downloader.dataset_key(item) for item in targets],
            ["Kpillow/SceneVTG-Erase"],
        )
        self.assertEqual(excluded, [])

    def test_explicit_selection_rejects_unselected_repository(self):
        with (
            patch.object(
                downloader,
                "load_or_collect",
                return_value={"datasets": [record("owner", "regular")]},
            ),
            patch.object(downloader, "classify", return_value="other"),
            patch.object(downloader, "should_download", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "owner/regular"):
                downloader.selected_records(["owner/regular"])

    def test_dataset_key_requires_owner_and_name(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            downloader.parse_dataset_key("SceneVTG-Erase")


if __name__ == "__main__":
    unittest.main()
