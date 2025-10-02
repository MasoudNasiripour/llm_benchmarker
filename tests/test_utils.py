import os
import sys
import unittest

from pathlib import Path

sys.path.append("/benchmarker")

from llm_benchmarker.utils import load_from_github, \
    github_url_to_raw_github_url, load_from_hf


class TestDownloadFromUrl(unittest.TestCase):

    def test_download_from_github(self, ):
        url = "https://github.com/sajjjadayobi/PersianQA/blob/main/dataset/pqa_test.json"

        base_path = Path(__file__).parent
        destination = os.path.join(base_path, Path("pqa_test_v1.json"))

        main_content_file = open(os.path.join(base_path, Path("pqa_test.json")),
                                 encoding="utf8")

        outcome = load_from_github(url, destination)

        self.assertTrue(outcome, "Error in downloading.")

        self.assertTrue(os.path.exists(destination), "File didn\'t Found. Maybe have problem in downloading it.")

        downloaded_file = open(destination, encoding="utf8")

        self.assertEqual(main_content_file.read(), downloaded_file.read())

        main_content_file.close()
        downloaded_file.close()

        os.remove(destination)

    def test_github_url_to_raw_github_url(self, ):
        github_urls = ["https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/requirements.txt",
                       "https://github.com/sajjjadayobi/PersianQA/blob/main/dataset/pqa_train.json"]

        raw_github_urls = ["https://raw.githubusercontent.com/TIGER-AI-Lab/MMLU-Pro/refs/heads/main/requirements.txt",
                           "https://raw.githubusercontent.com/sajjjadayobi/PersianQA/refs/heads/main/dataset/pqa_train.json"]

        self.assertListEqual(github_url_to_raw_github_url(github_urls), raw_github_urls)

        with self.assertRaises(Exception, msg="An Exception should be raised because of inputing wrong github url!"):
            github_url_to_raw_github_url("https://gitlab.com/TIGER-AI-Lab/MMLU-Pro/blob/main/requirements.txt")

    def test_huggingface_url(self):
        base_path = Path(__file__).parent
        destination = os.path.join(base_path, Path("mmlu"))
        self.assertTrue(load_from_hf("cais/mmlu", destination, name='all'))
        self.assertTrue(os.path.isdir(destination))
