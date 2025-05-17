# ✅ experiment.py (RAG + 템플릿 기반, 후처리 제거 버전)

import os
import pandas as pd
import requests
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv
from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.metrics import evaluate_correction
from prompts.rag_search import build_prompt
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[self.config.template_name]
        self.api_url = config.api_url
        self.model = config.model

    def _make_prompt(self, text: str) -> str:
        return build_prompt(
            query=text,
            template=self.template,
            strategy="levenshtein",
            k=20
        )

    def _call_api_single(self, prompt: str, retries: int = 10) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"⚠️ API 호출 실패 (시도 {attempt+1}/{retries}). {wait}초 대기 후 재시도...")
                    time.sleep(wait)
                else:
                    print(f"❌ 최종 실패: {e}")
                    return "[API_FAILED]"

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        result_dict = {}
        call_count = defaultdict(int)
        call_lock = threading.Lock()

        def process_row(row):
            err = row["err_sentence"]
            row_id = row["id"]

            with call_lock:
                if call_count[err] >= 1:
                    print(f"🚫 이미 호출된 문장입니다. 건너뜀: {err}")
                    return row_id, "[DUPLICATE_SKIPPED]"

                call_count[err] += 1

            prompt = self._make_prompt(err)
            try:
                corrected = self._call_api_single(prompt)
            except Exception as e:
                print(f"\n⚠️ API 호출 실패: {e}")
                corrected = "[error]"

            return row_id, corrected

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_row = {
                executor.submit(process_row, row): row for _, row in data.iterrows()
            }
            for future in tqdm(as_completed(future_to_row), total=len(future_to_row)):
                row_id, corrected = future.result()
                result_dict[row_id] = corrected

        output_df = pd.DataFrame({
            "id": data["id"],
            "cor_sentence": data["id"].map(result_dict)
        })
        return output_df

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")

        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data)
        train_recall = evaluate_correction(train_data, train_results)

        print("\n[검증 데이터 실험]")
        valid_results = self.run(valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)

        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        }
