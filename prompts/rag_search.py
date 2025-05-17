# rag_pipeline.py → 최종 버전: main.py에서 호출 시 템플릿에 예시 5개 자동 삽입

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from Levenshtein import distance as levenshtein_distance
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ExperimentConfig

# === Step 1: 환경설정 ===
load_dotenv()

# === Step 2: 데이터 로딩 ===
config = ExperimentConfig(template_name="rag_format")
df = pd.read_csv(os.path.join(config.data_dir, "train.csv"))  # 제공된 공식 데이터만 사용

# === Step 3: 다양한 유사도 기반 예시 선택 알고리즘 ===
def levenshtein_retrieve(query: str, k: int = 10, exclude: str = None) -> List[Tuple[str, str]]:
    filtered_df = df[df["err_sentence"] != exclude] if exclude else df
    filtered_df = filtered_df.copy()
    filtered_df["score"] = filtered_df["err_sentence"].apply(lambda x: levenshtein_distance(query, x))
    top_k = filtered_df.nsmallest(k, "score")
    return list(zip(top_k["err_sentence"], top_k["cor_sentence"]))

def jaccard_retrieve(query: str, k: int = 10, exclude: str = None) -> List[Tuple[str, str]]:
    def jaccard(a, b):
        a_set, b_set = set(a.split()), set(b.split())
        return len(a_set & b_set) / len(a_set | b_set) if len(a_set | b_set) > 0 else 0

    filtered_df = df[df["err_sentence"] != exclude] if exclude else df
    filtered_df = filtered_df.copy()
    filtered_df["score"] = filtered_df["err_sentence"].apply(lambda x: -jaccard(query, x))
    top_k = filtered_df.nsmallest(k, "score")
    return list(zip(top_k["err_sentence"], top_k["cor_sentence"]))

def ngram_overlap_retrieve(query: str, k: int = 10, n: int = 2, exclude: str = None) -> List[Tuple[str, str]]:
    def ngram_overlap(a, b):
        a_ngrams = set([" ".join(a.split()[i:i+n]) for i in range(len(a.split())-n+1)])
        b_ngrams = set([" ".join(b.split()[i:i+n]) for i in range(len(b.split())-n+1)])
        return len(a_ngrams & b_ngrams)

    filtered_df = df[df["err_sentence"] != exclude] if exclude else df
    filtered_df = filtered_df.copy()
    filtered_df["score"] = filtered_df["err_sentence"].apply(lambda x: -ngram_overlap(query, x))
    top_k = filtered_df.nsmallest(k, "score")
    return list(zip(top_k["err_sentence"], top_k["cor_sentence"]))


# === Step 4: 템플릿 전달용 프롬프트 생성 함수 ===
def build_prompt(query: str, template: str, strategy: str = "levenshtein", k: int = 5) -> str:
    if strategy == "levenshtein":
        examples = levenshtein_retrieve(query, k, exclude=query)
    elif strategy == "jaccard":
        examples = jaccard_retrieve(query, k, exclude=query)
    elif strategy == "ngram":
        examples = ngram_overlap_retrieve(query, k, exclude=query)
    else:
        raise ValueError("Unknown retrieval strategy")

    example_block = "\n\n".join([
        f"wrong sentence: {err}\ncorrect sentence: {cor}" for err, cor in examples
    ])
    
    prompt = template.format(examples=example_block.strip(), text=query)

    # ✅ 프롬프트 디버깅 출력 (1회용)
    #print("\n📤 [build_prompt 디버깅용 프롬프트 출력] =========================")
    #print(prompt)
    #print("=================================================================\n")

    return prompt

# === Step 5: 외부에서 import 하여 사용하게 함 ===
__all__ = ["build_prompt"]