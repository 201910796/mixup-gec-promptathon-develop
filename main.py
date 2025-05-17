from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from prompts.templates import TEMPLATES
from utils.experiment import ExperimentRunner

def debug_results(df, results, n_samples=10):
    print("\n=== 디버깅: 예측 결과 샘플 ===")
    sample_indices = df.sample(n=min(n_samples, len(df))).index

    for i, idx in enumerate(sample_indices):
        print(f"\n[샘플 {i+1}]")
        print(f"원본: {df.loc[idx, 'err_sentence']}")
        print(f"정답: {df.loc[idx, 'cor_sentence']}")
        print(f"예측: {results['cor_sentence'][idx]}")
        
        is_correct = df.loc[idx, 'cor_sentence'] == results['cor_sentence'][idx]
        print(f"정답 여부: {'✓' if is_correct else '✗'}")
        if not is_correct:
            print(f"차이점: 정답과 예측이 다릅니다")
    
    correct_count = sum(df['cor_sentence'] == results['cor_sentence'])
    accuracy = correct_count / len(df) * 100
    print(f"\n전체 정확도: {accuracy:.2f}% ({correct_count}/{len(df)})")

    print("\n=== 에러 패턴 분석 ===")
    error_indices = df[df['cor_sentence'] != results['cor_sentence']].index
    print(f"총 에러 수: {len(error_indices)}")

    if len(error_indices) > 0:
        print("\n[에러 샘플 상세]")
        for i, idx in enumerate(error_indices[:5]):
            print(f"\n에러 {i+1}:")
            print(f"원본: {df.loc[idx, 'err_sentence']}")
            print(f"정답: {df.loc[idx, 'cor_sentence']}")
            print(f"예측: {results['cor_sentence'][idx]}")
            print("-" * 50)


def split_dataframe(df, n):
    chunk_size = len(df) // n
    return [df.iloc[i * chunk_size: (i + 1) * chunk_size].copy() if i < n - 1 else df.iloc[i * chunk_size:].copy() for i in range(n)]


def run_on_chunk(test_chunk, api_key, template_name):
    config = ExperimentConfig(
        template_name=template_name,
        temperature=0.0,
        batch_size=5,
        experiment_name=f"parallel_submission_{template_name}"
    )
    runner = ExperimentRunner(config, api_key)
    return runner.run(test_chunk)


def main():
    # API 키 로드
    load_dotenv()
    api_keys = [os.getenv(f"UPSTAGE_API_KEY_{i+1}") for i in range(5)]
    if any(k is None for k in api_keys):
        raise ValueError("One or more API keys are missing in environment variables")
    
    base_config = ExperimentConfig(template_name='basic')

    # 데이터 로드
    try:
        train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'), encoding="utf-8")
        test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'), encoding="utf-8")
    except UnicodeDecodeError:
        print("utf-8 실패, cp949로 다시 시도합니다.")
        train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'), encoding="cp949")
        test = pd.read_csv(os.path.join(base_config.data_dir, 'test.csv'), encoding="cp949")

    # 토이 데이터셋 생성
    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)

    # train/valid 분할
    train_data, valid_data = train_test_split(
        toy_data,
        test_size=base_config.test_size,
        random_state=base_config.random_seed
    )

    # 모든 템플릿으로 실험
    results = {}
    detailed_results = {}

    for template_name in TEMPLATES.keys():
        print(f"\n{'='*50}")
        print(f"템플릿 테스트 중: {template_name}")
        print('='*50)

        config = ExperimentConfig(
            template_name=template_name,
            temperature=0.0,
            batch_size=5,
            experiment_name=f"toy_experiment_{template_name}"
        )
        runner = ExperimentRunner(config, api_keys[0])  # train/valid에는 첫 번째 키 사용

        # 실험 실행
        template_results = runner.run_template_experiment(train_data, valid_data)
        results[template_name] = template_results

        # 디버깅용 결과 저장
        valid_predictions = runner.run(valid_data)
        valid_predictions['cor_sentence'].index = valid_data.index
        detailed_results[template_name] = {
            'valid_data': valid_data,
            'predictions': valid_predictions
        }

        print(f"\n--- {template_name} 템플릿 디버깅 ---")
        debug_results(valid_data, valid_predictions, n_samples=5)

    # 최고 성능 템플릿 선택
    best_template = max(results.items(), key=lambda x: x[1]['valid_recall']['recall'])[0]
    print(f"\n최고 성능 템플릿: {best_template}")
    print(f"Valid Recall: {results[best_template]['valid_recall']['recall']:.2f}%")
    print(f"Valid Precision: {results[best_template]['valid_recall']['precision']:.2f}%")

    print(f"\n=== {best_template} 템플릿 상세 분석 ===")
    best_details = detailed_results[best_template]
    debug_results(best_details['valid_data'], best_details['predictions'], n_samples=10)

    # 테스트 데이터 병렬 예측 시작
    print("\n=== 테스트 데이터 병렬 예측 시작 ===")
    test_chunks = split_dataframe(test, 5)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(run_on_chunk, chunk, key, best_template)
            for chunk, key in zip(test_chunks, api_keys)
        ]
        test_results_chunks = [future.result() for future in futures]

    final_results = pd.concat(test_results_chunks, axis=0).sort_index()
    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': final_results['cor_sentence']
    })

    output.to_csv(r"C:\Users\user\python_project\datathon_prompting\code\submission_baseline.csv", index=False)
    print("\n제출 파일이 생성되었습니다: submission_baseline.csv")
    print(f"사용된 템플릿: {best_template}")
    print(f"예측된 샘플 수: {len(output)}")

if __name__ == "__main__":
    main()
