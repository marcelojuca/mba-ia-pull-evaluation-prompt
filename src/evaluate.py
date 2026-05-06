"""
Evaluate bug_to_user_story_v2 prompt against the dataset using LangSmith.

Usage:
    python src/evaluate.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI

from src.metrics import ALL_EVALUATORS, get_cache_stats, reset_cache_stats, record_usage
from src.utils import print_results

DATASET_NAME = "bug_to_user_story_dataset"
PROMPT_NAME = "marcelojuca/bug_to_user_story_v2"
DATASET_FILE = Path(__file__).parent.parent / "datasets" / "bug_to_user_story.jsonl"

client = Client()
_cached_chain = None


def _get_chain():
    global _cached_chain
    if _cached_chain is None:
        prompt = client.pull_prompt(PROMPT_NAME)
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0.0)
        _cached_chain = prompt | llm
    return _cached_chain


def ensure_dataset_exists():
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' já existe ({dataset.id})")
        return dataset
    except Exception:
        pass

    print(f"Criando dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="15 bug reports para avaliação de prompts de transformação em user stories",
    )

    examples = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                examples.append(item)

    client.create_examples(
        inputs=[e["input"] for e in examples],
        outputs=[e["output"] for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Dataset criado com {len(examples)} exemplos.")
    return dataset


def run_prompt(inputs: dict) -> dict:
    chain = _get_chain()
    result = chain.invoke(inputs)
    usage = result.usage_metadata or {}
    record_usage(
        input_tokens=usage.get("input_tokens", 0),
        cached_tokens=usage.get("input_token_details", {}).get("cache_read", 0),
    )
    return {"output": result.content}


def main():
    print("Executando avaliação dos prompts...")
    reset_cache_stats()

    ensure_dataset_exists()

    results = evaluate(
        run_prompt,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=f"{PROMPT_NAME}_eval",
        max_concurrency=2,
    )

    # Aggregate scores per metric
    scores: dict[str, list[float]] = {}
    for row in results:
        for eval_result in row.get("evaluation_results", {}).get("results", []):
            key = eval_result.key
            score = eval_result.score
            if score is not None:
                scores.setdefault(key, []).append(float(score))

    metrics = {k: sum(v) / len(v) for k, v in scores.items() if v}

    ordered = {}
    for key in ["helpfulness", "correctness", "f1_score", "clarity", "precision"]:
        if key in metrics:
            ordered[key] = metrics[key]

    print_results(PROMPT_NAME, ordered)

    cache = get_cache_stats()
    print("Cache Stats (OpenAI automatic prefix caching):")
    print(f"  Total LLM calls : {cache['calls']} (15 generation + 90 evaluation)")
    print(f"  Input tokens    : {cache['input_tokens']:,}")
    print(f"  Cached tokens   : {cache['cached_tokens']:,}")
    print(f"  Cache hit rate  : {cache['cache_hit_rate']:.1%}")
    if cache["cached_tokens"] == 0:
        print("  ℹ️  No cache hits detected. Prompts may be below the 1024-token threshold")
        print("     or the cache window expired between runs.")


if __name__ == "__main__":
    main()
