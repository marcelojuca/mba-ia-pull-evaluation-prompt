"""
5 evaluation metrics for bug_to_user_story prompts.

Each metric is a LangSmith evaluator: (run, example) -> EvaluationResult dict.
Scores are normalized to [0.0, 1.0].
"""
import re
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_eval_llm = None
_cache_stats = {"calls": 0, "input_tokens": 0, "cached_tokens": 0}


def _get_eval_llm():
    global _eval_llm
    if _eval_llm is None:
        import os
        model = os.getenv("EVAL_LLM_MODEL", "gpt-4o")
        _eval_llm = ChatOpenAI(model=model, temperature=0.0)
    return _eval_llm


def get_cache_stats() -> dict:
    stats = dict(_cache_stats)
    stats["cache_hit_rate"] = (
        stats["cached_tokens"] / stats["input_tokens"]
        if stats["input_tokens"] > 0 else 0.0
    )
    return stats


def reset_cache_stats():
    _cache_stats.update({"calls": 0, "input_tokens": 0, "cached_tokens": 0})


def _llm_score(prompt_text: str, system: str) -> float:
    llm = _get_eval_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", prompt_text),
    ])
    response = (prompt | llm).invoke({})

    usage = response.usage_metadata or {}
    _cache_stats["calls"] += 1
    _cache_stats["input_tokens"] += usage.get("input_tokens", 0)
    _cache_stats["cached_tokens"] += (
        usage.get("input_token_details", {}).get("cache_read", 0)
    )

    text = response.content.strip()
    match = re.search(r'\b([0-9]|10)(\.[0-9]+)?\b', text)
    if match:
        score = float(match.group())
        return min(score / 10.0, 1.0)
    return 0.5


def helpfulness_evaluator(run: Run, example: Example) -> dict:
    """Is this user story helpful for a developer to understand and fix the bug?"""
    output = run.outputs.get("output", "") if run.outputs else ""
    bug_report = example.inputs.get("bug_report", "")
    system = (
        "You are an expert software engineer evaluating user stories. "
        "Rate from 0 to 10 how helpful the following user story is for a developer "
        "to understand the bug and implement a fix. "
        "10 = extremely helpful (clear persona, acceptance criteria, technical context). "
        "0 = not helpful at all. "
        "Respond with ONLY a single number."
    )
    prompt = f"Bug Report:\n{bug_report}\n\nUser Story:\n{output}"
    score = _llm_score(prompt, system)
    return {"key": "helpfulness", "score": score}


def correctness_evaluator(run: Run, example: Example) -> dict:
    """Does the user story correctly capture the essence of the bug?"""
    output = run.outputs.get("output", "") if run.outputs else ""
    bug_report = example.inputs.get("bug_report", "")
    reference = ""
    if example.outputs:
        reference = example.outputs.get("user_story", "")

    system = (
        "You are an expert evaluating user stories against bug reports. "
        "Rate from 0 to 10 how correctly the generated user story captures "
        "the core issue described in the bug report. "
        "Consider: accuracy of problem description, correct identification of affected persona, "
        "and whether acceptance criteria address the actual bug. "
        "10 = perfectly correct. 0 = completely wrong or unrelated. "
        "Respond with ONLY a single number."
    )
    ref_section = f"\n\nReference User Story:\n{reference}" if reference else ""
    prompt = f"Bug Report:\n{bug_report}\n\nGenerated User Story:\n{output}{ref_section}"
    score = _llm_score(prompt, system)
    return {"key": "correctness", "score": score}


def f1_score_evaluator(run: Run, example: Example) -> dict:
    """Semantic F1: measures completeness and precision of the user story vs bug report.

    Uses LLM-as-judge to evaluate both recall (nothing important missing from the bug)
    and precision (no hallucinations beyond the bug report), then computes F1.
    """
    output = run.outputs.get("output", "") if run.outputs else ""
    bug_report = example.inputs.get("bug_report", "")

    # Recall: does the user story cover all important aspects of the bug?
    recall_system = (
        "You are an expert evaluator. Rate from 0 to 10 the RECALL of this user story: "
        "how well does it cover ALL the important information from the bug report? "
        "Consider: problem description, affected user, expected vs actual behavior, context. "
        "10 = captures everything important. 0 = misses most key information. "
        "Respond with ONLY a single number."
    )
    recall_score = _llm_score(
        f"Bug Report:\n{bug_report}\n\nUser Story:\n{output}",
        recall_system
    )

    # Precision: does the user story avoid adding information not in the bug report?
    precision_system = (
        "You are an expert evaluator. Rate from 0 to 10 the PRECISION of this user story: "
        "how well does it stay focused on the actual bug without adding irrelevant or "
        "incorrect information not supported by the bug report? "
        "10 = perfectly precise, no hallucinations. 0 = full of invented details. "
        "Respond with ONLY a single number."
    )
    precision_score = _llm_score(
        f"Bug Report:\n{bug_report}\n\nUser Story:\n{output}",
        precision_system
    )

    if recall_score + precision_score == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall_score * precision_score / (recall_score + precision_score)

    return {"key": "f1_score", "score": f1}


def clarity_evaluator(run: Run, example: Example) -> dict:
    """Is the user story clear, specific, and unambiguous?"""
    output = run.outputs.get("output", "") if run.outputs else ""
    system = (
        "You are an expert agile coach evaluating user stories. "
        "Rate from 0 to 10 how clear, specific, and unambiguous the following user story is. "
        "Consider: Is the language precise? Are acceptance criteria testable and unambiguous? "
        "Is the format standard and easy to understand? "
        "10 = crystal clear with no ambiguity. 0 = vague and confusing. "
        "Respond with ONLY a single number."
    )
    score = _llm_score(f"User Story:\n{output}", system)
    return {"key": "clarity", "score": score}


def precision_evaluator(run: Run, example: Example) -> dict:
    """Is the user story precise and actionable with specific acceptance criteria?"""
    output = run.outputs.get("output", "") if run.outputs else ""
    bug_report = example.inputs.get("bug_report", "")
    system = (
        "You are an expert product manager evaluating user stories. "
        "Rate from 0 to 10 how precise and actionable the user story is. "
        "Consider: Does it have specific, measurable acceptance criteria? "
        "Does it avoid vague language ('it should work', 'fix the bug')? "
        "Does it include enough technical context for implementation? "
        "10 = extremely precise and immediately actionable. 0 = vague and not actionable. "
        "Respond with ONLY a single number."
    )
    prompt = f"Bug Report:\n{bug_report}\n\nUser Story:\n{output}"
    score = _llm_score(prompt, system)
    return {"key": "precision", "score": score}


ALL_EVALUATORS = [
    helpfulness_evaluator,
    correctness_evaluator,
    f1_score_evaluator,
    clarity_evaluator,
    precision_evaluator,
]
