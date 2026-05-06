import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent


def load_prompt_from_yaml(yaml_path: str | Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yaml_to_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    messages = []
    for msg in prompt_data["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            messages.append(SystemMessagePromptTemplate.from_template(content))
        elif role == "human":
            messages.append(HumanMessagePromptTemplate.from_template(content))
    return ChatPromptTemplate.from_messages(messages)


def get_llm(model: str | None = None, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


def get_eval_llm():
    from langchain_openai import ChatOpenAI
    model = os.getenv("EVAL_LLM_MODEL", "gpt-4o")
    return ChatOpenAI(model=model, temperature=0.0)


def print_results(prompt_name: str, metrics: dict):
    print("=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)
    print("\nMétricas Derivadas:")
    for key in ["helpfulness", "correctness"]:
        if key in metrics:
            val = metrics[key]
            icon = "✓" if val >= 0.9 else "✗"
            print(f"  - {key.capitalize()}: {val:.2f} {icon}")
    print("\nMétricas Base:")
    for key in ["f1_score", "clarity", "precision"]:
        if key in metrics:
            val = metrics[key]
            icon = "✓" if val >= 0.9 else "✗"
            label = key.replace("_", "-").capitalize()
            print(f"  - {label}: {val:.2f} {icon}")
    failing = [k for k, v in metrics.items() if v < 0.9]
    print()
    if not failing:
        print("✅ STATUS: APROVADO - Todas as métricas >= 0.9")
    else:
        print("❌ STATUS: REPROVADO")
        print(f"⚠️  Métricas abaixo de 0.9: {', '.join(failing)}")
    print()
