"""Pull prompts from LangSmith and save locally as YAML."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
from langsmith import Client


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"
OUTPUT_FILE = PROMPTS_DIR / "bug_to_user_story_v1.yml"


def pull_prompt_from_langsmith(prompt_name: str) -> dict:
    client = Client()
    prompt = client.pull_prompt(prompt_name)

    messages = []
    for msg in prompt.messages:
        role_type = type(msg).__name__
        if "System" in role_type:
            role = "system"
        elif "Human" in role_type:
            role = "human"
        elif "AI" in role_type or "Assistant" in role_type:
            role = "assistant"
        else:
            role = "human"
        messages.append({"role": role, "content": msg.prompt.template})

    owner, repo = prompt_name.split("/") if "/" in prompt_name else ("", prompt_name)
    return {
        "name": repo,
        "version": "1.0.0",
        "description": f"Prompt pulled from LangSmith: {prompt_name}",
        "metadata": {
            "source": prompt_name,
            "techniques": [],
            "language": "pt-BR",
            "input_variables": list(prompt.input_variables),
        },
        "messages": messages,
    }


def save_prompt_yaml(prompt_data: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(prompt_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"Prompt salvo em: {output_path}")


def main():
    print(f"Fazendo pull do prompt: {SOURCE_PROMPT}")
    prompt_data = pull_prompt_from_langsmith(SOURCE_PROMPT)
    save_prompt_yaml(prompt_data, OUTPUT_FILE)
    print("Pull concluído com sucesso!")
    print(f"Técnicas: {prompt_data['metadata']['techniques']}")
    print(f"Variáveis de entrada: {prompt_data['metadata']['input_variables']}")


if __name__ == "__main__":
    main()
