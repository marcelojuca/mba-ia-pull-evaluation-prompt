"""Push optimized prompts to LangSmith Prompt Hub."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_FILE = PROMPTS_DIR / "bug_to_user_story_v2.yml"
PUSH_NAME = "marcelojuca/bug_to_user_story_v2"


def load_prompt_yaml(yaml_path: Path) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    messages = []
    for msg in prompt_data["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            messages.append(SystemMessagePromptTemplate.from_template(content))
        elif role == "human":
            messages.append(HumanMessagePromptTemplate.from_template(content))
    return ChatPromptTemplate.from_messages(messages)


def push_prompt_to_langsmith(prompt_data: dict, prompt_name: str, is_public: bool = True) -> str:
    client = Client()
    chat_prompt = build_chat_prompt(prompt_data)

    techniques = prompt_data.get("metadata", {}).get("techniques", [])
    description = prompt_data.get("description", "")
    tags = techniques + ["ChatPromptTemplate", "bug-to-user-story"]

    url = client.push_prompt(
        prompt_name,
        object=chat_prompt,
        is_public=is_public,
        description=description,
        tags=tags,
    )
    return url


def main():
    print(f"Carregando prompt de: {PROMPT_FILE}")
    prompt_data = load_prompt_yaml(PROMPT_FILE)

    techniques = prompt_data.get("metadata", {}).get("techniques", [])
    print(f"Técnicas aplicadas: {techniques}")

    print(f"\nFazendo push para LangSmith como: {PUSH_NAME}")
    url = push_prompt_to_langsmith(prompt_data, PUSH_NAME, is_public=True)

    print(f"\n✅ Push concluído com sucesso!")
    print(f"URL: {url}")
    print(f"\nPrompt publicado: {PUSH_NAME}")
    print(f"Versão: {prompt_data.get('version', 'N/A')}")
    print(f"Técnicas: {', '.join(techniques)}")


if __name__ == "__main__":
    main()
