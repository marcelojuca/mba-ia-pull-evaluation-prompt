"""
Validation tests for the optimized bug_to_user_story_v2 prompt.

Run: pytest tests/test_prompts.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


@pytest.fixture(scope="module")
def prompt_data():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def system_content(prompt_data):
    for msg in prompt_data.get("messages", []):
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""


def test_prompt_has_system_prompt(prompt_data, system_content):
    """Verifica se o campo system prompt existe e não está vazio."""
    assert system_content, "O prompt deve ter uma mensagem de sistema (role: system) não vazia"
    assert len(system_content.strip()) > 50, "O system prompt deve ter conteúdo substancial"


def test_prompt_has_role_definition(system_content):
    """Verifica se o prompt define uma persona (Role Prompting)."""
    role_indicators = ["você é", "voce é", "you are", "você atua", "sua função", "especialista", "sênior", "senior"]
    content_lower = system_content.lower()
    has_role = any(indicator in content_lower for indicator in role_indicators)
    assert has_role, (
        "O prompt deve definir uma persona/role (ex: 'Você é um Product Manager Sênior...'). "
        f"Procurado em: {content_lower[:200]}"
    )


def test_prompt_mentions_format(system_content):
    """Verifica se o prompt exige formato Markdown ou User Story padrão."""
    format_indicators = [
        "markdown", "user story", "como ", "quero ", "para que",
        "critérios de aceite", "criterios de aceite", "##", "**como**", "formato"
    ]
    content_lower = system_content.lower()
    has_format = any(indicator in content_lower for indicator in format_indicators)
    assert has_format, (
        "O prompt deve especificar o formato de saída (Markdown ou User Story padrão). "
        "Inclua 'Como [persona], Quero [ação], Para que [benefício]' e Critérios de Aceite."
    )


def test_prompt_has_few_shot_examples(system_content):
    """Verifica se o prompt contém exemplos de entrada/saída (Few-shot Learning)."""
    example_indicators = [
        "exemplo", "example", "input:", "output:", "bug report:", "user story:", "###"
    ]
    content_lower = system_content.lower()
    has_examples = any(indicator in content_lower for indicator in example_indicators)
    assert has_examples, (
        "O prompt deve incluir exemplos de entrada/saída (Few-shot Learning). "
        "Adicione ao menos 1 exemplo de Bug Report → User Story."
    )
    # Check for multiple examples
    example_count = content_lower.count("exemplo") + content_lower.count("example")
    assert example_count >= 1, "Deve haver pelo menos 1 exemplo de few-shot"


def test_prompt_no_todos(prompt_data):
    """Garante que não há [TODO] esquecidos no prompt."""
    full_content = yaml.dump(prompt_data, allow_unicode=True)
    assert "[TODO]" not in full_content.upper().replace(" ", ""), (
        "O prompt contém marcadores [TODO] não resolvidos. Remova-os antes de entregar."
    )
    assert "[TODO" not in full_content, "O prompt contém [TODO] não resolvido"


def test_minimum_techniques(prompt_data):
    """Verifica se pelo menos 2 técnicas foram listadas nos metadados."""
    techniques = prompt_data.get("metadata", {}).get("techniques", [])
    assert isinstance(techniques, list), "metadata.techniques deve ser uma lista"
    assert len(techniques) >= 2, (
        f"O prompt deve aplicar pelo menos 2 técnicas de prompt engineering. "
        f"Encontradas: {techniques}"
    )
    # Check that techniques are non-empty strings
    for tech in techniques:
        assert isinstance(tech, str) and tech.strip(), (
            f"Cada técnica deve ser uma string não vazia. Encontrado: {tech!r}"
        )
