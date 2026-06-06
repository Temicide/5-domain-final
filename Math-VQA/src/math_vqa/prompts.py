from __future__ import annotations


BASE_PROMPT = """You are solving a Thai/English math problem from an image.
Read all text, diagrams, graphs, shapes, and answer choices carefully.
Return only the final answer.
Do not explain.
Do not include "answer:".
Use Arabic digits when possible."""

DIAGRAM_PROMPT = """You are solving a math problem from an image.
The image may contain Thai text, English text, diagrams, graphs, shapes, or visual answer choices.
Use the visual information, not only OCR text.
Return only the final answer."""

STRICT_FORMAT_PROMPT = """Return exactly one short answer string.
No explanation.
No units unless the unit is necessary to distinguish the answer.
Use plain fractions such as 17/10.
Use sqrt notation for radicals."""

DIAGRAM_IMAGE_IDS = {"451", "569"}
PROMPTS = {
    "base": f"{BASE_PROMPT}\n\n{STRICT_FORMAT_PROMPT}",
    "diagram": f"{DIAGRAM_PROMPT}\n\n{STRICT_FORMAT_PROMPT}",
    "strict": STRICT_FORMAT_PROMPT,
}


def select_prompt_name(image_id: object) -> str:
    if str(image_id) in DIAGRAM_IMAGE_IDS:
        return "diagram"
    return "base"


def build_prompt(prompt_name: str = "base", few_shot_examples: list[dict[str, str]] | None = None) -> str:
    if prompt_name not in PROMPTS:
        raise ValueError(f"unknown prompt name: {prompt_name}")
    prompt = PROMPTS[prompt_name]
    if not few_shot_examples:
        return prompt
    example_lines = ["Examples of accepted final-answer strings:"]
    for example in few_shot_examples:
        example_lines.append(f"- id {example['id']}: {example['answer']}")
    return f"{prompt}\n\n" + "\n".join(example_lines)
