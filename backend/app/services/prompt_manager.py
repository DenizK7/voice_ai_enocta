import os
import yaml
from ruamel.yaml import YAML, CommentedMap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
PROMPT_FILE = os.path.join(BASE_DIR, "prompts", "prompts.yaml") 

class PromptManager:

    def __init__(self):
        """ `ruamel.yaml` yapılandırmasını hazırla. """
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.default_flow_style = False  
        self.yaml.width = 1000 

    def load_prompts(self) -> dict:
        if not os.path.exists(PROMPT_FILE):
            return {"adjustment_instructions": ""}
        with open(PROMPT_FILE, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {"adjustment_instructions": ""}  # Boş dosya kontrolü

    @staticmethod
    def save_prompts(new_prompt: dict):
        
        prompt_text = new_prompt.get("adjustment_instructions", "").strip()

        yaml_data = {"adjustment_instructions": prompt_text}

        with open(PROMPT_FILE, "w", encoding="utf-8") as file:
            yaml.dump(yaml_data, file, width=1000, allow_unicode=True, default_flow_style=False, default_style="|")



    def create_base_prompt(self, context: str, question: str) -> str:
        return f"""Aşağıdaki metinlerden yararlanarak soruya cevap veriniz.

        Context:
        {context}

        Question: {question}
        Answer:"""

    def optimize_prompt(self, base_prompt: str) -> str:
        prompt_data = self.load_prompts()
        adjustment_instructions = prompt_data.get("adjustment_instructions", "").strip()
        return f"{base_prompt.strip()}\n\n{adjustment_instructions}"
