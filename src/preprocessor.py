import jieba
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Preprocessor:
    def __init__(self, user_dict_path=None):
        if user_dict_path:
            jieba.load_userdict(user_dict_path)
    
    @staticmethod
    def clean_text(text: str) -> str:
        clean_text = re.sub(r'\{.*?\}|\[.*?\]|[a-zA-Z0-9]', '', text)
        clean_text = re.sub(r'\s+', '', clean_text)
        clean_text = clean_text.replace('.+', '').replace('\\', '').strip()
        return clean_text

    def tokenize(self, text: str) -> list[str]:
        return jieba.lcut(text)

    def split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[。！？])', text)
        return [s for s in sentences if s]