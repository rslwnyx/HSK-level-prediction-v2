import jieba
import pandas as pd
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class Preprocessor:
    def __init__(self, user_dict_path=None):
        if user_dict_path:
            jieba.load_userdict(user_dict_path)
    
    @staticmethod
    def clean_text(text):
        clean_text = re.sub(r'\{.*?\}|\[.*?\]|[a-zA-Z0-9]', '', text)
        clean_text = re.sub(r'\s+', '', clean_text)
        clean_text = clean_text.replace('.+', '').replace('\\', '').strip()
        return clean_text

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    @staticmethod
    def split_sentences(text):
        sentences = re.split(r'(?<=[。！？])', text)
        return [s for s in sentences if s]