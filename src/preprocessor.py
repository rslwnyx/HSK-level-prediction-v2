import jieba
import pandas as pd
import re

class Preprocessor:
    def __init__(self, user_dict_path=None):
        if user_dict_path:
            jieba.load_userdict(user_dict_path)
        pass
    
    @staticmethod
    def clean_text(text):
        clean_text = re.sub(r'\{.*?\}|\[.*?\]|[a-zA-Z0-9]', '', text)
        clean_text = re.sub(r'\s+', '', clean_text)
        return clean_text

    @staticmethod
    def segment_text(text):
        return jieba.lcut(text)

    @staticmethod
    def split_sentences(text):
        sentences = re.split(r'(?<=[。！？])', text)
        return [s for s in sentences if s]