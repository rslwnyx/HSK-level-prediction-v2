import pandas as pd
import jieba
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import HSK_DATA_PATH, HSK_COMP_PATH

class VocabAnalyzer:
    def __init__(self, user_dict_path=None):
        if user_dict_path:
            jieba.load_userdict(user_dict_path)
        self.vocab_data = self.load_vocab_data(HSK_DATA_PATH)
        self.grammar_data = self.load_grammar_data(HSK_DATA_PATH)
        self.hsk_compositions = pd.read_csv(HSK_COMP_PATH)

    @staticmethod
    def load_vocab_data(hsk_data_path):
        df = pd.read_csv(hsk_data_path)
        return df[df['type'] == 'word']
    
    @staticmethod
    def load_grammar_data(hsk_data_path):
        df = pd.read_csv(hsk_data_path)
        return df[df['type'] == 'grammar']