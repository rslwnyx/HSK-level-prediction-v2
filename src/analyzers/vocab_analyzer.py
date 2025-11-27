import pandas as pd
import jieba
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import HSK_DATA_PATH

class VocabAnalyzer:
    def __init__(self):
        self.vocab_data = self.load_vocab_data(HSK_DATA_PATH)

    def load_vocab_data(self, hsk_data_path):
        df = pd.read_csv(hsk_data_path)
        df.columns = ['phrase','level','type']
        word_dict = {}
        for _, row in df.iterrows():
            phrase = str(row['phrase']).strip()
            try:
                level = int(row['level'])
            except:
                continue
        
        dtype = str(row['type']).strip().lower()

        if dtype == 'word':
            word_dict[phrase] = level

        return word_dict

    @staticmethod
    def get_word_level( word, vocab_data):
        if word in vocab_data:
            return vocab_data[word]
        else:
            return 0
        
    def analyze(self, tokens, vocab_data):
        total_words, total_hsk_score = 0, 0,
        level_counts = {"HSK1": 0, "HSK2": 0, "HSK3": 0, "HSK4": 0, "HSK5": 0, "HSK6": 0}
        for token in tokens:
            word = token.strip()
            if not word:
                continue

            level = VocabAnalyzer.get_word_level(word, vocab_data)
            if level > 0:
                total_hsk_score += level
                level_counts[f"HSK{level}"] += 1
                total_words += 1

        avg_level = total_hsk_score / total_words if total_words > 0 else 0
        max_level = max([lvl for lvl in range(1, 7) if level_counts[f"HSK{lvl}"] > 0], default=0)

        return {
            "total_words": int(total_words),
            "avg_level": float(avg_level),
            "max_level": int(max_level),
            "level_counts": dict(level_counts)
        }
    

                
            
