import pandas as pd
import jieba
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import HSK_DATA_PATH, HSK_COMP_PATH
from preprocessor import Preprocessor

class GrammarAnalyzer:
    def __init__(self, user_dict_path=None):
        self.grammar_rules = []
        self.grammar_data = self.load_grammar_data(HSK_DATA_PATH)

    def load_grammar_data(self, hsk_data_path):
        df = pd.read_csv(hsk_data_path)
        df.columns = ['phrase','level','type']

        for _, row in df.iterrows():
            phrase = str(row['phrase']).strip()
            try:
                level = int(row['level'])
            except:
                continue
        
            dtype = str(row['type']).strip().lower()

            if dtype == 'grammar':
                safe_phrase = Preprocessor.clean_text(phrase)
                has_chinese = re.search(r'[\u4e00-\u9fff]', phrase)

                is_empty = len(safe_phrase) == 0

                if not has_chinese or is_empty:
                    continue

                self.grammar_rules.append((safe_phrase, min(6, level)))
        
        return self.grammar_rules
    
    def analyze_sentence(self, sentence):
        matched_rules = []

        for rule, level in self.grammar_data:
            if rule in sentence:
                sentence_score += level
                matched_rules.append((rule, level))

        return sentence_score, matched_rules
    
    def analyze_text(self, sentences):
        total_grammar_score = 0
        max_grammar_level = 0
        all_matched_rules = []

        for sentence in sentences:
            sentence_score, matched_rules = self.analyze_sentence(sentence)
            total_grammar_score += sentence_score
            all_matched_rules.extend(matched_rules)
            for _, level in matched_rules:
                if level > max_grammar_level:
                    max_grammar_level = level

        avg_grammar_score = total_grammar_score / len(sentences) if sentences else 0    

        return {
            "total_grammar_score": int(total_grammar_score),
            "avg_grammar_score": float(avg_grammar_score),
            "max_grammar_level": int(max_grammar_level),
            "matched_rules": all_matched_rules
        }

        