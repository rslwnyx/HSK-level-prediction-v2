import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import VOCAB_WEIGHT, GRAMMAR_WEIGHT, PENALTY_THRESHOLD, PENALTY_RATE


class Scorer:

    def calculate_final_prediction(self, vocab_result, grammar_result):
        vocab_score = vocab_result.get("avg_level", 0)
        grammar_score = grammar_result.get("avg_grammar_score", 0)

        base_score = (vocab_score * VOCAB_WEIGHT) + (grammar_score * GRAMMAR_WEIGHT)

        gap = abs(vocab_score - grammar_score)
        penalty = 0.0

        if gap > PENALTY_THRESHOLD:
            penalty = gap*PENALTY_RATE
            final_score = base_score - penalty
            print(f"Penalty applied: {penalty}, Final Score: {final_score}")
        else:
            final_score = base_score
            print(f"No penalty applied, Final Score: {final_score}")

        
        return max(0.0, final_score)


    def generate_report(self, final_score, vocab_result, grammar_result):
            estimated_level = round(final_score)
            report_string = f"Estimated HSK Level: HSK {estimated_level}\n"
            report_string += f"Vocabulary Average Level: {vocab_result.get('avg_level', 0):.2f}\n"
            report_string += f"Grammar Average Level: {grammar_result.get('avg_grammar_score', 0):.2f}\n"
            report_string += f"Total Vocabulary Words Analyzed: {vocab_result.get('total_words', 0)}\n"
            report_string += f"Vocabulary Level Counts: {vocab_result.get('level_counts', {})}\n"
            report_string += f"Total Grammar Points Matched: {grammar_result.get('total_grammar_points', 0)}\n"
            report_string += f"Grammar Level Counts: {grammar_result.get('grammar_level_counts', {})}\n"
            
            return report_string