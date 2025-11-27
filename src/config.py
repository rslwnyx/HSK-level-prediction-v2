import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#Scoring weights
VOCAB_WEIGHT = 0.4
GRAMMAR_WEIGHT = 0.5
SENTENCE_LENGTH_WEIGHT = 0.1

#Penalties and bonuses
PENALTY_THRESHOLD = 3.0
PENALTY_RATE = 0.2
LENGHTS_BONUS_THRESHOLD = 15

#Data file paths
HSK_DATA_PATH = r"data\hsk_data.csv"
HSK_COMP_PATH= r"data\hsk_composition.csv"

