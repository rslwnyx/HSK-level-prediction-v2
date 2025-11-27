import pandas as pd
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
MODEL_OUTPUT_PATH = os.path.join('models', 'hsk_rf_model.pkl')

from itertools import product
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessor import Preprocessor
from src.analyzers.vocab_analyzer import VocabAnalyzer
from src.analyzers.grammar_analyzer import GrammarAnalyzer
from src.scorer import Scorer
from src.config import *


col_names = ["composition_id", "nationality", "gender", "exam_date",
              "composition_title", "speaking_test_score", "composition_score",
              "listening_score", "reading_score", "general_score", "total_score", 
              "certificate", "sample_sentence"]

def load_data():
    df = pd.read_csv(HSK_COMP_PATH, names=col_names)
    return df

def derive_labels_from_scores(df):
    df.copy()
    df['total_score'] = pd.to_numeric(df['total_score'], errors='coerce')
    df = df.dropna(subset=['total_score'])
    df['total_score'] = df['total_score'].astype(float)

    df['hsk_label'] = pd.qcut(df['total_score'], q=6, labels=[1, 2, 3, 4, 5, 6])
    return df

def extract_features(text, vocab, grammar, prep):
    cleaned_text = prep.clean_text(text)
    tokens = prep.tokenize(cleaned_text)
    sentences = prep.split_sentences(cleaned_text)

    vocab_result = vocab.analyze(tokens, vocab.vocab_data)
    grammar_result = grammar.analyze_text(sentences)

    features = {
        "vocab_avg": vocab_result.get("avg_level", 0),
        "vocab_max": vocab_result.get("max_level", 0),
        "total_words": len(tokens),
        "unique_words": len(set(tokens)),


        "ratio_hsk1": vocab_result.get('level_counts', {}).get(1, 0) / max(1, len(tokens)),
        "ratio_hsk2": vocab_result.get('level_counts', {}).get(2, 0) / max(1, len(tokens)),
        "ratio_hsk3": vocab_result.get('level_counts', {}).get(3, 0) / max(1, len(tokens)),
        "ratio_hsk4": vocab_result.get('level_counts', {}).get(4, 0) / max(1, len(tokens)),
        "ratio_hsk5": vocab_result.get('level_counts', {}).get(5, 0) / max(1, len(tokens)),
        "ratio_hsk6": vocab_result.get('level_counts', {}).get(6, 0) / max(1, len(tokens)),


        "grammar_score": grammar_result.get("total_grammar_score", 0),
        "grammar_patterns_count": len(grammar_result.get("matched_rules", [])),


        "avg_sentence_length": np.mean([len(s) for s in sentences]) if sentences else 0,
        "sentence_count": len(sentences)
    }

    return features



def train_model():

    df = load_data()

    df = derive_labels_from_scores(df)

    prep = Preprocessor()
    vocab = VocabAnalyzer()
    grammar = GrammarAnalyzer()

    feature_list = []
    for text in df['sample_sentence']:
        features = extract_features(text, vocab, grammar, prep)
        feature_list.append(features)

    X = pd.DataFrame(feature_list)
    y = df['hsk_label'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_OUTPUT_PATH)


    all_preds = clf.predict(X)
    df['predicted_HSK_level'] = all_preds

    output_cols = ['sample_sentence', 'total_score', 'predicted_HSK_level']
    df[output_cols].to_csv(PREDICTIONS_OUTPUT_PATH, index=False)


if __name__ == "__main__":
    train_model()