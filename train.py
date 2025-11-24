import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import jieba

from src.utils import clean_chinese_text, load_hsk_data, extract_hsk_features, hsk_tokenizer

col_names = ["composition_id", "nationality", "gender", "exam_date",
              "composition_title", "speaking_test_score", "composition_score",
              "listening_score", "reading_score", "general_score", "total_score", 
              "certificate", "sample_sentence"]
def main():

    df = pd.read_csv(r'C:\Users\aleyna nur\Desktop\HSK LEVEL PREDICTION\data\hsk_composition.csv') 
    df.columns = col_names
    word_dict, grammar_patterns = load_hsk_data(r'C:\Users\aleyna nur\Desktop\HSK LEVEL PREDICTION\data\hsk_data.csv')
    

    df["clean_text"] = df["sample_sentence"].apply(clean_chinese_text)
    df = df.dropna(subset=["total_score", "clean_text"])


    X_stats = df['clean_text'].apply(
        lambda x: extract_hsk_features(x, word_dict, grammar_patterns)
    ).tolist()
    X_stats = np.array(X_stats)


    tfidf = TfidfVectorizer(tokenizer=hsk_tokenizer, max_features=3000, token_pattern=None)
    X_tfidf = tfidf.fit_transform(df['clean_text'])


    X_final = sp.hstack((X_tfidf, X_stats))
    y = df["total_score"].values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    #Random Forest Regressor
    reg_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    reg_model.fit(X_train, y_train)

    #Evaluation
    y_pred = reg_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\nModel Evaluation:\nRMSE: {rmse:.2f} \nR2 Score: {r2:.2f}")

    #Threshold Calibration
    _, raw_bins = pd.qcut(y_train, q=6, retbins=True, labels=False)
    final_thresholds = raw_bins.copy()
    final_thresholds[0] = 0.0  #Lowest possible score
    final_thresholds[-1] = 500.0  #Highest possible score
    
    print(f"Computed HSK boundaries: {final_thresholds}")

    #Artifact saving
    joblib.dump(reg_model, r'models/hsk_regressor.pkl')
    joblib.dump(tfidf, r'models/tfidf_vectorizer.pkl')
    joblib.dump(final_thresholds, r'models/hsk_thresholds.pkl')

    joblib.dump((word_dict, grammar_patterns), r'models/hsk_assets.pkl')


if __name__ == "__main__":
    main()