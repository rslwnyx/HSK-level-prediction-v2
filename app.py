import streamlit as st
import joblib
import numpy as np
import scipy.sparse as sp
import jieba
from src.utils import clean_chinese_text, extract_hsk_features, hsk_tokenizer

st.set_page_config(page_title="HSK AI Predictor", page_icon="🇨🇳", layout="centered")


@st.cache_resource
def load_model_assets():
    regressor = joblib.load(r'models/hsk_regressor.pkl')
    tfidf = joblib.load(r'models/tfidf_vectorizer.pkl')
    thresholds = joblib.load(r'models/hsk_thresholds.pkl')

    word_dict, grammar_patterns = joblib.load(r'models/hsk_assets.pkl')
    return regressor, tfidf, thresholds, word_dict, grammar_patterns


def main():
    st.title("🇨🇳 HSK LEVEL PREDICTOR AI")

    try:
        model, tfidf, thresholds, w_dict, g_patterns = load_model_assets()
    except FileNotFoundError:
        st.error("Please run train.py first.")
        st.stop()

    #User Input
    user_text = st.text_area("Enter a chinese sentence:", height=150, placeholder="我觉得学习汉语很有意思...")

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter text.")
        else:
            with st.spinner("Computing..."):

                clean_text = clean_chinese_text(user_text)
                
                stat_feats = extract_hsk_features(clean_text, w_dict, g_patterns)
                tfidf_feats = tfidf.transform([clean_text])
                
                final_input = sp.hstack((tfidf_feats, [stat_feats]))
                
                pred_score = model.predict(final_input)[0]
                pred_score = max(0, min(500, pred_score)) # 0-500 clip
                
                level_idx = np.digitize(pred_score, thresholds, right=True)
                final_level = int(max(1, min(6, level_idx)))
                

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted Level: ", f"HSK {final_level}")
                with c2:
                    st.metric("Model Score (0-500)", f"{pred_score:.1f}")
                
                st.subheader("Mathematical Analysis")
                c3, c4 = st.columns(2)
                with c3:
                    st.metric("Sentence Length", f"{int(stat_feats[-2])} words")
                with c4:
                    st.metric("Weighted Complexity Index", f"{stat_feats[-1]:.2f}")
                    st.caption("(Higher = More vocab difficulty)")
                

                st.subheader("Detected Structures")
                
                hsk_dist = {f"HSK{i} density": stat_feats[i-1] for i in range(1, 7)}
                st.bar_chart(hsk_dist)
                
                tokens = jieba.lcut(clean_text)
                st.text("Tokenizasyon:")
                st.code(" | ".join(tokens))

if __name__ == "__main__":
    main()