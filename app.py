import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

try:
    from src.preprocessor import Preprocessor
    from src.analyzers.vocab_analyzer import VocabAnalyzer
    from src.analyzers.grammar_analyzer import GrammarAnalyzer
    from src.config import MODEL_PATH
except ImportError:
    st.error("Error: There are missing files.")
    st.stop()

st.set_page_config(page_title="HSK Predictor", page_icon="🇨🇳")


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            return None
    return None

def extract_features_ui(text, prep, vocab, grammar):

    cleaned = prep.clean_text(text)
    tokens = prep.tokenize(cleaned)
    sentences = prep.split_sentences(cleaned)
    
    v_res = vocab.analyze(tokens)
    g_res = grammar.analyze_text(sentences)
    
    features = {
        'vocab_avg': v_res.get('average_level', 0),
        'vocab_max': v_res.get('max_level', 0),
        'total_words': len(tokens),
        'unique_words': len(set(tokens)),
        'grammar_score': g_res.get('grammar_score', 0),
        'grammar_patterns_count': len(g_res.get('matched_patterns', [])),
        'avg_sentence_len': np.mean([len(s) for s in sentences]) if sentences else 0,
        'sentence_count': len(sentences)
    }
    return pd.DataFrame([features]).fillna(0), v_res, g_res


st.title("🇨🇳 HSK AI Predictor (ML)")
st.info("Chinese level prediction with Random Forest Algorithm.")

model = load_model()

if not model:
    st.warning("Model not found. Please train the model first.")
    if st.button("Train model"):
        with st.spinner("Training..."):
            os.system("python train.py")
            time.sleep(2)
            st.rerun()
else:
    st.success("Model loaded successfully.")

text_input = st.text_area("Çince Metin:", "我正在学习汉语，虽然很难但是很有趣。", height=100)

if st.button("Analyze", type="primary"):
    if not model:
        st.error("Model not found.")
    else:
        prep = Preprocessor()
        vocab = VocabAnalyzer()
        grammar = GrammarAnalyzer()
        
        with st.spinner("Analyzing..."):
            features_df, v_res, g_res = extract_features_ui(text_input, prep, vocab, grammar)
            
            # Tahmin
            prediction = model.predict(features_df)[0]
            probs = model.predict_proba(features_df)[0]
            confidence = np.max(probs) * 100
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.metric("Predicted level", f"HSK {prediction}")
            st.caption(f"Confidence score: %{confidence:.1f}")
        
        with c2:
            st.subheader("Porbability Distribution")
            chart_data = pd.DataFrame({
                "Level": [f"HSK {i+1}" for i in range(len(probs))],
                "Probabilty": probs
            })
            st.bar_chart(chart_data.set_index("Level"))
            
        with st.expander("Detailed Features and Analysis"):
            st.json(features_df.to_dict(orient='records')[0])