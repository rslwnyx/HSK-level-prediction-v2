import jieba
import pandas as pd
import re

def clean_text(text):
    clean_text = re.sub(r'\{.*?\}|\[.*?\]|[a-zA-Z0-9]', '', text)
    clean_text = re.sub(r'\s+', '', clean_text)
    return clean_text

def segment_text(text):
    return jieba.lcut(text)

def split_sentences(text):
    sentences = re.split(r'(?<=[。！？])', text)
    return [s for s in sentences if s]