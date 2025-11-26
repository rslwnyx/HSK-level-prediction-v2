import pandas as pd

word_files = [r"data\HSK\HSK1.csv",
              r"data\HSK\HSK2.csv",
              r"data\HSK\HSK3.csv",
              r"data\HSK\HSK4.csv",
              r"data\HSK\HSK5.csv",
              r"data\HSK\HSK6.csv"]

words_df = []

for i, file in enumerate(word_files, start=1):
    df = pd.read_csv(file)
    df["hsk_level"] = i
    df["type"] = "word"
    words_df.append(df)

words_df = pd.concat(words_df, ignore_index=True)
words_df = words_df.drop(columns=["en", "id"])
print(words_df.head())


grammar = pd.read_csv(r"data\HSK\grammar_chart.csv")
grammar = grammar.drop(columns=["pattern","pinyin","english","review","example","exampleTranslation","url"])
grammar["hsk_level"] = grammar["code"].astype(str).str.split(".").str[0]
grammar = grammar.drop(columns=["id", "code"])
grammar["type"] = "grammar"
grammar.rename(columns={"structure": "phrase"}, inplace=True)
print(grammar.head())

hsk_df = pd.concat([words_df, grammar], ignore_index=True)
hsk_df.to_csv(r"data\hsk_data.csv", index= False)