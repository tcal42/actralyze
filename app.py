import streamlit as st

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

# Import VADER sentiment analyzer.

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

st.title('Actralyze - analysis of survey responses')

st.subheader('Average sentiment by category')

df = pd.read_csv('train_comments.csv')

le = preprocessing.LabelEncoder()
le.fit(df['category'].tolist())
categories = list(le.classes_)

df2 = pd.read_csv('labeled_pool_comments.csv')

category = pd.DataFrame(le.inverse_transform(df2["preds"].values))
df2["category"] = category

# Compute and plot polarity scores.

positive_scores = []
negative_scores = []

for comment in df2.comment.values:
    vs = analyzer.polarity_scores(comment)
    positive_scores.append(vs["pos"])
    negative_scores.append(vs["neg"])
    
positive_df = pd.DataFrame(positive_scores)
negative_df = pd.DataFrame(negative_scores)

df2['positive'] = positive_df
df2['negative'] = -1*negative_df

df4 = df2.groupby(['category']).mean()
df4.drop(columns = ['preds'], inplace = True)
st.bar_chart(df4)

# Show distribution of categories

st.subheader('Distribution of categories')

df3 = df2.groupby(['category']).count()

hist_data = df3['comment']
st.bar_chart(hist_data)

st.subheader('Example comments:')

# Add a dropdown menu to view example text

add_selectbox = st.selectbox(
    'Choose a category to view example text',
    tuple(categories)
)

df_category = df2.loc[df2['category'] == add_selectbox]

if df_category.empty:
    st.write('No comments in this category.')
else:
    example_quotes = df_category.comment.values
    example_quote = example_quotes[np.random.randint(len(example_quotes))]
    st.write(example_quote)
    example_quote = example_quotes[np.random.randint(len(example_quotes))]
    st.write(example_quote)
    example_quote = example_quotes[np.random.randint(len(example_quotes))]
    st.write(example_quote)

