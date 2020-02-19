# Actralyze

Actralyze is an active transfer learning framework for learning from sparsely labeled text data. In particular, it makes use of the deep language representation model BERT and is fine-tuned with an additional classification layer for classifying text survey responses into a set of pre-defined categories. Additionally, the model outputs a set of the least certain predictions, which can then be labeled by an expert annotator and fed back into the model in an active learning loop. This provides an efficient and surprisingly simple framework for semi-supervised learning from text data with a limited labeling budget. 

This project was done in consultation with the company Activated Insights, who specialize in analyzing survey responses from employees and residents at senior care facilities. In addition to classifying the survey responses into different categories, sentiment analysis is also done using the VADER package. The average sentiment scores for each class are displayed along with the distribution of class labels and randomly selected example texts from each class in an interactive dashboard.

In an attempt to keep the code simple and reusable for my clients, the model is contained within a single IPython notebook, and the front-end visualizations are created using Streamlit. Model fine-tuning can be accelerated with a GPU, but it is not strictly necessary.
