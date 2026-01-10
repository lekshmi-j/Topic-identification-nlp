**Topic Identification using NLP**

(Unsupervised Topic Modeling & Supervised Topic Classification)

🔍** Project Overview**

This project explores topic identification in text data using both unsupervised and supervised NLP techniques.
The goal is to understand how topics can be discovered, interpreted, and predicted from raw text using classical machine learning methods.

The project covers the full NLP pipeline:

Text preprocessing

Topic discovery (LDA, NMF)

Topic evaluation & interpretation

Supervised topic classification

Inference on unseen text

Clean modular ML code

❓** Problem Statement**

Given a large collection of text documents:

How can we discover hidden topics without labels?

How can we assign topics to new documents automatically?

What are the trade-offs between topic modeling and classification?

This project answers these questions using the 20 Newsgroups dataset.

📊** Dataset**

Dataset: 20 Newsgroups
Description:
A classic NLP dataset containing ~20,000 documents across 20 discussion topics such as:

Politics

Religion

Sports

Science

Technology

Each document belongs to exactly one newsgroup, which allows us to:

Use the data without labels (topic modeling)

Use the same data with labels (topic classification)

🧠 **Approach**
1️⃣ Text Preprocessing

Lowercasing

Tokenization

Stopword removal

Lemmatization

Bigram generation

Rare & overly common word filtering

Cleaned text is stored for reuse across models.

2️⃣ Unsupervised Topic Modeling

Used to discover latent topics without labels.

Models:

LDA (Latent Dirichlet Allocation) — probabilistic, generative

NMF (Non-negative Matrix Factorization) — matrix factorization, highly interpretable

Vectorization:

LDA → Bag of Words

NMF → TF-IDF

Each document is represented as a mixture of topics, not a single label.

3️⃣ Topic Evaluation & Interpretation

Since topic modeling has no “accuracy”, evaluation is done using:

Topic coherence

Topic diversity

Human-in-the-loop interpretation

Topics were manually named based on their most representative words.

4️⃣ Supervised Topic Classification

Used to predict known topic labels.

Models:

Logistic Regression

Multinomial Naive Bayes

Pipeline:

TF-IDF vectorization

Train/test split (stratified)

Accuracy, precision, recall, F1-score

Confusion matrix analysis

5️⃣ Inference

The project supports inference for:

Topic Modeling → topic mixtures for unseen text

Topic Classification → predicted topic label

This demonstrates end-to-end usability, not just training.

🤖 **Models Used**
Task	Model
Topic Discovery	LDA
Topic Discovery	NMF
Topic Prediction	Logistic Regression
Topic Prediction	Naive Bayes

Results
🔹 Supervised Classification (Examples)
Text: NASA successfully launched a new satellite into space
Predicted topic: sci.space

Text: The car engine performance has improved significantly
Predicted topic: rec.autos

Text: The government announced new gun control policies
Predicted topic: talk.politics.guns

🔹 Topic Modeling Inference (LDA Example)
Text: NASA successfully launched a new satellite into orbit
Politics / Space / Media → probability 0.864
Information Requests → probability 0.007
Government / Products → probability 0.007

🔹 Topic Modeling Inference (NMF Example)
Text: NASA successfully launched a new satellite into orbit
Science / Space → weight 0.721
General Discussion → weight 0.041

🧪 Model Comparison & Insights
Aspect	Topic Modeling	Topic Classification
Learning Type	Unsupervised	Supervised
Requires Labels	No	Yes
Output	Topic mixture	Single topic
Evaluation	Coherence	Accuracy / F1
Best Use	Discovery	Automation

Key insight:
No single model is “best” — model choice depends on the task and data availability.

📚** Key Learnings**

Topic modeling and classification solve different problems

Accuracy is not meaningful for unsupervised models

Human interpretation is essential for topic quality

Modular ML code improves maintainability

Inference support is crucial for real-world usage

🛠️ Project Structure
topic-identification/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_topic_classification.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   ├── classify.py
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── requirements.txt
└── README.md
**
🚀 How to Run
**
Install dependencies

pip install -r requirements.txt


Run notebooks in order:

01_eda.ipynb

02_preprocessing.ipynb

03_topic_modeling.ipynb

04_topic_classification.ipynb

