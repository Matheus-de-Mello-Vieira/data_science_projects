import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# %% Article content

df = pd.read_csv("news.csv")

texts = df.text
labels = df.label

x_train, x_test, y_train, y_test = train_test_split(texts,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# %% My contribution
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint

clf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words="english", max_df=0.7)),
    ('classifier', RandomForestClassifier(n_jobs=-1,
                                          oob_score=True,
                                          random_state=42,
                                          class_weight='balanced_subsample'))
])

clf.fit(x_train, y_train)
accuracy_score(y_test, clf.predict(x_test))
# 0.91

