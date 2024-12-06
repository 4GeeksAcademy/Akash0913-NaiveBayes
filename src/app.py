import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pickle import dump

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv")
print(total_data.head())

def apply_preprocess(df):
    df = df.drop("package_name", axis=1)
    df["review"] = df["review"].str.strip().str.lower()

    return df

total_data = apply_preprocess(total_data)

print(total_data.head())

X = total_data["review"]
y = total_data["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.head())

vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()

print(X_train)

model = MultinomialNB()
print(model.fit(X_train, y_train))

y_pred = model.predict(X_test)
print(y_pred)

print(accuracy_score(y_test, y_pred))

for model_aux in [GaussianNB(), BernoulliNB()]:
    model_aux.fit(X_train, y_train)
    y_pred_aux = model_aux.predict(X_test)
    print(f"{model_aux} with accuracy: {accuracy_score(y_test, y_pred_aux)}")

hyperparams = {
    "alpha": np.linspace(0.01, 10.0, 200),
    "fit_prior": [True, False]
}

random_search = RandomizedSearchCV(model, hyperparams, n_iter = 50, scoring = "accuracy", cv = 5, random_state = 42)
print(random_search)

random_search.fit(X_train, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")

model = MultinomialNB(alpha = 1.917638190954774, fit_prior = False)
model.fit(X_train, y_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

