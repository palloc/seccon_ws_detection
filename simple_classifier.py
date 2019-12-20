#!/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import codecs
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

# File name and full path.
file_name = os.path.basename(__file__)
full_path = os.path.dirname(os.path.abspath(__file__))

# Dataset directory.
DATA_DIR = os.path.join(full_path, 'dataset')

# Dataset.
TRAIN_FILE = 'level1_train.tsv'
TRAIN_PATH = os.path.join(DATA_DIR, TRAIN_FILE)

# Define original stop words.
STOP_WORDS = []


# Check Grid Search option.
def check_grid_search(argv):
    if len(argv) == 2 and argv[1] == 'GRIDSEARCH':
        return True
    else:
        return False


# Load dataset.
def load_dataset(path):
    dataset = []
    X = []
    y = []
    with codecs.open(path, mode='r', encoding='utf-8') as fin:
        dataset.extend(fin.readlines())
    for payload in dataset:
        X.append(payload.split('\t')[0])
        y.append(payload.split('\t')[1].replace('\n', ''))
    return X, y


# Display classifier result.
def show_result(X_test, y_test, y_preds, classes, show_list=False, show_cm=False, show_acc=False):
    y_pred = []
    for pred in y_preds:
        if pred[0] > pred[1]:
            y_pred.append(classes[0])
        else:
            y_pred.append(classes[1])

    # Display result's list.
    if show_list:
        # Idx: データNo, Status: 成否, Label: 正解, Pred: モデルの予測結果, Payload: ペイロード
        print('Idx\tStatus\tLabel\tPred\tPayload')
        for idx, pred in enumerate(y_pred):
            status = ''
            if y_test[idx] == pred:
                status = 'o'
            else:
                status = 'x'
            print('{}\t{}\t{}\t{}\t{}'.format(idx + 1, status, y_test[idx], pred, X_test[idx]))

    # Display confusion matrix.
    if show_cm:
        print('Confusion Matrix:\n{}'.format(confusion_matrix(y_test, y_pred)))

    # Display accuracy.
    if show_acc:
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))

    return y_pred


# Vectorization.
def vectorization():
    # Vectorization.
    return CountVectorizer(lowercase=True,
                           tokenizer=None,
                           stop_words='english',
                           token_pattern=r'(?u)\b\w\w+\b|/|\)|;',
                           analyzer='word',
                           max_df=1.0,
                           min_df=1)


# Grid Search.
def grid_search(X, y):
    # Vectorization.
    vectorizer = vectorization()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    # Grid Search for MultinomialNB.
    print('MultinomialNB.')
    clf_nb = MultinomialNB()
    params = {'alpha': [0.1, 0.5, 1.0],
              'fit_prior': [True, False]}
    grid_search = GridSearchCV(clf_nb, param_grid=params, cv=10)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)

    # Grid Search MLP.
    print('MLPClassifier')
    clf_mlp = MLPClassifier()
    params = {'hidden_layer_sizes': [100, 200],
              'activation': ['tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'learning_rate_init': [0.01, 0.001],
              'max_iter': [100, 200]}
    grid_search = GridSearchCV(clf_mlp, param_grid=params, cv=10)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    print(grid_search.best_params_)


if __name__ == '__main__':
    # Load train data.
    print('Load train data from {}'.format(TRAIN_PATH))
    X, y = load_dataset(TRAIN_PATH)

    # Check Grid Search option.
    if check_grid_search(sys.argv):
        print('Grid Search.')
        grid_search(X, y)
        sys.exit(0)

    # Data split for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Vectorization.
    vectorizer = vectorization()
    vectorizer.fit(X_train)
    X = vectorizer.transform(X_train)

    # Fit using Naive Bayes.
    clf_nb = MultinomialNB(alpha=1.0,
                           fit_prior=False,
                           class_prior=None).fit(X, y_train)

    # Fit using MLP.
    clf_mlp = MLPClassifier(hidden_layer_sizes=100,
                            activation='relu',
                            solver='adam',
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            max_iter=200).fit(X, y_train)

    # Evaluate model.
    X = vectorizer.transform(X_test)
    print('Classifier: MultinomialNB')
    _ = show_result(X_test, y_test, clf_nb.predict_proba(X), clf_nb.classes_, show_list=True, show_cm=True, show_acc=True)
    print('-' * 200)
    print('Classifier: MLPClassifier')
    _ = show_result(X_test, y_test, clf_mlp.predict_proba(X), clf_mlp.classes_, show_list=True, show_cm=True, show_acc=True)
    print('-' * 200)
