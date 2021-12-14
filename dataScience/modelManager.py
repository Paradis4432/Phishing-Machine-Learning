from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import pickle
import numpy as np
import pandas as pd



def makeNBmodel(df, targetColumn, saveModel, returnDict):
    from sklearn.naive_bayes import GaussianNB

    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]
    scaler = StandardScaler()

    # Setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Setup model and fit
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    if saveModel:
        with open('./pickleModels/NBmodel', 'wb') as f:
            pickle.dump(model, f)

    # Generate confusion matrix and get matrix
    import graphManager as gm

    mat = gm.make7x7ConsusionMatrix(y_test, y_pred, "Naive Bayes", "Predicted", "Actual",
                                    "./images/NB_confusion_matrix.png", True, True)

    # Get extra data
    TP = mat[1, 1]
    TN = mat[0, 0]
    FP = mat[0, 1]
    FN = mat[1, 0]

    classErrorNB = (1 - accuracy)
    recallNB = recall_score(y_test, y_pred)
    specificityNB = TN / (TN + FP)
    precisionNB = precision_score(y_test, y_pred)
    f1NB = f1_score(y_test, y_pred)
    
    if returnDict:
        return {
            'Modelo': 'Naive Bayes',
            'Accuracy': accuracy,
            'Classification Error': classErrorNB,
            'Recall': recallNB,
            'Specificity': specificityNB,
            'Precision': precisionNB,
            'F1 Score': f1NB,
        }, y_pred


def findOptimalCvalueForLG(X_train, y_train):
    # this function does magic and finds the optimal C value for the Logistic Regression model
    from sklearn import linear_model

    scores_para_df = []
    C_tunning = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for i in C_tunning:
        lg = linear_model.LogisticRegression(C=i, max_iter=500)
        cv_scores = cross_val_score(lg, X_train, y_train, cv=5)
        dict_row_score = {'score_medio': np.mean(
            cv_scores), 'score_std': np.std(cv_scores), 'C': i}
        scores_para_df.append(dict_row_score)

    scores = pd.DataFrame(scores_para_df)
    return scores[scores["score_medio"] == scores["score_medio"].max()]["C"].values[0]


def makeLGmodel(df, targetColumn, saveModel, returnDict):
    from sklearn.linear_model import LogisticRegression
    # Setup X, y
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]
    scaler = StandardScaler()

    # Setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    optimalCvalue = findOptimalCvalueForLG(X_train, y_train)

    # Setup model and fit
    model = LogisticRegression(C=optimalCvalue)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    if saveModel:
        with open('./pickleModels/LGmodel', 'wb') as f:
            pickle.dump(model, f)

    # Generate confusion matrix and get matrix
    import graphManager as gm

    mat = gm.make7x7ConsusionMatrix(y_test, y_pred, "Logistic Regression", "Predicted", "Actual",
                                    "./images/LG_confusion_matrix.png", True, True)

    # Get extra data
    TP = mat[1, 1]
    TN = mat[0, 0]
    FP = mat[0, 1]
    FN = mat[1, 0]

    classErrorLG = (1 - accuracy)
    recallLG = recall_score(y_test, y_pred)
    specificityLG = TN / (TN + FP)
    precisionLG = precision_score(y_test, y_pred)
    f1LG = f1_score(y_test, y_pred)

    if returnDict:
        return {
            'Modelo': 'Regresion Logistica',
            'Accuracy': accuracy,
            'Classification Error': classErrorLG,
            'Recall': recallLG,
            'Specificity': specificityLG,
            'Precision': precisionLG,
            'F1 Score': f1LG,
        }, y_pred


def findOptimalNeighborsForKNN(X_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier

    KNN_scores_para_df = []

    for i in range(1, 100, 10):
        KNN_model = KNeighborsClassifier(n_neighbors=i)
        KNN_cv_scores = cross_val_score(KNN_model, X_train, y_train, cv=5)
        KNN_dict_row_score = {'score_medio': np.mean(
            KNN_cv_scores), 'score_std': np.std(KNN_cv_scores), 'n_neighbours': i}
        KNN_scores_para_df.append(KNN_dict_row_score)
    scoresKNN = pd.DataFrame(KNN_scores_para_df)
    return scoresKNN[scoresKNN["score_medio"] == scoresKNN["score_medio"].max()]["n_neighbours"].values[0]


def makeKNNmodel(df, targetColumn, saveModel, returnDict):
    from sklearn.neighbors import KNeighborsClassifier

    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]
    scaler = StandardScaler()

    # Setup train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    optimalNeiValue = findOptimalNeighborsForKNN(X_train, y_train)

    # Setup model and fit
    model = KNeighborsClassifier(n_neighbors=optimalNeiValue)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    if saveModel:
        with open('./pickleModels/KNNmodel', 'wb') as f:
            pickle.dump(model, f)

    # Generate confusion matrix and get matrix
    import graphManager as gm

    mat = gm.make7x7ConsusionMatrix(y_test, y_pred, "KNN", "Predicted", "Actual",
                                    "./images/KNN_confusion_matrix.png", True, True)

    # Get extra data
    TP = mat[1, 1]
    TN = mat[0, 0]
    FP = mat[0, 1]
    FN = mat[1, 0]

    classErrorKNN = (1 - accuracy)
    recallKNN = recall_score(y_test, y_pred)
    specificityKNN = TN / (TN + FP)
    precisionKNN = precision_score(y_test, y_pred)
    f1KNN = f1_score(y_test, y_pred)

    if returnDict:
        return {
            'Modelo': 'KNN',
            'Accuracy': accuracy,
            'Classification Error': classErrorKNN,
            'Recall': recallKNN,
            'Specificity': specificityKNN,
            'Precision': precisionKNN,
            'F1 Score': f1KNN,
        }, y_pred