import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numpy import mean, std
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def RandomForest_trainer(X_train, y_train, X_test, y_test, n= 100, md= 20, split= 5, leaf= 2, state= 42, njobs= -1):
    
    rf_model = RandomForestClassifier(
    n_estimators=n,
    max_depth=md,
    min_samples_split= split,
    min_samples_leaf= leaf,
    random_state= state,
    n_jobs=njobs
)
    trainedModel = rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:,1]
    print("\n" + "="*60)
    print("RESULTADOS DEL MODELO")
    print("="*60)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    return trainedModel
  
  
def Evaluation_model (X_test, y_test, N= 10):
    ''' Implementación de validación cruzada
    Evalua el modelo varias veces'''
    Specificity = make_scorer(recall_score, pos_label=0)


    cross_value_accuracy = RepeatedStratifiedKFold ( n_splits = N, n_repeats=5, random_state=1 ) # Divide los datos en 10 partes, nueve para entrenar 1 para validar. Repite 5 veces con divisiones aleatorias
    n_scores_accuracy = cross_val_score (RandomForestClassifier(), X_test, y_test, scoring = 'accuracy', cv=cross_value_accuracy, 
                                n_jobs =-1, error_score='raise') #
    cross_value_sensitivity = RepeatedStratifiedKFold(n_splits = N, n_repeats=5, random_state=1)
    n_scores_sensititivity = cross_val_score(RandomForestClassifier(), X_test, y_test, scoring='recall', cv=cross_value_sensitivity, 
                                      n_jobs=-1, error_score='raise')
    
    cross_value_specificity = RepeatedStratifiedKFold(n_splits =N, n_repeats=5, random_state=1)
    n_scores_specificity = cross_val_score(RandomForestClassifier(), X_test, y_test, cv=cross_value_specificity, scoring = Specificity,
                                n_jobs=-1, error_score='raise')
    cross_value_precision = RepeatedStratifiedKFold(n_splits =N, n_repeats=5, random_state=1)
    n_scores_precision = cross_val_score(RandomForestClassifier(), X_test, y_test, cv=cross_value_precision, scoring = 'precision' ,
                                n_jobs=-1, error_score='raise')
    cross_value_f1 = RepeatedStratifiedKFold(n_splits =N,  n_repeats=5, random_state=1)
    n_scores_f1 = cross_val_score(RandomForestClassifier(), X_test, y_test, cv=cross_value_f1, scoring = 'f1',
                                n_jobs=-1, error_score='raise')
    #print(n_scores)
    # report performance
    print('Accuracy: %.4f (%.4f)' % (mean(n_scores_accuracy), std(n_scores_accuracy))) #Promedio de las 50 accuracies
    print('sensitivity: %.4f (%.4f)' % (mean(n_scores_sensititivity), std(n_scores_sensititivity)))
    print('specificity: %.4f (%.4f)' % (mean(n_scores_specificity), std(n_scores_specificity)))
    print('precision: %.4f (%.4f)' % (mean(n_scores_precision), std(n_scores_precision)))
    print('f1: %.4f (%.4f)' % (mean(n_scores_f1), std(n_scores_f1)))
    
    return {
      'accuracy': mean(n_scores_accuracy),
      'sensitivity': mean(n_scores_sensititivity),
      'specificity': mean(n_scores_specificity),
      'precision': mean(n_scores_precision),
      'f1_score': mean(n_scores_f1)
    }
    
def Predictions (X_test):
    y_pred_accuracy = rf_model.predict(X_test)
    print(" Accuracy: {:.3f}%".format(accuracy(*classification_report(y_test, y_pred_accuracy)) * 100))

    y_pred_sensitivity = rf_model.predict(X_test)
    print(" Sensitivity: {:.3f}%".format(sensitivity(*classification_report(y_test, y_pred_sensitivity)) * 100))

    y_pred_precision = rf_model.predict(X_test)
    print(" Precision: {:.3f}%".format(precision(*classification_report(y_test, y_pred_precision)) * 100))

    y_pred_f1 = rf_model.predict(X_test)
    print(" F1 Score: {:.3f}%".format(f1_score(*classification_report(y_test, y_pred_f1)) * 100))
    return {
        'Accuracy': y_pred_accuracy,
        'Sensitivity': y_pred_sensitivity,
        'Precision': y_pred_precision,
        'F1': y_pred_f1
    }