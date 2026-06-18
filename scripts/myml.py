import os
import pandas as pd
import datetime as dt
import numpy as np
import subprocess
import scipy 
import math
#from statsmodels.stats.descriptivestats import sign_test
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn import metrics, svm, linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.stats import skew, kurtosis
#from statsmodels.stats.descriptivestats import sign_test
import shap
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
from scipy.stats import t

#graphics
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle

def split_and_shuffle(df, features, goal, random_seed=42):
    
    """"splits dataset into X and y and shuffles the participants"""
    X=np.array(df[features])
    y=np.array(df[goal])
    vpn=np.array(df.vpn)
    
    np.random.seed(random_seed)

    #shuffle
    index=np.array(np.random.choice(len(y), size=len(y), replace=False))
    X=X[index]
    vpn=vpn[index]
    y=y[index]
    
    print ('Shape of X:')
    print (X.shape)
    print ('Shape of y:')
    print (y.shape)
    
    return X, y, vpn
    


def roc_graph(y_true, pr, name):
    
    """plots a ROC-Curve for one classifier"""
    
    fpr, tpr, thresholds = roc_curve(y_true, pr) #, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))   
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')   
    plt.savefig('ROC_' +name+'.png')
    plt.show()   
    plt.close()
    
    

def classification_with_nested_cv(X, y, model, space, scaling='no'):
    y_pred= list()
    y_prob= list()
    y_true = list()
    outer_results = list()
    loo = LeaveOneOut()
    #cv_outer= StratifiedKFold(n_splits=10)
    scaler = StandardScaler()

    for train, test in loo.split(X):
        
        if scaling=='yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train=X[train]
            X_test=X[test]
            
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y[train])
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y[test], yhat)
        
        # store the result
        outer_results.append(acc)
        y_pred.append(np.int(yhat))
        y_prob.append(np.float(best_model.predict_proba(X_test)[:,1]))
        y_true.append(np.int(y[test]))
        
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
    
    roc_graph(y_true, y_prob, 'nested_cv')  
    
    print ('Crossvalidierte Ergebnisse on test data')
    print (metrics.classification_report(y_true, y_pred))
       
    print ('Crossvalidierte Ergebnisse on training data')
    model.fit(X, y)          
    y_pred_train = np.array(best_model.predict(X)) #prediction
    print (metrics.classification_report(y_true, y_pred_train))
    
    #results=pd.DataFrame([y_true, y_prob])
    # am Ende jeder classification_* Funktion:
    return pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })
    
def classification_10foldcv(X, y, model, name, scaling='no'):
    y_pred= list()
    y_prob= list()
    y_true = list()
    outer_results = list()
    outer_results_f1 = list()
    cv_outer= StratifiedKFold(n_splits=10)
    scaler = StandardScaler()

    for train, test in cv_outer.split(X, y):
        
        if scaling=='yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train=X[train]
            X_test=X[test]
            
        model.fit(X_train, y[train])         
        yhat = model.predict(X_test)
        
        # evaluate the model
        acc = accuracy_score(y[test], yhat)
        f1 = metrics.f1_score(y[test], yhat)
        
        # store the result
        outer_results.append(acc)
        outer_results_f1.append(f1)
        y_pred.extend(yhat)
        y_prob.extend(model.predict_proba(X_test)[:,1])
        y_true.extend(y[test])
        # report progress
        print('>acc=%.3f' % (acc))
    
    label_counts(y_true, "y_true (echte Labels)")
    label_counts(y_pred, "y_pred (Vorhersagen)")

    #Acc + F1 + std
    print(f"Accuracy: {np.mean(outer_results):.3f} "
      f"(± {np.std(outer_results):.3f})")

    print(f"F1-Score: {np.mean(outer_results_f1):.3f} "
      f"(± {np.std(outer_results_f1):.3f})")

    roc_graph(y_true, y_prob, 'loo')  
    print ('Crossvalidierte Ergebnisse on test data')
    print (metrics.classification_report(y_true, y_pred))
       
    print ('Crossvalidierte Ergebnisse on training data')
    model.fit(X, y) #fitting     
    y_pred_train = np.array(model.predict(X)) #prediction
    print (metrics.classification_report(y_true, y_pred_train))

    # Plotten der Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    labels = ["TSST", "f-TSST"]

    plt.figure(figsize=(4.5, 4.5))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(name)

    plt.tight_layout()
    plt.savefig("ConfusionMatrix_" + name + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    #results=pd.DataFrame([y_true, y_prob])
    return pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })

def classification_10foldcv_with_shap(X, y, model, feature_names=None, scaling='no'):
    y_pred, y_prob, y_true = [], [], []
    outer_results = []
    
    cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scaler = StandardScaler()

    shap_all = []
    Xtest_all = []

    for train_idx, test_idx in cv_outer.split(X, y):
        if scaling == 'yes':
            X_train = scaler.fit_transform(X[train_idx])
            X_test  = scaler.transform(X[test_idx])
        else:
            X_train = X[train_idx]
            X_test  = X[test_idx]

        model.fit(X_train, y[train_idx])
        yhat = model.predict(X_test)

        acc = accuracy_score(y[test_idx], yhat)
        outer_results.append(acc)
        y_pred.extend(yhat)
        y_prob.extend(model.predict_proba(X_test)[:, 1])
        y_true.extend(y[test_idx])

        # --- SHAP: TreeExplainer für XGBoost ---
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test) 

        shap_all.append(shap_values)
        Xtest_all.append(X_test)

        print(f'>acc={acc:.3f}')
        
    
    roc_graph(y_true, y_prob, 'loo')  
    print ('Crossvalidierte Ergebnisse on test data')
    print (metrics.classification_report(y_true, y_pred))
       
    print ('Crossvalidierte Ergebnisse on training data')
    model.fit(X, y) #fitting     
    y_pred_train = np.array(model.predict(X)) #prediction
    print (metrics.classification_report(y_true, y_pred_train))

    # Stapeln über alle Folds
    shap_all = np.vstack(shap_all)
    Xtest_all = np.vstack(Xtest_all)

    # Global importance
    mean_abs_shap = np.mean(np.abs(shap_all), axis=0)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    shap_importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    # Plots (optional)
    plt.figure()
    shap.summary_plot(shap_all, Xtest_all, feature_names=feature_names, show=False)  # beeswarm

    plt.figure()
    shap.summary_plot(shap_all, Xtest_all, feature_names=feature_names, plot_type="bar", show=False)

    #return shap_importance, pd.DataFrame([y_true, y_prob])
    return shap_importance, pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })


def classification_10foldcv_nested(X, y, model, name, space, scaling='no'):
    y_pred= list()
    y_prob= list()
    y_true = list()
    y_pred_base = list()
    outer_results = list()
    outer_results_f1 = list()
    model_fold_scores = []
    baseline_fold_scores = []
    n_train_list = []
    n_test_list = []
    fold_ids = []
    cv_outer= StratifiedKFold(n_splits=10)
    scaler = StandardScaler()

    for fold_id, (train, test) in enumerate(cv_outer.split(X, y), start=1):
        
        if scaling=='yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train=X[train]
            X_test=X[test]
            
        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y[train])
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        #basline prediction (mehrheitsklasse ist die am meisten in train)
        yhat_base = baseline_predict(y_train=y[train], n_samples=len(test))
        
        # evaluate the model
        acc = accuracy_score(y[test], yhat)
        acc_base = accuracy_score(y[test], yhat_base)
        f1 = metrics.f1_score(y[test], yhat)
        
        # speichern für t-test
        model_fold_scores.append(acc)
        baseline_fold_scores.append(acc_base)

        n_train_list.append(len(train))
        n_test_list.append(len(test))
        fold_ids.append(fold_id)
        
        # store the result
        outer_results.append(acc)
        outer_results_f1.append(f1)

        y_pred.extend(yhat)
        y_prob.extend(best_model.predict_proba(X_test)[:,1])
        y_true.extend(y[test])
        y_pred_base.extend(yhat_base)
        # report progress
        print('>acc=%.3f' % (acc))

    label_counts(y_true, "y_true (echte Labels)")
    label_counts(y_pred, "y_pred (Vorhersagen)")
    label_counts(y_pred_base, "y_pred_base (Baseline)")

    print(f"Accuracy: {np.mean(outer_results):.3f} "
      f"(± {np.std(outer_results):.3f})")

    print(f"F1-Score: {np.mean(outer_results_f1):.3f} "
      f"(± {np.std(outer_results_f1):.3f})")
    
    # Plotten der Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    labels = ["TSST", "f-TSST"]

    plt.figure(figsize=(4.5, 4.5))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    #plt.title(name)

    plt.tight_layout()
    plt.savefig("ConfusionMatrix_" + name + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

    table, stat, pvalue = run_mcnemar(
        y_true=np.array(y_true),
        y_pred_model=np.array(y_pred),
        y_pred_base=np.array(y_pred_base)
    )

    print("McNemar (Model vs Baseline)")
    print(f"statistic={stat}, p-value={pvalue}")
        
    roc_graph(y_true, y_prob, 'loo')  
    print ('Crossvalidierte Ergebnisse on test data')
    print (metrics.classification_report(y_true, y_pred))
       
    print ('Crossvalidierte Ergebnisse on training data')
    model.fit(X, y) #fitting     
    #y_pred_train = np.array(best_model.predict(X)) #prediction
    #print (metrics.classification_report(y_true, y_pred_train))
    y_pred_train = np.array(model.predict(X)) #prediction
    print (metrics.classification_report(y, y_pred_train))

    #CSV erzeugen
    fold_scores_filename = f"{name}_classification_kfold_fold_scores.csv"
    df_folds = pd.DataFrame({
        "fold": np.array(fold_ids, dtype=int),
        "metric": "accuracy",
        "model_score": np.array(model_fold_scores, dtype=float),
        "baseline_score": np.array(baseline_fold_scores, dtype=float),
        "n_train": np.array(n_train_list, dtype=int),
        "n_test": np.array(n_test_list, dtype=int),
        "model_name": name,
        "scaling": scaling
    })

    df_folds.to_csv(fold_scores_filename, index=False)

    print(f"Saved CSV (fold scores): {fold_scores_filename}")
        
    #results=pd.DataFrame([y_true, y_prob])
    return pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })


def classification_10foldcv_nested_with_pca(X, y, model, space, scaling='no'):
    '''
    Neu: 
    SatndartScaler wird nun innerhalb von GridSearch gefittet
    PCA wird ebenfalls innerhalb von GridSearch gefittet
    '''
    y_pred= list()
    y_prob= list()
    y_true = list()
    outer_results = list()
    outer_results_f1 = list()

    cv_outer = StratifiedKFold(n_splits=10)
    scaler = StandardScaler()

    #Pipline
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('model', model)
    ])
    

    for train, test in cv_outer.split(X, y):
        
        if scaling=='yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train=X[train]
            X_test=X[test]

        # configure the cross-validation procedure
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        # define search
        search = GridSearchCV(pipeline, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y[train])
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
    

        
        # evaluate the model
        acc = accuracy_score(y[test], yhat)
        f1 = metrics.f1_score(y[test], yhat)
        
        # store the result
        outer_results.append(acc)
        outer_results_f1.append(f1)
        y_pred.extend(yhat)
        y_prob.extend(best_model.predict_proba(X_test)[:,1])
        y_true.extend(y[test])

        print(f'>acc={acc:.3f}, est={result.best_score_:.3f}, cfg={result.best_params_}')

    label_counts(y_true, "y_true (echte Labels)")
    label_counts(y_pred, "y_pred (Vorhersagen)")

    
    print(f"Accuracy: {np.mean(outer_results):.3f} "
      f"(± {np.std(outer_results):.3f})")

    print(f"F1-Score: {np.mean(outer_results_f1):.3f} "
      f"(± {np.std(outer_results_f1):.3f})")
    
        
    roc_graph(y_true, y_prob, 'loo')  
    print ('Crossvalidierte Ergebnisse on test data')
    print (metrics.classification_report(y_true, y_pred))

    #results=pd.DataFrame([y_true, y_prob])
    return pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })
    
def classification_10foldcv_nested_feature_importance_LR(X, y, model, space, scaling='no', audio=None, df=None):

    if hasattr(X, "columns"):
        feature_names = X.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    y_pred = []
    y_prob = []
    y_true = []
    outer_results = []
    outer_results_f1 = []
    model_fold_scores = []
    baseline_fold_scores = []
    n_train_list = []
    n_test_list = []
    fold_ids = []

    cv_outer = StratifiedKFold(n_splits=10)
    scaler = StandardScaler()

    coef_list = []

    for fold_id, (train, test) in enumerate(cv_outer.split(X, y), start=1):

        if scaling == 'yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train = X[train]
            X_test = X[test]

        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

        search = GridSearchCV(
            model,
            space,
            scoring='accuracy',
            cv=cv_inner,
            refit=True
        )

        result = search.fit(X_train, y[train])

        best_model = result.best_estimator_

        coef_list.append(best_model.coef_[0])

        yhat = best_model.predict(X_test)
        yhat_base = baseline_predict(y_train=y[train], n_samples=len(test))

        acc = accuracy_score(y[test], yhat)
        acc_base = accuracy_score(y[test], yhat_base)
        f1 = metrics.f1_score(y[test], yhat)
        
        outer_results.append(acc)
        outer_results_f1.append(f1)
        # speichern für t-test
        model_fold_scores.append(acc)
        baseline_fold_scores.append(acc_base)
        n_train_list.append(len(train))
        n_test_list.append(len(test))
        fold_ids.append(fold_id)

        y_pred.extend(yhat)
        y_prob.extend(best_model.predict_proba(X_test)[:, 1])
        y_true.extend(y[test])

        print(f">acc={acc:.3f}")

    coef_array = np.array(coef_list)

    mean_importance = np.mean(np.abs(coef_array), axis=0)


    if hasattr(X, "columns"):
        feature_names = X.columns
    else:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsImportance": mean_importance
    }).sort_values("MeanAbsImportance", ascending=False)

    importance_df = pd.DataFrame({
        "Variable": df[audio].columns,
        "Importance": abs(mean_importance)
    }).sort_values("Importance", ascending=False)


    print(importance_df.head(5))

    print(f"Accuracy: {np.mean(outer_results):.3f} "
      f"(± {np.std(outer_results):.3f})")

    print(f"F1-Score: {np.mean(outer_results_f1):.3f} "
      f"(± {np.std(outer_results_f1):.3f})")

    roc_graph(y_true, y_prob, 'loo')

    print("\nCrossvalidierte Ergebnisse on test data")
    print(metrics.classification_report(y_true, y_pred))

    fold_scores_filename = f"LR-nested_kfold_fold_scores.csv"
    df_folds = pd.DataFrame({
        "fold": np.array(fold_ids, dtype=int),
        "metric": "accuracy",
        "model_score": np.array(model_fold_scores, dtype=float),
        "baseline_score": np.array(baseline_fold_scores, dtype=float),
        "n_train": np.array(n_train_list, dtype=int),
        "n_test": np.array(n_test_list, dtype=int),
        "scaling": scaling,
        "model_name": type(model).__name__
    })
    df_folds.to_csv(fold_scores_filename, index=False)
    print(f"Saved CSV (fold scores): {fold_scores_filename}")

    return feat_imp, importance_df, pd.DataFrame({
        "y_true": np.array(y_true, dtype=int),
        "y_score": np.array(y_prob, dtype=float),
    })


def calc_features(df, features):
    
    new_features=[]

    operation=['mean', 'max', 'std', 
               lambda x: (np.argmax(x.reset_index(drop=True))), #hier war vorher der Fehler, weil x nicht matrix!
               lambda x: (skew(x)), 
               lambda x: (kurtosis(x))]
    operation_name=['mean', 'max', 'std', 'argmax', 'skew', 'kurtosis']
 
    # for each conversation-part  
    #try:
    for part in set(df.conversation.dropna()):
        for unit in features: #for every feature
            print (unit)
               
            for i, op in enumerate(operation):
                feature_name=unit+'_'+operation_name[i]+'_'+str(part)
                df[feature_name]=df[df.conversation==part].groupby('vpn')[unit].transform(op)                
                new_features.append(feature_name)
                print (feature_name + "_has_been_calculated")
            
    return df, new_features

def calc_features_voice(df, features):
    
    new_features=[]

    operation=['mean', 'max', 'std', 
               lambda x: (np.argmax(x.reset_index(drop=True))), #hier war vorher der Fehler, weil x nicht matrix!
               lambda x: (skew(x)), 
               lambda x: (kurtosis(x))]
    operation_name=['mean', 'max', 'std', 'argmax', 'skew', 'kurtosis']
 
    # for each conversation-part  
    #try:
        
    for unit in features: #for every feature
        print (unit)
        for i, op in enumerate(operation):
            feature_name=unit+'_'+operation_name[i]
            df[feature_name]=df.groupby('vpn')[unit].transform(op)                
            new_features.append(feature_name)
            print (feature_name + "_has_been_calculated")
            
    return df, new_features

    # es entstehen NaNs dadurch, dass nur für jeden Abschnitt die jeweiligen Abschnitte berechnet werden.



def regression(X, y, z, model, name, space, scaling='no', features=None, random_seed=42): 
    
    y_pred = []
    y_true=[]
    y_base=[]
    z_test=[]
    test_index_all = []
    fold_id_all = []

    rf_importance_folds = []
    permutation_importance_folds = []
    folds = []
    model_fold_rmse = []
    base_fold_rmse = []
    n_train_list = []
    n_test_list = []
    
    scaler = StandardScaler()

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)

    for fold_id, (train, test) in enumerate(kfold.split(X), start=1):

        folds.append((train.copy(), test.copy()))

        print(f"Fold {fold_id:02d}: Train size: {len(train)} Test size: {len(test)}")
        print("Train indices head:", train[:5])
        print("Test indices head:", test[:5])
        print("-----")

        if scaling == 'yes':
            X_train = scaler.fit_transform(X[train])
            X_test  = scaler.transform(X[test])
        else:
            X_train = X[train]
            X_test  = X[test]
            
        m = GridSearchCV(model, space, cv=5)
        m.fit(X_train, y[train])

        y_pred=np.append(y_pred, m.best_estimator_.predict(X_test))
        y_true=np.append(y_true, y[test])
        y_base=np.append(y_base, np.repeat(np.mean(y[train]), len(y[test])))
        z_test=np.append(z_test, z[test])

        #für csv
        test_index_all = np.append(test_index_all, test)
        fold_id_all = np.append(fold_id_all, np.repeat(fold_id, len(test)))

        # für wilcoxon signed rank test
        model_fold_rmse.append(math.sqrt(mean_squared_error(y[test], m.best_estimator_.predict(X_test))))
        base_fold_rmse.append(math.sqrt(mean_squared_error(y[test], np.repeat(np.mean(y[train]), len(test)))))
        
        n_train_list.append(len(train))
        n_test_list.append(len(test))

    #Corrected t-test (Nadeau & Bengio)
    model_fold_rmse = np.array(model_fold_rmse)
    base_fold_rmse = np.array(base_fold_rmse)

    # RMSE: kleiner ist besser, daher: baseline - model
    differences = base_fold_rmse - model_fold_rmse

    n = len(differences)
    dfree = n - 1
    n_train = n_train_list[0]
    n_test = n_test_list[0]

    t_stat, p_val = compute_corrected_ttest(differences, dfree, n_train, n_test)

    if features is not None:
        if hasattr(m.best_estimator_, "feature_importances_"):
            rf_importance_folds.append(np.asarray(m.best_estimator_.feature_importances_, dtype=float))
        else:
            pi = permutation_importance(
                m.best_estimator_,
                X_test,
                y[test],
                n_repeats=20,
                random_state=42,
                scoring="neg_mean_absolute_error"   
            )
            permutation_importance_folds.append(np.asarray(pi.importances_mean, dtype=float))

    if features is not None:
        if len(rf_importance_folds) > 0:
            imp = np.vstack(rf_importance_folds)             
            mean_imp = imp.mean(axis=0)
            std_imp = imp.std(axis=0)
            top_idx = np.argsort(mean_imp)[::-1][:5]

            print("Top 5 Features (gemittelt über alle Folds):")
            for rank, j in enumerate(top_idx, start=1):
                print(f"{rank:>2}. {features[j]:<20}  {mean_imp[j]:>10.6f}  ± {std_imp[j]:>10.6f}")
            print()

        if len(permutation_importance_folds) > 0:
            imp = np.vstack(permutation_importance_folds)           
            mean_imp = imp.mean(axis=0)
            std_imp = imp.std(axis=0)
            top_idx = np.argsort(mean_imp)[::-1][:5]

            print("Top 5 Features (Permutation Importance, gemittelt über alle Folds):")
            for rank, j in enumerate(top_idx, start=1):
                print(f"{rank:>2}. {features[j]:<20}  {mean_imp[j]:>10.6f}  ± {std_imp[j]:>10.6f}")
            print()
    
    #corrected t-test
    print("\nCorrected Repeated k-Fold t-Test (Nadeau & Bengio)")
    print("---------------------------------------------------")
    print(f"t-value: {t_stat}")
    print(f"p-value: {p_val}")
    print("Mean difference (baseline - model):", np.mean(differences))
    print()
    #wilcoxon signed rank test
    stat, p = wilcoxon(model_fold_rmse, base_fold_rmse)
    print("\n")
    print("Wilcoxon Signed Rank Test")
    print("---------------------------------")
    print("Statistic:", stat, "pvalue:", p)
    print("\n")
    print ('Crossvalidierte Ergebnisse')
    print("---------------------------------")
    print( 'Root Mean Squared Error:')
    print (math.sqrt(mean_squared_error(y_true, y_pred)))
    print ('Mean_Absolute_Error:')
    errors=np.abs(y_true-y_pred)
    print (np.mean(errors))
    print ('Standard Deviation of the Error:')
    print (np.std(errors))
    print('Spearman Correlation:')
    rho, pval = spearmanr(y_true, y_pred)
    print(f"Spearman r: {rho:.8f}  (p={pval:.8f})")

    print("\n") 
    print("\n")    
    print ('Baseline: Root Mean Squared Error')
    print("---------------------------------")
    print( 'Root Mean Squared Error:')
    print (math.sqrt(mean_squared_error(y_true, y_base)))
    print ('Mean_Absolute_Error:')
    errors=np.abs(y_true-y_base)
    print (np.mean(errors))
    print ('Standard Deviation of the abs Error:')
    print (np.std(errors))

    #CSV 
    fold_scores_filename = f"{name}_kfold_fold_scores.csv"
    df_folds = pd.DataFrame({
        "fold": np.arange(1, len(model_fold_rmse) + 1),
        "metric": "rmse",
        "model_score": np.asarray(model_fold_rmse, dtype=float),
        "baseline_score": np.asarray(base_fold_rmse, dtype=float),
        "n_train": np.asarray(n_train_list, dtype=int),
        "n_test": np.asarray(n_test_list, dtype=int),
        "model_name": name,
        "scaling": scaling,
        "random_seed": random_seed
    })
    df_folds.to_csv(fold_scores_filename, index=False)
    print(f"Saved CSV (fold scores): {fold_scores_filename}")
    
    return y_pred, y_true, z_test 

def loo_regression(X, y, z, model, name,  space, scaling='no', features=None, random_seed=42):

    y_pred = []
    y_true = []
    y_base = []
    z_test = []

    rf_importance_folds = []
    permutation_importance_folds = []
    folds = []

    model_fold_ae = []
    base_fold_ae = []

    scaler = StandardScaler()
    loo = LeaveOneOut()

    for fold_id, (train, test) in enumerate(loo.split(X), start=1):

        # Folds speichern
        folds.append((train.copy(), test.copy()))

        if scaling == 'yes':
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])
        else:
            X_train = X[train]
            X_test = X[test]

        m = GridSearchCV(model, space, cv=5)
        m.fit(X_train, y[train])

        y_pred = np.append(y_pred, m.best_estimator_.predict(X_test))
        y_true = np.append(y_true, y[test])
        y_base = np.append(y_base, np.repeat(np.mean(y[train]), len(y[test])))
        z_test = np.append(z_test, z[test])

        # for Wilcoxon Signed Rank Test
        yhat = m.best_estimator_.predict(X_test)          
        ybase = np.repeat(np.mean(y[train]), len(y[test]))

        model_fold_ae.append(float(np.abs(y[test] - yhat)))
        base_fold_ae.append(float(np.abs(y[test] - ybase)))

        if features is not None:
            if hasattr(m.best_estimator_, "feature_importances_"):
                rf_importance_folds.append(
                    np.asarray(m.best_estimator_.feature_importances_,
                               dtype=float)
                )
            else:
                pi = permutation_importance(
                    m.best_estimator_,
                    X_train,
                    y[train],
                    n_repeats=20,
                    random_state=42,
                    scoring="neg_mean_absolute_error"
                )
                permutation_importance_folds.append(
                    np.asarray(pi.importances_mean, dtype=float)
                )

    if features is not None:

        if len(rf_importance_folds) > 0:
            imp = np.vstack(rf_importance_folds)
            mean_imp = imp.mean(axis=0)
            std_imp = imp.std(axis=0)
            top_idx = np.argsort(mean_imp)[::-1][:5]

            print("Top 5 Features (gemittelt über alle Folds):")
            for rank, j in enumerate(top_idx, start=1):
                print(f"{rank:>2}. {features[j]:<20} "
                      f"{mean_imp[j]:>10.6f} ± {std_imp[j]:>10.6f}")
            print()

        if len(permutation_importance_folds) > 0:
            imp = np.vstack(permutation_importance_folds)
            mean_imp = imp.mean(axis=0)
            std_imp = imp.std(axis=0)
            top_idx = np.argsort(mean_imp)[::-1][:5]

            print("Top 5 Features (Permutation Importance, "
                  "gemittelt über alle Folds):")
            for rank, j in enumerate(top_idx, start=1):
                print(f"{rank:>2}. {features[j]:<20} "
                      f"{mean_imp[j]:>10.6f} ± {std_imp[j]:>10.6f}")
            print()

    stat, p = wilcoxon(model_fold_ae, base_fold_ae)

    print("\n")
    print("Wilcoxon Signed Rank Test")
    print("---------------------------------")
    print("Statistic:", stat, "pvalue:", p)
    print("\n")

    print('Crossvalidierte Ergebnisse')
    print("---------------------------------")

    print('Root Mean Squared Error:')
    print(math.sqrt(mean_squared_error(y_true, y_pred)))

    print('Mean_Absolute_Error:')
    errors = np.abs(y_true - y_pred)
    print(np.mean(errors))

    print('Standard Deviation of the Error:')
    print(np.std(errors))

    print('Spearman Correlation:')
    rho, pval = spearmanr(y_true, y_pred)
    print(f"Spearman r: {rho:.8f} (p={pval:.8f})")

    print("\n")
    print("\n")

    print('Baseline: Root Mean Squared Error')
    print("---------------------------------")

    print('Root Mean Squared Error:')
    print(math.sqrt(mean_squared_error(y_true, y_base)))

    print('Mean_Absolute_Error:')
    errors = np.abs(y_true - y_base)
    print(np.mean(errors))

    print('Standard Deviation of the abs Error:')
    print(np.std(errors))

    #CSV 
    filename = f"{name}_loo_abs_error.csv"
    df = pd.DataFrame({
        "fold": np.arange(1, len(model_fold_ae) + 1),
        "model_abs_error": np.asarray(model_fold_ae, dtype=float),
        "baseline_abs_error": np.asarray(base_fold_ae, dtype=float),
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "y_base": np.asarray(y_base, dtype=float),
        "z_test": np.asarray(z_test, dtype=float),
        "model_name": name,
        "scaling": scaling
    })

    df.to_csv(filename, index=False)
    print(f"Saved CSV: {filename}")

    return y_pred, y_true, z_test

# In[ ]:

def classification(X, y, name, svm_use=False, n_hypercv=5):
    #Initialisierung verschiedener Variablen

    y_pred_rf = []
    y_pred_svm = []
    rf_parameter=[]
    svm_parameter=[]
    
    rf_pr=np.zeros([len(y),2])
    svm_pr=np.zeros([len(y),2])
    y_true=np.zeros(len(y)) 
    test_index=np.zeros(len(y)) 


    tuned_rt_parameters = [{'max_depth':[1, 2, 4, 8, 16, 32, 64],
                           'min_samples_leaf':[1, 2, 4, 8, 16, 32, 64]}]
        
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    tuned_svm_parameters = [{'C':C_range, 'gamma':gamma_range}]  

    #Nested Cross-Validation
    i=0   
    loo = LeaveOneOut()
    X_scaled=X
    #Scaling fuer SVM
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #X = scaler.fit_transform(X)
    

    for train, test in loo.split(X):
               
        #Random-Forest Classifier     
        rf_clf = GridSearchCV(RandomForestClassifier(n_estimators=1000, class_weight='balanced', n_jobs=-1), 
                                                     tuned_rt_parameters, cv=3,  n_jobs=-1)
        rf_clf.fit(X[train], y[train]) #fitting                        
        y_pred_rf = np.append(y_pred_rf, rf_clf.predict(X[test])) #prediction
        probas_rf = rf_clf.predict_proba(X[test]) #class probability
        
        rf_pr[i]=np.array(probas_rf)
        rf_parameter=np.append(rf_parameter, rf_clf.best_params_)   
 
        #SVM
        if svm_use==True:     
            svm_clf = GridSearchCV(svm.SVC(class_weight=None, probability=True), 
                                   tuned_svm_parameters, cv=3, n_jobs=-1, iid=False) 

            svm_clf.fit(X[train], y[train]) #fitting
            y_pred_svm = np.append(y_pred_svm, svm_clf.predict(X_scaled[test])) #prediction
            probas_svm = svm_clf.predict_proba(X_scaled[test]) #class probability
            svm_pr[i]=np.array(probas_svm)
            svm_parameter=np.append(svm_parameter, svm_clf.best_params_)

        
        #Parameter der Classifier auf den Training-Test-Splits
        #print "Features sorted by their score:"
        #print sorted(zip(map(lambda x: round(x, 4), rf_clf.best_estimator_.feature_importances_), names), 
        #         reverse=True)           
            
                   
        y_true[i]=int(y[test])
        test_index[i]=int(test)
        i=i+1   

    #crossvalidated ROC-Curve
    roc_RF_name='RandomForest_'+name
    roc_graph(y_true, rf_pr[:, 1], roc_RF_name)
    
    #Evaluation der crossvalidierten Ergebnisse 
    print ('Crossvalidierte Ergebnisse fuer Random Forest')
    print (metrics.classification_report(y_true,y_pred_rf))
        
    results=pd.DataFrame([y_true, rf_pr, rf_parameter, test_index])
    
    if svm_use==True:
        roc_SVM_name='SVM_'+name
        roc_graph(y_true, svm_pr[:, 1], roc_SVM_name)
        results=pd.DataFrame([y_true, rf_pr, rf_parameter, svm_pr, svm_parameter, test_index])
        print ('Crossvalidierte Ergebnisse fuer Support Vector Machine')
        print (metrics.classification_report(y_true,y_pred_svm))  

    results.to_csv(name +'_Results.csv')
    
    return rf_pr, svm_pr, y_true, test_index, rf_parameter, svm_parameter


# In[ ]:

# Prediction based on AQ

def threshold(X, y, n_hypercv=5):
    #Initialisierung verschiedener Variablen

    y_pred_lr = []  
    lr_pr=np.zeros([len(y),2])
    y_true=np.zeros(len(y)) 
    test_index=np.zeros(len(y)) 


    tuned_lr_parameters = [{'C':[ 0.001, 0.01,  0.1, 1.,  5., 10., 20],  
                          }]
      
    i=0   
    loo = LeaveOneOut()

    for train, test in loo.split(X):
   
        y_true[i]=int(y[test])
        test_index[i]=int(test)

        #LR
        lr_clf = GridSearchCV(linear_model.LogisticRegression(class_weight='balanced'), 
                               tuned_lr_parameters) 
        lr_clf.fit(X[train], y[train])
        y_pred_lr = np.append(y_pred_lr, lr_clf.predict(X[test]))
        probas_lr = lr_clf.predict_proba(X[test])
        lr_pr[i]=np.array(probas_lr)
        
            
        i=i+1   

    roc_graph(y_true, lr_pr[:, 1], 'LogRec')
    
    #Evaluation der crossvalidierten Ergebnisse 
    print ('Crossvalidierte Ergebnisse fuer Logistic Regression')
    print (metrics.classification_report(y_true,y_pred_lr))
        
    results=pd.DataFrame([lr_pr, y_true, test_index])
    results.to_csv('AQ_LogRec_Results.csv')
    
    return lr_pr, y_true, test_index




    
    #ROC-Curve to compare Classifier

def all_roc_graph(y_true, predictions, name): 
   
    plt.figure(figsize=(12, 8))  
    pred=0
    fz=18 #16
    lw=3 #1.5
    sns.set_style("white") 
    
    colors = [
    (0, 128, 128),      
    (0, 0, 128),        
    (200, 128, 0),      
    (128, 0, 0),        
    (31, 119, 180),     
    (255, 127, 14),     
    (44, 160, 44),     
    (214, 39, 40),      
    (148, 103, 189)   
    ]

    linestyle = ['-', '-.', '--', ':', '-.', '-', '--', ':', '-']
    markers = [' ', ' ', ' ', ' ', 'o', 'x', 's', 'd', '^']
    
    plt.plot([-0.01, 1.01], [-0.01, 1.01], color=(0 / 255., 0 / 255., 0 / 255.), 
             linestyle='-',lw=0.5)
    
    for i in range(len(colors)): 
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.) 
        
    for key in predictions:
        
        fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_true, predictions[key], drop_intermediate=False)
        roc_auc_nn = auc(fpr_nn, tpr_nn)

        
        plt.plot(fpr_nn, tpr_nn, label=key + ': AUC = %0.2f' % (roc_auc_nn),
                 lw=lw,
                 linestyle=linestyle[pred],
                 color=colors[pred],
                 marker=markers[pred]
                )    
        pred=pred+1 
        
        
   
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.yticks(fontsize=fz)    
    plt.xticks(fontsize=fz)  
    plt.xlabel('False Positive Rate', fontsize=fz)
    plt.ylabel('True Positive Rate', fontsize=fz)
    plt.title('Receiver Operating Characteristic', fontsize=fz)
    plt.legend(loc='lower right', fontsize=fz-3, frameon=True)   
    plt.savefig('ROC_' + name +'.png', format='png')
    plt.savefig('ROC_' + name +'.pdf', format='pdf')
    plt.show()   
    plt.close()


from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def all_roc_graph_colorblindness(y_true, predictions, name):
    fz = 30
    plt.rcParams.update({
        "axes.titlesize": fz,
        "axes.labelsize": fz,
        "xtick.labelsize": fz,
        "ytick.labelsize": fz,
        "legend.fontsize": fz - 3,
        "pdf.fonttype": 42,  
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    okabe_ito = [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    ]

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    markers = ["o", "s", "^", "D", "x", "P", "v", ">"]

    color_c = cycle(okabe_ito)
    ls_c = cycle(linestyles)
    mk_c = cycle(markers)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="0.6", label="_nolegend_")

    lw = 3

    for label, y_score in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr,
            label=f"{label}: AUC = {roc_auc:.2f}",
            lw=lw,
            linestyle=next(ls_c),
            color=next(color_c),
            marker=next(mk_c),
            markevery=max(len(fpr)//12, 1),
            markersize=7,
            markerfacecolor="none",     
            markeredgewidth=1.8,
        )

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")

    leg = ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    leg.get_frame().set_linewidth(0.8)

    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)

    fig.savefig(f"ROC_{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"ROC_{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
# In[ ]:

# ## Statistical Testing of the different prediction
def evaluate(y_true, predictions, asq, asq_index):
    i=0
    stat_key=[]
    stat_M=[]
    stat_Mp=[]
    stat_R=[]
    stat_Rp=[]
    
    for key in predictions:        
        pred=predictions[key]>0.5
        correcte_pred=(y_true==pred) # Prüft, ob korrekte Vorhersage durch Classifier
        correcte_mayority_pred=(np.ones(len(y_true))==y_true) #Baseline: Mehrheitsklasse zum vergleichen
        

        stat_key.append(key)
        M, p=sign_test((correcte_mayority_pred-correcte_pred), mu0=0) #returns M = (N(+) - N(-))/2 and p
        stat_results_M.append(M)
        stat_results_Mp.append(p)
        
        plt.figure()
        sns.regplot(predictions[key][asq_index], asq[asq_index])
        plt.savefig('Correlation'+ key +'.png')
        plt.show()
        plt.close()
        
        R, p = stats.pearsonr(predictions[key][asq_index], asq[asq_index])
        stat_results_R.append(R)
        stat_results_Rp.append(p)
        

    results=pd.DataFrame([stat_key, stat_M, stat_Mp, stat_R, stat_Rp])
    results.to_csv('RF_Stats.csv') 


# In[ ]:

def pca_on_X(X):
    pca = PCA(n_components=10)
    pca.fit(X)
    X_new=pca.transform(X) 
    return X_new


# Regression

   
def evaluate_reg(y_pred_svr, y_pred_tree, y_true, y_base):

    tree_error=np.abs(y_true-y_pred_tree)
    svr_error=np.abs(y_true-y_pred_svr)
    base_error=np.abs(y_true-y_base)
    
    print ('SVR: ' + str(np.mean(svr_error)))
    print ('Tree: ' + str(np.mean(tree_error)))
    print ('base: ' + str(np.mean(base_error)))

    print ('Tree better then Baseline')
    print (stats.ttest_rel(tree_error, base_error, axis=0, nan_policy='omit'))
    
    print (stats.kruskal(tree_error, base_error))

    print ('SVR better then Baseline')
    print (stats.ttest_rel(svr_error, base_error, axis=0, nan_policy='omit'))
    
    print (stats.kruskal(svr_error, base_error))
    
def reg_cor(x, y, x_label, y_label):
    plt.figure(figsize=(12,8))
    sns.regplot(x=x, y=y)
    plt.yticks(fontsize=14)    
    plt.xticks(fontsize=14)  
    plt.xlabel(x_label, fontsize=16) 
    plt.ylabel(y_label, fontsize=16) 
    plt.savefig(x_label + y_label + '_Regression_Correlation.png')
    plt.show()
    plt.close()


def label_counts(y, name):
    labels, counts = np.unique(y, return_counts=True)
    print(f"\n{name}:")
    for l, c in zip(labels, counts):
        print(f"  Klasse {int(l)} → {int(c)} Samples")


### McNemar Test Statistic
def baseline_predict(y_train, n_samples):
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    return np.full(n_samples, majority_class)

def mcnemar_table(y_true, y_pred_model, y_pred_base):
    y_true = np.asarray(y_true)
    y_pred_model = np.asarray(y_pred_model)
    y_pred_base = np.asarray(y_pred_base)

    a = np.sum((y_pred_model == y_true) & (y_pred_base == y_true))
    b = np.sum((y_pred_model == y_true) & (y_pred_base != y_true))
    c = np.sum((y_pred_model != y_true) & (y_pred_base == y_true))
    d = np.sum((y_pred_model != y_true) & (y_pred_base != y_true))

    return np.array([[a, b],
                     [c, d]])

def run_mcnemar(y_true, y_pred_model, y_pred_base, correction=True, exact=False):
    table = mcnemar_table(y_true, y_pred_model, y_pred_base)
    result = mcnemar(table, exact=exact, correction=correction)
    return table, result.statistic, result.pvalue


#corrected paired t-test
def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated 
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val