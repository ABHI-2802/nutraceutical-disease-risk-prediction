from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay, brier_score_loss)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from src.pipeline import TARGETS, make_preprocessor, features_for_target
from src.audit import run_audit

ROOT=Path(__file__).resolve().parents[1]
RANDOM_STATE=42

def safe_auc(metric, y, p):
    try: return float(metric(y,p)) if len(np.unique(y))==2 else None
    except Exception: return None

def main():
    df=pd.read_csv(ROOT/'data/raw/disease_risk_dataset.csv')
    run_audit()
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'reports/figures').mkdir(parents=True,exist_ok=True)
    (ROOT/'reports/metrics').mkdir(parents=True,exist_ok=True)
    summary={"dataset_rows":int(len(df)),"random_state":RANDOM_STATE,"targets":{}}
    for target in TARGETS:
        y=pd.to_numeric(df[target],errors='coerce')
        valid=y.notna() & y.isin([0,1])
        X=features_for_target(df.loc[valid].drop(columns=TARGETS,errors='ignore').assign(**{target:y[valid]}), target)
        y=y.loc[valid].astype(int)
        counts=y.value_counts()
        if len(counts)<2 or counts.min()<5:
            summary['targets'][target]={"status":"skipped","reason":"Insufficient examples in one class for stratified evaluation."}; continue
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=RANDOM_STATE,stratify=y)
        cv=StratifiedKFold(n_splits=min(5,int(y_train.value_counts().min())),shuffle=True,random_state=RANDOM_STATE)
        candidates={
            "DummyPrior": DummyClassifier(strategy="prior"),
            "LogisticRegression": LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0,random_state=RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=300,min_samples_leaf=3,max_features="sqrt",class_weight="balanced_subsample",random_state=RANDOM_STATE,n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,max_features="sqrt",class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150,learning_rate=0.06,max_leaf_nodes=15,l2_regularization=1.0,random_state=RANDOM_STATE)
        }
        cv_rows=[]; fitted={}
        for name,est in candidates.items():
            pipe=Pipeline([("prep",make_preprocessor()),("model",est)])
            try:
                scores=cross_validate(pipe,X_train,y_train,cv=cv,scoring={"roc_auc":"roc_auc","avg_precision":"average_precision","f1":"f1"},n_jobs=1,error_score="raise")
                pipe.fit(X_train,y_train)
                cv_rows.append({"model":name,"cv_roc_auc_mean":float(np.mean(scores['test_roc_auc'])),"cv_roc_auc_std":float(np.std(scores['test_roc_auc'])),"cv_pr_auc_mean":float(np.mean(scores['test_avg_precision'])),"cv_f1_mean":float(np.mean(scores['test_f1']))})
                fitted[name]=pipe
            except Exception as e:
                warnings.warn(f"{target}/{name} failed: {e}")
        if not fitted:
            summary['targets'][target]={"status":"failed","reason":"No model trained successfully."}; continue
        cv_df=pd.DataFrame(cv_rows).sort_values('cv_roc_auc_mean',ascending=False)
        cv_df.to_csv(ROOT/f'reports/metrics/{target.lower()}_cv_comparison.csv',index=False)
        best_name=str(cv_df.iloc[0]['model']); best=fitted[best_name]
        proba=best.predict_proba(X_test)[:,1]; pred=(proba>=0.5).astype(int)
        metrics={"selected_model":best_name,"test_rows":int(len(y_test)),"positive_rate_test":float(y_test.mean()),"accuracy":float(accuracy_score(y_test,pred)),"precision":float(precision_score(y_test,pred,zero_division=0)),"recall":float(recall_score(y_test,pred,zero_division=0)),"f1":float(f1_score(y_test,pred,zero_division=0)),"roc_auc":safe_auc(roc_auc_score,y_test,proba),"pr_auc":safe_auc(average_precision_score,y_test,proba),"brier_score":float(brier_score_loss(y_test,proba)),"threshold":0.5,"cv_models":cv_rows,"classification_report":classification_report(y_test,pred,output_dict=True,zero_division=0)}
        joblib.dump(best,ROOT/f'models/{target.lower()}_pipeline.joblib')
        # Permutation importance measured on holdout; use ROC-AUC scoring. It is descriptive, not causal.
        try:
            from sklearn.inspection import permutation_importance
            pi=permutation_importance(best,X_test,y_test,n_repeats=8,random_state=RANDOM_STATE,scoring='roc_auc')
            imp=pd.DataFrame({'feature':X_test.columns,'importance_mean':pi.importances_mean,'importance_std':pi.importances_std}).sort_values('importance_mean',ascending=False)
            imp.to_csv(ROOT/f'reports/metrics/{target.lower()}_permutation_importance.csv',index=False)
            plt.figure(figsize=(8,5)); sns.barplot(data=imp.head(10),x='importance_mean',y='feature'); plt.title(f'{target}: permutation importance (holdout)'); plt.tight_layout(); plt.savefig(ROOT/f'reports/figures/{target.lower()}_importance.png',dpi=140); plt.close()
        except Exception as e: metrics['importance_warning']=str(e)
        cm=confusion_matrix(y_test,pred,labels=[0,1])
        plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Pred 0','Pred 1'],yticklabels=['Actual 0','Actual 1']); plt.title(f'{target}: holdout confusion matrix'); plt.tight_layout(); plt.savefig(ROOT/f'reports/figures/{target.lower()}_confusion_matrix.png',dpi=140); plt.close()
        try:
            fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test,proba,ax=ax); ax.set_title(f'{target}: ROC curve'); fig.tight_layout(); fig.savefig(ROOT/f'reports/figures/{target.lower()}_roc.png',dpi=140); plt.close(fig)
            fig,ax=plt.subplots(figsize=(6,5)); PrecisionRecallDisplay.from_predictions(y_test,proba,ax=ax); ax.set_title(f'{target}: precision-recall curve'); fig.tight_layout(); fig.savefig(ROOT/f'reports/figures/{target.lower()}_pr.png',dpi=140); plt.close(fig)
        except Exception: pass
        summary['targets'][target]={"status":"trained","metrics":metrics,"test_class_counts":{str(k):int(v) for k,v in y_test.value_counts().items()}}
        print(f"\n{target}: selected={best_name}; test ROC-AUC={metrics['roc_auc']:.3f}; PR-AUC={metrics['pr_auc']:.3f}; recall={metrics['recall']:.3f}; F1={metrics['f1']:.3f}")
    (ROOT/'reports/metrics/model_evaluation.json').write_text(json.dumps(summary,indent=2,default=float))
    print(f"\nSaved results to {ROOT/'reports/metrics/model_evaluation.json'}")
if __name__=='__main__': main()
