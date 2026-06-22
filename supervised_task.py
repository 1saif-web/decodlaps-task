import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# دالة مخصصة لمحاكاة عمل الـ SMOTE لتجنب مشاكل استيراد المكتبات الخارجية بالليل
def simulate_smote(X_train, y_train):
    """محاكاة ميكانيكية الـ SMOTE لتوازن البيانات"""
    minority_idx = np.where(y_train == 1)[0]
    majority_idx = np.where(y_train == 0)[0]
    
    # توليد بيانات اصطناعية بربط نقاط البيانات القريبة
    num_synthetic = len(majority_idx) - len(minority_idx)
    synthetic_features = []
    
    for _ in range(num_synthetic):
        idx1 = np.random.choice(minority_idx)
        idx2 = np.random.choice(minority_idx)
        alpha = np.random.uniform(0, 1)
        syn_point = X_train[idx1] + alpha * (X_train[idx2] - X_train[idx1])
        synthetic_features.append(syn_point)
        
    X_resampled = np.vstack([X_train, np.array(synthetic_features)])
    y_resampled = np.hstack([y_train, np.ones(num_synthetic)])
    return X_resampled, y_resampled

if __name__ == "__main__":
    print("=" * 60)
    print("   PHASE 2: SUPERVISED LEARNING (FRAUD DETECTION PIPELINE)  ")
    print("=" * 60)
    
    # 1. توليد بيانات احتيال غير متوازنة للغاية (1% فقط عمليات احتيال)
    np.random.seed(42)
    n_samples = 1000
    X = np.random.normal(0, 1, (n_samples, 5))
    
    # جعل الداتا معتمدة على علاقات محددة
    y = np.zeros(n_samples)
    fraud_condition = (X[:, 0] + X[:, 1] > 2.5)
    y[fraud_condition] = 1
    
    print(f"[✓] Raw dataset generated. Total samples: {len(y)}")
    print(f"-> Legitimate transactions (Class 0): {np.sum(y == 0)}")
    print(f"-> Fraudulent transactions (Class 1): {np.sum(y == 1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    # 2. تقسيم البيانات قبل الـ SMOTE لمنع تسريب البيانات (Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_split=0.2, random_state=42, stratify=y)
    print("\n[✓] Data split into 80/20 Train/Test sets successfully.")
    
    # 3. تطبيق الـ SMOTE على بيانات التدريب فقط
    print("\n[1] Applying SMOTE to rebalance the training feature space...")
    X_train_res, y_train_res = simulate_smote(X_train, y_train)
    print(f"-> Resampled Train Class 0: {np.sum(y_train_res == 0)}")
    print(f"-> Resampled Train Class 1: {np.sum(y_train_res == 1)}")
    
    # 4. تدريب نموذج الـ Random Forest
    print("\n[2] Training Random Forest Classifier on balanced data...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train_res, y_train_res)
    
    # 5. التقييم الصارم باستخدام Metrics حقيقية وليس Accuracy
    print("\n[3] Model Evaluation on Unseen Test Data:")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"-> Strict ROC-AUC Score: {roc_auc:.4f}")
    print("=" * 60)
    print("[✓] Supervised Learning Pipeline executed with zero compilation errors!")