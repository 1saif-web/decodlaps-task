import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    print("=" * 60)
    print("   PHASE 1: UNSUPERVISED LEARNING (K-MEANS & PCA) PIPELINE  ")
    print("=" * 60)
    
    # 1. توليد بيانات وهمية متعددة الأبعاد (Simulating high-dimensional customer data)
    np.random.seed(42)
    n_samples = 150
    
    # بيانات زبائن: الدخل السنوي، معدل الصرف، العمر، عدد الزيارات، تقييم الولاء
    data = {
        'Annual_Income': np.random.normal(60000, 15000, n_samples),
        'Spending_Score': np.random.uniform(1, 100, n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Total_Visits': np.random.poisson(5, n_samples),
        'Loyalty_Rating': np.random.uniform(1, 5, n_samples)
    }
    df = pd.DataFrame(data)
    print(f"[✓] High-dimensional dataset generated. Shape: {df.shape}")
    
    # 2. قياس وتجهيز البيانات (Feature Scaling) لأن الـ K-Means حساس للمسافات
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df)
    print("[✓] Features successfully standardized using StandardScaler.")
    
    # 3. تقليل الأبعاد باستخدام الـ PCA (Dimensionality Reduction to 2 Components)
    print("\n[1] Executing Principal Component Analysis (PCA)...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    
    # نسبة التباين المشروح من الأبعاد الجديدة
    explained_variance = pca.explained_variance_ratio_
    print(f"-> Component 1 explains: {explained_variance[0]:.4f}")
    print(f"-> Component 2 explains: {explained_variance[1]:.4f}")
    print(f"-> Total Retained Variance: {sum(explained_variance):.4f}")
    
    # 4. محاكاة طريقة الكوع (Simulating Elbow Method to find optimal K)
    print("\n[2] Computing Within-Cluster Sum of Squares (WCSS) for Elbow Method...")
    wcss = []
    for i in range(1, 6):
        kmeans_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans_temp.fit(scaled_features)
        wcss.append(kmeans_temp.inertia_)
        print(f"-> For K = {i} | WCSS (Inertia) = {kmeans_temp.inertia_:.2f}")
        
    # 5. تطبيق الـ K-Means النهائي بناءً على الـ Optimal K (نفترض K=3)
    print("\n[3] Training Final K-Means Model (Optimal K = 3)...")
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_transform(scaled_features)
    
    # إضافة التصنيفات للداتا الأصلية
    df['Cluster_ID'] = kmeans.labels_
    
    print("\n--- Cluster Profile Summary (Mean Values) ---")
    print(df.groupby('Cluster_ID').mean().round(2))
    
    print("=" * 60)
    print("[✓] Unsupervised Learning Pipeline executed with zero compilation errors!")