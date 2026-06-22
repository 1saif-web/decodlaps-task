import pandas as pd
import numpy as np

def generate_robust_dataset():
    """توليد مجموعة بيانات محاكاة ضخمة لأسعار العقارات لمحاكاة مشروع حقيقي"""
    np.random.seed(42)
    n_samples = 200
    
    # توليد بيانات طبيعية
    sq_ft = np.random.normal(2000, 500, n_samples).astype(int)
    rooms = (sq_ft / 400).astype(int) + np.random.randint(1, 3, n_samples)
    age = np.random.randint(1, 30, n_samples)
    
    # السعر يعتمد على المساحة والغرف والعمر مع وجود بعض العشوائية
    price = (sq_ft * 150) + (rooms * 10000) - (age * 1200) + np.random.normal(50000, 20000, n_samples)
    
    df = pd.DataFrame({
        'Square_Feet': sq_ft,
        'Rooms': rooms,
        'House_Age': age,
        'Price': price.astype(int)
    })
    
    # إدخال قيم شاذة (Outliers) عمدًا لاختبار جودة الـ Pipeline
    df.loc[15, 'Square_Feet'] = 15000  # قصر ضخم عشوائي
    df.loc[42, 'Price'] = 3000000       # سعر مبالغ فيه جداً
    df.loc[88, 'House_Age'] = 120       # عمر بيت مستحيل في المنطقة
    
    # إدخال قيم مفقودة (Missing Values) لاختبار التنظيف
    df.loc[10:14, 'Rooms'] = np.nan
    
    return df

def clean_missing_values(df):
    """التعامل مع القيم المفقودة هندسياً"""
    print("[1] Handling Missing Values...")
    df_cleaned = df.copy()
    # تعويض الغرف المفقودة بالـ Median بناءً على طبيعة البيانات الإحصائية
    rooms_median = df_cleaned['Rooms'].median()
    df_cleaned['Rooms'] = df_cleaned['Rooms'].fillna(rooms_median)
    print(f"-> Filled missing values in 'Rooms' with median: {rooms_median}")
    return df_cleaned

def remove_outliers_iqr(df, column):
    """دالة احترافية لحساب وإزالة القيم الشاذة باستخدام الـ IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_count = len(df)
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed = initial_count - len(df_filtered)
    
    print(f"-> Outlier Analysis for '{column}': Bounds [{lower_bound:.1f}, {upper_bound:.1f}] | Removed: {removed} rows.")
    return df_filtered

if __name__ == "__main__":
    print("=" * 60)
    print("        COMPREHENSIVE EXPLORATORY DATA ANALYSIS PIPELINE       ")
    print("=" * 60)
    
    # 1. تحميل وتوليد الداتا
    df_raw = generate_robust_dataset()
    print(f"Dataset loaded successfully. Shape: {df_raw.shape}")
    print("\n--- Raw Data Statistics Summary ---")
    print(df_raw.describe().T)
    print("-" * 60)
    
    # 2. خطوة تنظيف البيانات
    df_clean = clean_missing_values(df_raw)
    print("-" * 60)
    
    # 3. تصفية القيم الشاذة لجميع الأعمدة المهمة
    print("[2] Detecting and Removing Outliers (IQR Method)...")
    for col in ['Square_Feet', 'Price', 'House_Age']:
        df_clean = remove_outliers_iqr(df_clean, col)
    print(f"Final Cleaned Dataset Shape: {df_clean.shape}")
    print("-" * 60)
    
    # 4. تحليل الارتباط (Correlation Analysis Matrix)
    print("[3] Computing Correlation Matrix (Feature Relationships)...")
    corr_matrix = df_clean.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(4))
    print("\nInsights:")
    target_corr = corr_matrix['Price'].sort_values(ascending=False)
    print(f"-> Strongest predictor for Price is: '{target_corr.index[1]}' with r = {target_corr.values[1]:.4f}")
    print(f"-> Inverse relationship found with: '{target_corr.index[-1]}' with r = {target_corr.values[-1]:.4f}")
    print("=" * 60)
    print("[✓] Phase 3 EDA Pipeline executed with zero compilation errors!")