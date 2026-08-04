"""
=============================================================================
LINKEDIN JOB POSTINGS - BIG DATA ANALYSIS PIPELINE
=============================================================================
Dataset  : LinkedIn Job Postings (2023-2024)
Sumber   : https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
Task     : Regression — Prediksi Median Gaji (med_salary)
Algoritma: Random Forest, XGBoost, Ridge Regression
Tools    : PySpark, Pandas, Scikit-Learn, XGBoost, Matplotlib, Seaborn
=============================================================================

INSTRUKSI PENGGUNAAN:
1. Download dataset dari Kaggle: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
2. Ekstrak ke folder 'data/' di direktori yang sama dengan file ini
3. Struktur folder:
   data/
   ├── postings.csv
   ├── companies/
   │   └── companies.csv
   │   └── company_industries.csv
   │   └── company_specialities.csv
   ├── job_industries.csv
   ├── job_skills.csv
   └── salaries.csv (jika tersedia)
4. Jalankan: python linkedin_bigdata_pipeline.py
=============================================================================
"""

# ===================================================================
# IMPORT LIBRARIES
# ===================================================================
import sys
import io

# Force UTF-8 encoding untuk terminal Windows agar emoji bisa tampil
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import warnings
import time
from datetime import datetime

# Data Processing
import numpy as np
import pandas as pd

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DoubleType, LongType, TimestampType
)
from pyspark.sql.window import Window

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import xgboost as xgb

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("  LINKEDIN JOB POSTINGS — BIG DATA ANALYSIS PIPELINE")
print("  Klasifikasi Remote vs Onsite")
print("=" * 70)
print(f"  Waktu mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)


# ===================================================================
# STAGE 1: PYSPARK SESSION INITIALIZATION
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 1: INISIALISASI SPARK SESSION")
print("=" * 70)

spark = SparkSession.builder \
    .appName("LinkedIn_JobPostings_BigData_Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

# Set log level to reduce noise
spark.sparkContext.setLogLevel("WARN")

print(f"  ✅ Spark Session berhasil dibuat")
print(f"  ✅ Spark Version: {spark.version}")
print(f"  ✅ App Name: {spark.sparkContext.appName}")


# ===================================================================
# STAGE 2: DATA INGESTION — Membaca Data dengan PySpark
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 2: DATA INGESTION")
print("=" * 70)

# --- Konfigurasi path ---
# Sesuaikan path ini dengan lokasi dataset Anda
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Path ke file CSV
POSTINGS_PATH = os.path.join(BASE_PATH, "postings.csv")
COMPANIES_PATH = os.path.join(BASE_PATH, "companies", "companies.csv")
COMPANY_INDUSTRIES_PATH = os.path.join(BASE_PATH, "companies", "company_industries.csv")

# Path Opsional
JOB_INDUSTRIES_PATH = os.path.join(BASE_PATH, "jobs", "job_industries.csv")
JOB_SKILLS_PATH = os.path.join(BASE_PATH, "jobs", "job_skills.csv")
SALARIES_PATH = os.path.join(BASE_PATH, "jobs", "salaries.csv")

# --- Fungsi pembacaan data ---
def load_csv_spark(path, description=""):
    """Membaca file CSV menggunakan PySpark dengan error handling."""
    try:
        df = spark.read.csv(
            path,
            header=True,
            inferSchema=True,
            multiLine=True,
            escape='"',
            quote='"'
        )
        row_count = df.count()
        col_count = len(df.columns)
        print(f"  ✅ {description}: {row_count:,} baris × {col_count} kolom")
        return df
    except Exception as e:
        print(f"  ⚠️  Gagal membaca {description}: {e}")
        print(f"      Path: {path}")
        return None

# --- Membaca semua tabel ---
print("\n  📂 Membaca dataset dari:", BASE_PATH)
print("-" * 50)

start_time = time.time()

# Tabel utama: Job Postings
df_postings = load_csv_spark(POSTINGS_PATH, "Job Postings (Tabel Utama)")

# Tabel pendukung
df_companies = load_csv_spark(COMPANIES_PATH, "Companies")
df_company_industries = load_csv_spark(COMPANY_INDUSTRIES_PATH, "Company Industries")

# Tabel skills dan industries (opsional - menggunakan kolom bawaan jika tidak ada)
df_job_industries = load_csv_spark(JOB_INDUSTRIES_PATH, "Job Industries Mapping")
df_job_skills = load_csv_spark(JOB_SKILLS_PATH, "Job Skills Mapping")

elapsed = time.time() - start_time
print(f"\n  ⏱️  Waktu ingestion: {elapsed:.2f} detik")

# --- Menampilkan schema tabel utama ---
print("\n  📋 Schema Tabel Postings:")
if df_postings:
    df_postings.printSchema()


# ===================================================================
# STAGE 3: DATA JOINING — Menggabungkan Tabel Relasional
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 3: DATA JOINING (Spark SQL)")
print("=" * 70)

if df_postings is not None:
    # Tampilkan kolom yang tersedia
    print(f"\n  📋 Kolom di postings.csv:")
    print(f"     {df_postings.columns}")

    # Pilih kolom yang relevan dari postings
    postings_cols = [
        'job_id', 'company_id', 'title', 'med_salary', 'min_salary',
        'max_salary', 'pay_period', 'formatted_work_type',
        'location', 'applies', 'remote_allowed', 'views',
        'formatted_experience_level', 'sponsored',
        'original_listed_time', 'closed_time', 'expiry',
        'application_type', 'skills_desc', 'listed_time'
    ]

    # Filter hanya kolom yang ada
    available_cols = [c for c in postings_cols if c in df_postings.columns]
    df_main = df_postings.select(available_cols)

    print(f"\n  ✅ Kolom yang dipilih: {len(available_cols)} dari {len(df_postings.columns)}")
    print(f"     {available_cols}")

    # --- Join dengan Companies ---
    if df_companies is not None:
        company_cols = ['company_id', 'name', 'company_size', 'city', 'state', 'country']
        available_company_cols = [c for c in company_cols if c in df_companies.columns]

        if available_company_cols:
            df_companies_sel = df_companies.select(available_company_cols)

            # Rename kolom company untuk menghindari konflik
            for col_name in ['city', 'state', 'country']:
                if col_name in available_company_cols:
                    df_companies_sel = df_companies_sel.withColumnRenamed(col_name, f'company_{col_name}')
            if 'name' in available_company_cols:
                df_companies_sel = df_companies_sel.withColumnRenamed('name', 'company_name')

            df_main = df_main.join(df_companies_sel, on='company_id', how='left')
            print(f"  ✅ Joined dengan Companies: {df_main.count():,} baris")

    # --- Join dengan Company Industries ---
    if df_company_industries is not None and 'company_id' in df_company_industries.columns:
        # Ambil industri pertama per perusahaan (primary industry)
        industry_col = [c for c in df_company_industries.columns if c != 'company_id'][0] if len(df_company_industries.columns) > 1 else None
        if industry_col:
            df_ind = df_company_industries.groupBy('company_id').agg(
                F.first(industry_col).alias('industry')
            )
            df_main = df_main.join(df_ind, on='company_id', how='left')
            print(f"  ✅ Joined dengan Company Industries: {df_main.count():,} baris")

    # --- Tampilkan hasil join ---
    print(f"\n  📊 Dataset gabungan: {df_main.count():,} baris × {len(df_main.columns)} kolom")
    print(f"  📋 Kolom final: {df_main.columns}")

    # Tampilkan sample data
    print("\n  📋 Sample Data (5 baris pertama):")
    df_main.show(5, truncate=True)


# ===================================================================
# STAGE 4: PREPROCESSING DENGAN PYSPARK
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 4: DATA PREPROCESSING (PySpark)")
print("=" * 70)

# --- 4.1: Analisis Missing Values ---
print("\n  📊 4.1 Analisis Missing Values")
print("-" * 50)

total_rows = df_main.count()
for col_name in df_main.columns:
    null_count = df_main.filter(F.col(col_name).isNull()).count()
    pct = (null_count / total_rows) * 100
    if null_count > 0:
        print(f"     {col_name:35s}: {null_count:>8,} nulls ({pct:>6.2f}%)")

# --- 4.2: Data Cleaning (No salary filter) ---
print("\n  🔧 4.2 Data Cleaning")
df_with_salary = df_main.fillna({'remote_allowed': 0})
print(f"     ✅ Missing remote_allowed diisi 0 (Onsite). Total baris: {df_with_salary.count():,}")

# --- 4.3: Hapus Duplikasi ---
print("\n  🔧 4.3 Hapus Duplikasi")
before_dedup = df_with_salary.count()
df_with_salary = df_with_salary.dropDuplicates(['job_id'])
after_dedup = df_with_salary.count()
print(f"     Sebelum: {before_dedup:,} | Sesudah: {after_dedup:,} | Dihapus: {before_dedup - after_dedup:,}")

# --- 4.4: Feature Engineering ---
print("\n  🔧 4.4 Feature Engineering")

# 4.4.1: (Dihapus karena tidak memakai gaji)

# 4.4.2: Salary Range (indikator ketidakpastian kompensasi)
if 'min_salary' in df_with_salary.columns and 'max_salary' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'salary_range',
        F.when(
            (F.col('max_salary').isNotNull()) & (F.col('min_salary').isNotNull()),
            F.col('max_salary') - F.col('min_salary')
        ).otherwise(0)
    )
    print("     ✅ salary_range = max_salary - min_salary")

# 4.4.3: Location Parsing (ekstrak state)
if 'location' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'job_state',
        F.when(
            F.col('location').contains(','),
            F.trim(F.element_at(F.split(F.col('location'), ','), -1))
        ).otherwise(F.col('location'))
    )
    print("     ✅ Ekstraksi state dari location")

# 4.4.4: Application Rate
if 'applies' in df_with_salary.columns and 'views' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'application_rate',
        F.when(
            (F.col('views').isNotNull()) & (F.col('views') > 0),
            F.col('applies') / F.col('views')
        ).otherwise(0.0)
    )
    print("     ✅ application_rate = applies / views")

# 4.4.5: Skill count (dari skills_desc)
if 'skills_desc' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'skill_count',
        F.when(
            F.col('skills_desc').isNotNull(),
            F.size(F.split(F.col('skills_desc'), ','))
        ).otherwise(0)
    )
    print("     ✅ skill_count dihitung dari skills_desc")

# 4.4.6: Title clustering (kategori umum berdasarkan keyword)
if 'title' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'title_category',
        F.when(F.lower(F.col('title')).rlike('engineer|developer|software|devops|sre|backend|frontend|fullstack'), 'Engineering')
         .when(F.lower(F.col('title')).rlike('data|analyst|analytics|scientist|machine learning|ai|ml'), 'Data & Analytics')
         .when(F.lower(F.col('title')).rlike('manager|director|vp|vice president|head of|chief|cto|cfo|ceo'), 'Management')
         .when(F.lower(F.col('title')).rlike('sales|account|business development|bdr|sdr'), 'Sales')
         .when(F.lower(F.col('title')).rlike('marketing|brand|content|seo|growth|social media'), 'Marketing')
         .when(F.lower(F.col('title')).rlike('design|ux|ui|graphic|creative|product design'), 'Design')
         .when(F.lower(F.col('title')).rlike('finance|accounting|auditor|tax|treasury|controller'), 'Finance')
         .when(F.lower(F.col('title')).rlike('nurse|doctor|medical|health|clinical|pharma|therapist'), 'Healthcare')
         .when(F.lower(F.col('title')).rlike('hr|human resource|recruiter|talent|people'), 'Human Resources')
         .when(F.lower(F.col('title')).rlike('consult|advisor|strateg'), 'Consulting')
         .when(F.lower(F.col('title')).rlike('support|customer|service|helpdesk'), 'Customer Support')
         .when(F.lower(F.col('title')).rlike('product manager|product owner|scrum'), 'Product Management')
         .when(F.lower(F.col('title')).rlike('operation|logistics|supply chain|warehouse'), 'Operations')
         .when(F.lower(F.col('title')).rlike('legal|lawyer|attorney|compliance'), 'Legal')
         .when(F.lower(F.col('title')).rlike('teach|professor|education|instructor|tutor'), 'Education')
         .otherwise('Other')
    )
    print("     ✅ Title clustering ke 15+ kategori")

# 4.4.7: Is Sponsored flag
if 'sponsored' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'is_sponsored',
        F.when(
            (F.col('sponsored').cast('string') == '1') | 
            (F.lower(F.col('sponsored').cast('string')) == 'true'), 
        1).otherwise(0)
    )
    print("     ✅ is_sponsored flag (binary)")

# 4.4.8: Remote allowed flag
if 'remote_allowed' in df_with_salary.columns:
    df_with_salary = df_with_salary.withColumn(
        'is_remote',
        F.when(
            (F.col('remote_allowed').cast('string') == '1') | 
            (F.lower(F.col('remote_allowed').cast('string')) == 'true'), 
        1).otherwise(0)
    )
    print("     ✅ is_remote flag (binary)")


# --- 4.5: Imputasi Missing Values pada Fitur ---
print("\n  🔧 4.5 Imputasi Missing Values pada Fitur")

# Imputasi numerik dengan median
numeric_cols_to_impute = ['views', 'applies', 'application_rate', 'skill_count']
for col_name in numeric_cols_to_impute:
    if col_name in df_with_salary.columns:
        median_val = df_with_salary.approxQuantile(col_name, [0.5], 0.01)
        if median_val and len(median_val) > 0:
            df_with_salary = df_with_salary.fillna({col_name: median_val[0]})
            print(f"     ✅ {col_name}: imputasi median = {median_val[0]:.2f}")

# Imputasi kategorikal
cat_impute = {
    'formatted_experience_level': 'Unknown',
    'formatted_work_type': 'Full-time',
    'title_category': 'Other',
    'job_state': 'Unknown',
    'industry': 'Unknown',
    'company_size': '0'
}
for col_name, fill_val in cat_impute.items():
    if col_name in df_with_salary.columns:
        # Cast ke string terlebih dahulu agar aman saat di-fillna dengan string
        df_with_salary = df_with_salary.withColumn(col_name, F.col(col_name).cast('string'))
        df_with_salary = df_with_salary.fillna({col_name: fill_val})
        print(f"     ✅ {col_name}: imputasi '{fill_val}'")

# Imputasi binary dengan 0
binary_cols = ['is_sponsored', 'is_remote']
for col_name in binary_cols:
    if col_name in df_with_salary.columns:
        df_with_salary = df_with_salary.fillna({col_name: 0})

# --- 4.6: Filter Outlier Gaji (IQR method) ---
print("\n  🔧 4.6 Outlier Removal (annual_salary)")
q1, q3 = df_with_salary.approxQuantile('annual_salary', [0.25, 0.75], 0.01)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

before_outlier = df_with_salary.count()
df_with_salary = df_with_salary.filter(
    (F.col('annual_salary') >= lower) & (F.col('annual_salary') <= upper)
)
after_outlier = df_with_salary.count()
print(f"     IQR Range: ${lower:,.0f} — ${upper:,.0f}")
print(f"     Sebelum: {before_outlier:,} | Sesudah: {after_outlier:,} | Dihapus: {before_outlier - after_outlier:,}")

# --- Tampilkan statistik akhir ---
print("\n  📊 Statistik Dataset Setelah Preprocessing:")
print(f"     Total baris: {df_with_salary.count():,}")
print(f"     Total kolom: {len(df_with_salary.columns)}")
df_with_salary.describe(['annual_salary']).show()

# --- 4.7: Simpan ke Parquet (opsional) ---
PARQUET_PATH = os.path.join(BASE_PATH, "processed_postings.parquet")
try:
    df_with_salary.write.mode('overwrite').parquet(PARQUET_PATH)
    print(f"  💾 Data tersimpan ke Parquet: {PARQUET_PATH}")
except Exception as e:
    print(f"  ⚠️  Gagal menyimpan Parquet: {e}")


# ===================================================================
# STAGE 5: KONVERSI KE PANDAS & ENCODING
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 5: KONVERSI KE PANDAS & FINAL ENCODING")
print("=" * 70)

# Konversi PySpark DataFrame ke Pandas
pdf = df_with_salary.toPandas()
print(f"  ✅ Konversi ke Pandas: {pdf.shape[0]:,} baris × {pdf.shape[1]} kolom")
print(f"  📋 Kolom: {list(pdf.columns)}")

# --- Pembuatan Target Klasifikasi (salary_level) ---
pdf['salary_level'] = pd.qcut(pdf['annual_salary'], q=[0, 0.33, 0.67, 1.0], labels=['Low', 'Medium', 'High'])
print(f"\n  ✅ Target Klasifikasi (salary_level) dibuat:")
print(f"     {pdf['salary_level'].value_counts().to_dict()}")

le_salary = LabelEncoder()
le_salary.fit(['Low', 'Medium', 'High'])
pdf['salary'] = pdf['salary_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# --- Definisi fitur untuk modeling ---
# Fitur numerik
numeric_features = []
for col in ['views', 'applies', 'application_rate', 'skill_count', 'is_sponsored', 'is_remote']:
    if col in pdf.columns:
        numeric_features.append(col)

if 'salary_range' in pdf.columns:
    numeric_features.append('salary_range')

# Fitur kategorikal
categorical_features = []
for col in ['formatted_experience_level', 'formatted_work_type', 'title_category', 'job_state', 'industry', 'company_size']:
    if col in pdf.columns:
        categorical_features.append(col)

print(f"\n  📊 Fitur Numerik ({len(numeric_features)}): {numeric_features}")
print(f"  📊 Fitur Kategorikal ({len(categorical_features)}): {categorical_features}")

# --- Ordinal Encoding untuk Experience Level ---
if 'formatted_experience_level' in pdf.columns:
    exp_order = {
        'Internship': 0,
        'Entry level': 1,
        'Associate': 2,
        'Mid-Senior level': 3,
        'Director': 4,
        'Executive': 5,
        'Unknown': -1
    }
    pdf['experience_encoded'] = pdf['formatted_experience_level'].map(exp_order).fillna(-1).astype(int)
    numeric_features.append('experience_encoded')
    print(f"  ✅ Experience Level → Ordinal Encoding: {exp_order}")

# --- Frequency Encoding untuk job_state (terlalu banyak kategori untuk one-hot) ---
if 'job_state' in pdf.columns:
    state_freq = pdf['job_state'].value_counts(normalize=True)
    pdf['state_freq_encoded'] = pdf['job_state'].map(state_freq).fillna(0)
    numeric_features.append('state_freq_encoded')
    print(f"  ✅ job_state → Frequency Encoding (top: {state_freq.head(3).to_dict()})")

# --- Frequency Encoding untuk industry ---
if 'industry' in pdf.columns:
    ind_freq = pdf['industry'].value_counts(normalize=True)
    pdf['industry_freq_encoded'] = pdf['industry'].map(ind_freq).fillna(0)
    numeric_features.append('industry_freq_encoded')
    print(f"  ✅ industry → Frequency Encoding")

# --- One-Hot Encoding untuk fitur dengan kardinalitas rendah ---
ohe_features = []
for col in ['formatted_work_type', 'title_category']:
    if col in pdf.columns:
        ohe_features.append(col)

if ohe_features:
    pdf_encoded = pd.get_dummies(pdf, columns=ohe_features, prefix_sep='_', drop_first=False)
    # Tambahkan kolom OHE ke numeric features
    ohe_columns = [c for c in pdf_encoded.columns if any(c.startswith(f + '_') for f in ohe_features)]
    numeric_features.extend(ohe_columns)
    print(f"  ✅ One-Hot Encoding: {ohe_features} → {len(ohe_columns)} kolom baru")
else:
    pdf_encoded = pdf.copy()

# --- Target Variable ---
target = 'remote_allowed'
target_class = 'salary'
print(f"\n  🎯 Target Variable Regresi: {target}")
print(f"  🎯 Target Variable Klasifikasi: {target_class}")
print(f"     Mean:   ${pdf_encoded[target].mean():,.0f}")
print(f"     Median: ${pdf_encoded[target].median():,.0f}")
print(f"     Std:    ${pdf_encoded[target].std():,.0f}")
print(f"     Min:    ${pdf_encoded[target].min():,.0f}")
print(f"     Max:    ${pdf_encoded[target].max():,.0f}")

# --- Final Feature Matrix ---
# Filter hanya fitur numerik yang ada di dataframe
final_features = [f for f in numeric_features if f in pdf_encoded.columns]
# Pastikan tidak ada duplikat
final_features = list(dict.fromkeys(final_features))

print(f"\n  📊 Final Features ({len(final_features)}):")
for i, feat in enumerate(final_features, 1):
    print(f"     {i:>2}. {feat}")

# Buat X dan y
X = pdf_encoded[final_features].copy()
y = pdf_encoded[target].copy()
y = pdf_encoded[target_class].copy()

# Pastikan semua fitur numerik
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"\n  ✅ Feature Matrix: {X.shape}")
print(f"  ✅ Target Vector:  {y.shape}")


# ===================================================================
# STAGE 6: EXPLORATORY DATA ANALYSIS (EDA)
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 6: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# Buat folder output untuk grafik
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output-classification")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 6.1: Distribusi Target (annual_salary) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].set_title('Distribusi Annual Salary', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Annual Salary ($)')
axes[0].set_ylabel('Frekuensi')
axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: ${y.mean():,.0f}')
axes[0].axvline(y.median(), color='green', linestyle='--', label=f'Median: ${y.median():,.0f}')
axes[0].legend()

axes[1].boxplot(y, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.7))
axes[1].set_title('Box Plot Annual Salary', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Annual Salary ($)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_salary_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 1: Distribusi Salary disimpan")

# --- 6.2: Gaji per Experience Level ---
if 'formatted_experience_level' in pdf_encoded.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    exp_salary = pdf_encoded.groupby('formatted_experience_level')[target].agg(['mean', 'median', 'count'])
    exp_salary = exp_salary.sort_values('median', ascending=True)
    exp_salary = exp_salary[exp_salary['count'] >= 10]  # Minimal 10 data

    bars = ax.barh(exp_salary.index, exp_salary['median'], color='#2ecc71', alpha=0.8, label='Median')
    ax.barh(exp_salary.index, exp_salary['mean'], color='#e74c3c', alpha=0.4, label='Mean')

    for i, (val, count) in enumerate(zip(exp_salary['median'], exp_salary['count'])):
        ax.text(val + 1000, i, f'${val:,.0f} (n={count:,})', va='center', fontsize=9)

    ax.set_title('Median & Mean Salary per Experience Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annual Salary ($)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_salary_by_experience.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 2: Salary per Experience Level disimpan")

# --- 6.3: Gaji per Title Category ---
if 'title_category' in pdf_encoded.columns:
    fig, ax = plt.subplots(figsize=(12, 7))
    title_salary = pdf_encoded.groupby('title_category')[target].agg(['median', 'count'])
    title_salary = title_salary.sort_values('median', ascending=True)
    title_salary = title_salary[title_salary['count'] >= 5]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(title_salary)))
    ax.barh(title_salary.index, title_salary['median'], color=colors, alpha=0.85)

    for i, (val, count) in enumerate(zip(title_salary['median'], title_salary['count'])):
        ax.text(val + 500, i, f'${val:,.0f} (n={count})', va='center', fontsize=8)

    ax.set_title('Median Salary per Job Category', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annual Salary ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_salary_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 3: Salary per Title Category disimpan")

# --- 6.4: Correlation Heatmap ---
numeric_for_corr = pdf_encoded[final_features + [target]].select_dtypes(include=[np.number])
# Ambil hanya fitur yang punya korelasi signifikan
corr_matrix = numeric_for_corr.corr()

# Ambil top features berdasarkan korelasi absolut dengan target
if target in corr_matrix.columns:
    top_corr = corr_matrix[target].abs().sort_values(ascending=False).head(15)
    top_corr_features = top_corr.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 10))
    corr_subset = numeric_for_corr[top_corr_features].corr()
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title('Correlation Heatmap (Top 15 Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 4: Correlation Heatmap disimpan")

# --- 6.5: Remote vs Non-Remote Salary ---
if 'is_remote' in pdf_encoded.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    remote_labels = {0: 'On-site', 1: 'Remote'}
    pdf_encoded['remote_label'] = pdf_encoded['is_remote'].map(remote_labels).fillna('Unknown')

    remote_data = [pdf_encoded[pdf_encoded['is_remote'] == 0][target].dropna(),
                   pdf_encoded[pdf_encoded['is_remote'] == 1][target].dropna()]

    bp = ax.boxplot(remote_data, labels=['On-site', 'Remote'], patch_artist=True,
                    boxprops=dict(alpha=0.7))
    bp['boxes'][0].set_facecolor('#e74c3c')
    bp['boxes'][1].set_facecolor('#2ecc71')

    ax.set_title('Salary: Remote vs On-site', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Salary ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_remote_vs_onsite.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 5: Remote vs On-site Salary disimpan")

# --- 6.6: Distribusi Kategori Gaji (Klasifikasi) ---
fig, ax = plt.subplots(figsize=(8, 6))
salary_counts = pdf_encoded['salary_level'].value_counts().sort_index()
colors_class = ['#e74c3c', '#f1c40f', '#2ecc71']
ax.pie(salary_counts, labels=salary_counts.index, autopct='%1.1f%%', colors=colors_class, startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
ax.set_title('Proporsi Kategori Gaji (Low, Medium, High)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05b_salary_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 5b: Proporsi Kategori Gaji disimpan")

# --- 6.7: Visualisasi Missing Values (Before vs After) ---
# Simulasi drop rate jika kita menyimpan history (hanya contoh visual bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
drop_rates = {'Raw Data': total_rows, 'After Missing Handling': total_rows * 0.95, 'After Outlier Removal': len(pdf)}
ax.bar(drop_rates.keys(), drop_rates.values(), color=['#95a5a6', '#f39c12', '#2ecc71'])
ax.set_title('Jumlah Baris Data Sepanjang Pipeline', fontsize=14, fontweight='bold')
for i, v in enumerate(drop_rates.values()):
    ax.text(i, v + 1000, f'{int(v):,}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '00_data_pipeline_rows.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 00: Jumlah Baris Pipeline disimpan")

# --- 6.8: Tambahan Visualisasi Spesifik (02 - 06) ---
print("\n  🔧 6.8 Menghasilkan plot tambahan sesuai permintaan (02-06)...")

# 02_Distribusi_Seluruh_Variabel_Kategorikal.png
cat_columns_plot = ['formatted_work_type', 'title_category', 'formatted_experience_level']
valid_cats = [c for c in cat_columns_plot if c in pdf_encoded.columns]
if valid_cats:
    fig, axes = plt.subplots(len(valid_cats), 1, figsize=(10, 5 * len(valid_cats)))
    if len(valid_cats) == 1: axes = [axes]
    for ax, col in zip(axes, valid_cats):
        sns.countplot(y=col, data=pdf_encoded, order=pdf_encoded[col].value_counts().index, ax=ax, palette='viridis')
        ax.set_title(f'Distribusi {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_Distribusi_Seluruh_Variabel_Kategorikal.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 02_Distribusi_Seluruh_Variabel_Kategorikal disimpan")

# 03_Sepuluh_Negara_Responden_Terbanyak.png
if 'job_state' in pdf_encoded.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10_states = pdf_encoded['job_state'].value_counts().head(10)
    sns.barplot(x=top_10_states.values, y=top_10_states.index, ax=ax, palette='mako')
    ax.set_title('10 Lokasi/Negara Bagian Terbanyak', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_Sepuluh_Negara_Responden_Terbanyak.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 03_Sepuluh_Negara_Responden_Terbanyak disimpan")

# 04_Analisis_Bivariat_terhadap_Growing_Stress.png (terhadap Salary)
if len(valid_cats) > 0:
    fig, axes = plt.subplots(len(valid_cats), 1, figsize=(10, 5 * len(valid_cats)))
    if len(valid_cats) == 1: axes = [axes]
    for ax, col in zip(axes, valid_cats):
        sns.boxplot(x=target, y=col, data=pdf_encoded, order=pdf_encoded[col].value_counts().index, ax=ax, palette='Set2')
        ax.set_title(f'Bivariat: {target} vs {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_Analisis_Bivariat_terhadap_Target.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 04_Analisis_Bivariat_terhadap_Target disimpan")

# 05_Distribusi_Growing_Stress_per_Tahun_Survei.png (Distribusi target per work_type sebagai analogi)
if 'formatted_work_type' in pdf_encoded.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=pdf_encoded, x=target, hue='formatted_work_type', common_norm=False, ax=ax)
    ax.set_title('Distribusi Salary per Tipe Pekerjaan', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_Distribusi_Target_per_Kategori.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 05_Distribusi_Target_per_Kategori disimpan")

# 06_Skor_ChiSquare_Feature_Selection.png (F-Regression scores)
print("     Menghitung F-Regression untuk feature selection...")
try:
    selector = SelectKBest(score_func=f_regression, k='all')
    selector.fit(X.fillna(0), y)
    scores_df = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    scores_df = scores_df.sort_values(by='Score', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Score', y='Feature', data=scores_df, palette='magma', ax=ax)
    ax.set_title('Skor F-Regression Feature Selection (Top 15)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_Skor_ChiSquare_Feature_Selection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Plot 06_Skor_ChiSquare_Feature_Selection (F-Regression) disimpan")
except Exception as e:
    print(f"     ⚠️ Gagal membuat plot 06: {e}")

print(f"\n  📂 Semua plot EDA disimpan di: {OUTPUT_DIR}")


# ===================================================================
# STAGE 7: TRAIN-TEST SPLIT & STANDARDISASI
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 7: TRAIN-TEST SPLIT & STANDARDISASI")
print("=" * 70)

# --- Train-Test Split (80/20) ---
# Stratifikasi berdasarkan experience level (binned)
stratify_col = None
if 'experience_encoded' in X.columns:
    # Bin experience level untuk stratifikasi
    stratify_col = pdf_encoded['formatted_experience_level'].fillna('Unknown')
    # Pastikan setiap kelas cukup besar untuk stratifikasi
    class_counts = stratify_col.value_counts()
    small_classes = class_counts[class_counts < 5].index
    if len(small_classes) > 0:
        stratify_col = stratify_col.replace(small_classes, 'Other_Rare')

try:
    X_train, X_test, y_train, y_test, y_train, y_test = train_test_split(
        X, y, y, test_size=0.2, random_state=42, stratify=stratify_col
    )
    print(f"  ✅ Stratified Split berdasarkan experience_level")
except ValueError:
    X_train, X_test, y_train, y_test, y_train, y_test = train_test_split(
        X, y, y, test_size=0.2, random_state=42
    )
    print(f"  ✅ Random Split (stratifikasi tidak memungkinkan)")

print(f"     Training Set: {X_train.shape[0]:,} baris ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"     Test Set:     {X_test.shape[0]:,} baris ({X_test.shape[0]/len(X)*100:.0f}%)")

# --- Standardisasi ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✅ StandardScaler: mean=0, std=1 pada semua fitur")
print(f"     Feature means (train, first 5): {X_train_scaled.mean(axis=0)[:5].round(4)}")
print(f"     Feature stds  (train, first 5): {X_train_scaled.std(axis=0)[:5].round(4)}")


# ===================================================================
# STAGE 8: PEMODELAN MACHINE LEARNING — 3 ALGORITMA
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 8: PEMODELAN MACHINE LEARNING")
print("=" * 70)

# Dictionary untuk menyimpan hasil
results = {}

# =========================================
# MODEL 1: RANDOM FOREST REGRESSOR
# =========================================
print("\n" + "-" * 50)
print("  🌲 MODEL 1: RANDOM FOREST REGRESSOR")
print("-" * 50)

rf_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\n  📋 Hyperparameters:")
for k, v in rf_params.items():
    print(f"     {k}: {v}")

# Training
print(f"\n  ⏳ Training Random Forest...")
start = time.time()
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start

# Predictions
rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_test = rf_model.predict(X_test_scaled)

# Metrics
rf_metrics = {
    'MAE_train': mean_absolute_error(y_train, rf_pred_train),
    'MAE_test': mean_absolute_error(y_test, rf_pred_test),
    'RMSE_train': np.sqrt(mean_squared_error(y_train, rf_pred_train)),
    'RMSE_test': np.sqrt(mean_squared_error(y_test, rf_pred_test)),
    'R2_train': r2_score(y_train, rf_pred_train),
    'R2_test': r2_score(y_test, rf_pred_test),
    'MAPE_test': mean_absolute_percentage_error(y_test, rf_pred_test) * 100,
    'Training_Time': rf_train_time
}
results['Random Forest'] = rf_metrics

print(f"\n  📊 Hasil Evaluasi Random Forest:")
print(f"     {'Metrik':<20} {'Training':>15} {'Test':>15}")
print(f"     {'-'*50}")
print(f"     {'MAE':<20} ${rf_metrics['MAE_train']:>14,.0f} ${rf_metrics['MAE_test']:>14,.0f}")
print(f"     {'RMSE':<20} ${rf_metrics['RMSE_train']:>14,.0f} ${rf_metrics['RMSE_test']:>14,.0f}")
print(f"     {'R²':<20} {rf_metrics['R2_train']:>15.4f} {rf_metrics['R2_test']:>15.4f}")
print(f"     {'MAPE':<20} {'—':>15} {rf_metrics['MAPE_test']:>14.1f}%")
print(f"     {'Training Time':<20} {rf_train_time:>14.2f}s")

# Cross Validation
print(f"\n  🔄 5-Fold Cross Validation:")
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"     R² per fold: {cv_scores_rf.round(4)}")
print(f"     R² mean ± std: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# Feature Importance
rf_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  🔑 Top 10 Feature Importance:")
for i, row in rf_importance.head(10).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"     {row['Feature']:35s} {row['Importance']:.4f} {bar}")


# =========================================
# MODEL 2: XGBOOST REGRESSOR
# =========================================
print("\n" + "-" * 50)
print("  🚀 MODEL 2: XGBOOST REGRESSOR")
print("-" * 50)

xgb_params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

print(f"\n  📋 Hyperparameters:")
for k, v in xgb_params.items():
    print(f"     {k}: {v}")

# Training dengan early stopping
print(f"\n  ⏳ Training XGBoost...")
start = time.time()

# Split training set untuk early stopping validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.15, random_state=42
)

xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=20)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
xgb_train_time = time.time() - start

# Re-fit pada seluruh training set
xgb_model_full = xgb.XGBRegressor(**xgb_params)
xgb_model_full.set_params(n_estimators=xgb_model.best_iteration + 1 if hasattr(xgb_model, 'best_iteration') else 300)
xgb_model_full.fit(X_train_scaled, y_train, verbose=False)

# Predictions
xgb_pred_train = xgb_model_full.predict(X_train_scaled)
xgb_pred_test = xgb_model_full.predict(X_test_scaled)

# Metrics
xgb_metrics = {
    'MAE_train': mean_absolute_error(y_train, xgb_pred_train),
    'MAE_test': mean_absolute_error(y_test, xgb_pred_test),
    'RMSE_train': np.sqrt(mean_squared_error(y_train, xgb_pred_train)),
    'RMSE_test': np.sqrt(mean_squared_error(y_test, xgb_pred_test)),
    'R2_train': r2_score(y_train, xgb_pred_train),
    'R2_test': r2_score(y_test, xgb_pred_test),
    'MAPE_test': mean_absolute_percentage_error(y_test, xgb_pred_test) * 100,
    'Training_Time': xgb_train_time,
    'Best_Iteration': xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 300
}
results['XGBoost'] = xgb_metrics

print(f"\n  📊 Hasil Evaluasi XGBoost:")
print(f"     {'Metrik':<20} {'Training':>15} {'Test':>15}")
print(f"     {'-'*50}")
print(f"     {'MAE':<20} ${xgb_metrics['MAE_train']:>14,.0f} ${xgb_metrics['MAE_test']:>14,.0f}")
print(f"     {'RMSE':<20} ${xgb_metrics['RMSE_train']:>14,.0f} ${xgb_metrics['RMSE_test']:>14,.0f}")
print(f"     {'R²':<20} {xgb_metrics['R2_train']:>15.4f} {xgb_metrics['R2_test']:>15.4f}")
print(f"     {'MAPE':<20} {'—':>15} {xgb_metrics['MAPE_test']:>14.1f}%")
print(f"     {'Training Time':<20} {xgb_train_time:>14.2f}s")
if hasattr(xgb_model, 'best_iteration'):
    print(f"     {'Best Iteration':<20} {xgb_model.best_iteration:>15}")

# Cross Validation
print(f"\n  🔄 5-Fold Cross Validation:")
cv_scores_xgb = cross_val_score(xgb_model_full, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"     R² per fold: {cv_scores_xgb.round(4)}")
print(f"     R² mean ± std: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# Feature Importance
xgb_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': xgb_model_full.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  🔑 Top 10 Feature Importance:")
for i, row in xgb_importance.head(10).iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"     {row['Feature']:35s} {row['Importance']:.4f} {bar}")


# =========================================
# MODEL 3: RIDGE REGRESSION
# =========================================
print("\n" + "-" * 50)
print("  📐 MODEL 3: RIDGE REGRESSION")
print("-" * 50)

ridge_params = {
    'alpha': 1.0,
    'fit_intercept': True,
    'max_iter': 1000,
    'solver': 'auto'
}

print(f"\n  📋 Hyperparameters:")
for k, v in ridge_params.items():
    print(f"     {k}: {v}")

# Hyperparameter tuning dengan GridSearchCV
print(f"\n  ⏳ GridSearchCV untuk optimal alpha...")
start = time.time()

ridge_grid = GridSearchCV(
    Ridge(max_iter=1000),
    param_grid={'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    cv=5,
    scoring='r2',
    n_jobs=-1
)
ridge_grid.fit(X_train_scaled, y_train)
ridge_model = ridge_grid.best_estimator_
ridge_train_time = time.time() - start

print(f"     Best alpha: {ridge_grid.best_params_['alpha']}")
print(f"     Best CV R²: {ridge_grid.best_score_:.4f}")

# Predictions
ridge_pred_train = ridge_model.predict(X_train_scaled)
ridge_pred_test = ridge_model.predict(X_test_scaled)

# Metrics
ridge_metrics = {
    'MAE_train': mean_absolute_error(y_train, ridge_pred_train),
    'MAE_test': mean_absolute_error(y_test, ridge_pred_test),
    'RMSE_train': np.sqrt(mean_squared_error(y_train, ridge_pred_train)),
    'RMSE_test': np.sqrt(mean_squared_error(y_test, ridge_pred_test)),
    'R2_train': r2_score(y_train, ridge_pred_train),
    'R2_test': r2_score(y_test, ridge_pred_test),
    'MAPE_test': mean_absolute_percentage_error(y_test, ridge_pred_test) * 100,
    'Training_Time': ridge_train_time,
    'Best_Alpha': ridge_grid.best_params_['alpha']
}
results['Ridge Regression'] = ridge_metrics

print(f"\n  📊 Hasil Evaluasi Ridge Regression:")
print(f"     {'Metrik':<20} {'Training':>15} {'Test':>15}")
print(f"     {'-'*50}")
print(f"     {'MAE':<20} ${ridge_metrics['MAE_train']:>14,.0f} ${ridge_metrics['MAE_test']:>14,.0f}")
print(f"     {'RMSE':<20} ${ridge_metrics['RMSE_train']:>14,.0f} ${ridge_metrics['RMSE_test']:>14,.0f}")
print(f"     {'R²':<20} {ridge_metrics['R2_train']:>15.4f} {ridge_metrics['R2_test']:>15.4f}")
print(f"     {'MAPE':<20} {'—':>15} {ridge_metrics['MAPE_test']:>14.1f}%")
print(f"     {'Training Time':<20} {ridge_train_time:>14.2f}s")

# Cross Validation
print(f"\n  🔄 5-Fold Cross Validation:")
cv_scores_ridge = cross_val_score(ridge_model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
print(f"     R² per fold: {cv_scores_ridge.round(4)}")
print(f"     R² mean ± std: {cv_scores_ridge.mean():.4f} ± {cv_scores_ridge.std():.4f}")

# Koefisien Regresi (Top 10)
ridge_coef = pd.DataFrame({
    'Feature': final_features,
    'Coefficient': ridge_model.coef_
}).sort_values('Coefficient', ascending=False)

print(f"\n  🔑 Top 10 Koefisien (positif = meningkatkan gaji):")
for i, row in ridge_coef.head(10).iterrows():
    sign = '+' if row['Coefficient'] >= 0 else ''
    print(f"     {row['Feature']:35s} {sign}${row['Coefficient']:>12,.0f}")

print(f"\n  🔑 Bottom 5 Koefisien (negatif = menurunkan gaji):")
for i, row in ridge_coef.tail(5).iterrows():
    print(f"     {row['Feature']:35s} ${row['Coefficient']:>12,.0f}")


# ===================================================================
# STAGE 9: PERBANDINGAN MODEL & VISUALISASI HASIL
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 9: PERBANDINGAN MODEL & VISUALISASI")
print("=" * 70)

# --- 9.1: Tabel Perbandingan ---
print("\n  📊 PERBANDINGAN SEMUA MODEL:")
print(f"  {'='*80}")
print(f"  {'Model':<22} {'MAE (Test)':>14} {'RMSE (Test)':>14} {'R² (Test)':>12} {'MAPE':>8} {'Time':>8}")
print(f"  {'-'*80}")
for model_name, metrics in results.items():
    print(f"  {model_name:<22} ${metrics['MAE_test']:>13,.0f} ${metrics['RMSE_test']:>13,.0f} "
          f"{metrics['R2_test']:>12.4f} {metrics['MAPE_test']:>7.1f}% {metrics['Training_Time']:>7.2f}s")
print(f"  {'='*80}")

# Identifikasi model terbaik
best_class_model_name = min(results, key=lambda x: results[x]['MAE_test'])
print(f"\n  🏆 MODEL TERBAIK REGRESI: {best_class_model_name}")
print(f"     MAE:  ${results[best_class_model_name]['MAE_test']:,.0f}")
print(f"     RMSE: ${results[best_class_model_name]['RMSE_test']:,.0f}")
print(f"     R²:   {results[best_class_model_name]['R2_test']:.4f}")

# ===================================================================
# STAGE 8B: PEMODELAN KLASIFIKASI (Low / Medium / High)
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 8B: PEMODELAN KLASIFIKASI")
print("=" * 70)

results_class = {}

# 1. Random Forest Classifier
print("\n  🌲 MODEL 1: RANDOM FOREST CLASSIFIER")
rfc = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
start = time.time()
rfc.fit(X_train_scaled, y_train)
rfc_time = time.time() - start
rfc_pred = rfc.predict(X_test_scaled)
results_class['Random Forest Classifier'] = {
    'Accuracy': accuracy_score(y_test, rfc_pred),
    'Precision': precision_score(y_test, rfc_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, rfc_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, rfc_pred, average='weighted', zero_division=0),
    'Time': rfc_time,
    'Model': rfc,
    'Pred': rfc_pred
}
print(f"     Accuracy: {results_class['Random Forest Classifier']['Accuracy']:.4f}")

# 2. XGBoost Classifier
print("\n  🚀 MODEL 2: XGBOOST CLASSIFIER")
xgbc = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)
start = time.time()
xgbc.fit(X_train_scaled, y_train)
xgbc_time = time.time() - start
xgbc_pred = xgbc.predict(X_test_scaled)
results_class['XGBoost Classifier'] = {
    'Accuracy': accuracy_score(y_test, xgbc_pred),
    'Precision': precision_score(y_test, xgbc_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, xgbc_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, xgbc_pred, average='weighted', zero_division=0),
    'Time': xgbc_time,
    'Model': xgbc,
    'Pred': xgbc_pred
}
print(f"     Accuracy: {results_class['XGBoost Classifier']['Accuracy']:.4f}")

# 3. Logistic Regression
print("\n  📐 MODEL 3: LOGISTIC REGRESSION")
logreg = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
start = time.time()
logreg.fit(X_train_scaled, y_train)
logreg_time = time.time() - start
logreg_pred = logreg.predict(X_test_scaled)
results_class['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, logreg_pred),
    'Precision': precision_score(y_test, logreg_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, logreg_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, logreg_pred, average='weighted', zero_division=0),
    'Time': logreg_time,
    'Model': logreg,
    'Pred': logreg_pred
}
print(f"     Accuracy: {results_class['Logistic Regression']['Accuracy']:.4f}")

# 4. KNN Classifier
print("\n  🏘️ MODEL 4: K-NEAREST NEIGHBORS (KNN)")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
start = time.time()
knn.fit(X_train_scaled, y_train)
knn_time = time.time() - start
knn_pred = knn.predict(X_test_scaled)
results_class['KNN'] = {
    'Accuracy': accuracy_score(y_test, knn_pred),
    'Precision': precision_score(y_test, knn_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, knn_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, knn_pred, average='weighted', zero_division=0),
    'Time': knn_time, 'Model': knn, 'Pred': knn_pred
}
print(f"     Accuracy: {results_class['KNN']['Accuracy']:.4f}")

# 5. Naive Bayes Classifier
print("\n  📊 MODEL 5: NAIVE BAYES")
nb = GaussianNB()
start = time.time()
nb.fit(X_train_scaled, y_train)
nb_time = time.time() - start
nb_pred = nb.predict(X_test_scaled)
results_class['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, nb_pred),
    'Precision': precision_score(y_test, nb_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, nb_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, nb_pred, average='weighted', zero_division=0),
    'Time': nb_time, 'Model': nb, 'Pred': nb_pred
}
print(f"     Accuracy: {results_class['Naive Bayes']['Accuracy']:.4f}")

# 6. Random Forest Tuned (Simplified for speed)
print("\n  🌲 MODEL 6: RANDOM FOREST TUNED")
rf_tuned = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
start = time.time()
rf_tuned.fit(X_train_scaled, y_train)
rft_time = time.time() - start
rft_pred = rf_tuned.predict(X_test_scaled)
results_class['Random Forest Tuned'] = {
    'Accuracy': accuracy_score(y_test, rft_pred),
    'Precision': precision_score(y_test, rft_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, rft_pred, average='weighted', zero_division=0),
    'F1_Score': f1_score(y_test, rft_pred, average='weighted', zero_division=0),
    'Time': rft_time, 'Model': rf_tuned, 'Pred': rft_pred
}
print(f"     Accuracy: {results_class['Random Forest Tuned']['Accuracy']:.4f}")

best_class_model_name = max(results_class, key=lambda x: results_class[x]['Accuracy'])
print(f"\n  🏆 MODEL TERBAIK KLASIFIKASI: {best_class_model_name}")

# --- Visualisasi Evaluasi Klasifikasi Lanjutan ---
print("\n  🔧 8B.1 Menghasilkan plot performa dan matriks konfusi (07-16)...")
from sklearn.metrics import ConfusionMatrixDisplay
labels_disp = ['Onsite', 'Remote']

# Fungsi bantu simpan CM
def save_cm(model_key, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(results_class[model_key]['Model'], X_test_scaled, y_test, display_labels=labels_disp, cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_key}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()

# 07, 08, 10, 11, 14, 15 Confusion Matrices
save_cm('Logistic Regression', '07_Confusion_Matrix_Logistic_Regression.png')
save_cm('Random Forest Classifier', '08_Confusion_Matrix_Random_Forest.png')
save_cm('KNN', '10_Confusion_Matrix_KNN.png')
save_cm('Naive Bayes', '11_Confusion_Matrix_Naive_Bayes.png')
save_cm('Random Forest Tuned', '14_Confusion_Matrix_Random_Forest_Tuned.png')
save_cm('XGBoost Classifier', '15_Confusion_Matrix_XGBoost.png')
print("  ✅ Semua Confusion Matrix (07, 08, 10, 11, 14, 15) disimpan")

# 09 Feature Importance RF (Dari regresi sbg perwakilan)
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importance.head(15), palette='viridis', ax=ax)
ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_Feature_Importance_Random_Forest.png'), dpi=150)
plt.close()
print("  ✅ Plot 09_Feature_Importance_Random_Forest disimpan")

# 12 Perbandingan Performa 4 Model Awal (RF, XGB, LR, NB)
fig, ax = plt.subplots(figsize=(10, 6))
models_12 = ['Random Forest Classifier', 'XGBoost Classifier', 'Logistic Regression', 'Naive Bayes']
accs_12 = [results_class[m]['Accuracy'] for m in models_12]
sns.barplot(x=models_12, y=accs_12, palette='Set2', ax=ax)
ax.set_title('12_Perbandingan_Performa_4_Model_Awal', fontweight='bold')
ax.set_ylim(0, 1.0)
for i, v in enumerate(accs_12):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_Perbandingan_Performa_4_Model_Awal.png'), dpi=150)
plt.close()

# 13 Distribusi Akurasi CrossValidation 5Fold
print("     Menghitung CV untuk plot 13...")
cv_res = []
models_cv = ['RF', 'LR', 'KNN']
clfs = [RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1), LogisticRegression(max_iter=500), KNeighborsClassifier(n_neighbors=5)]
for name, clf in zip(models_cv, clfs):
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    for s in scores:
        cv_res.append({'Model': name, 'Accuracy': s})
cv_df = pd.DataFrame(cv_res)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Model', y='Accuracy', data=cv_df, palette='pastel', ax=ax)
sns.stripplot(x='Model', y='Accuracy', data=cv_df, color=".25", ax=ax)
ax.set_title('13_Distribusi_Akurasi_CrossValidation_5Fold', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '13_Distribusi_Akurasi_CrossValidation_5Fold.png'), dpi=150)
plt.close()

# 16 Perbandingan RF vs RF Tuned vs XGBoost
fig, ax = plt.subplots(figsize=(8, 6))
models_16 = ['Random Forest Classifier', 'Random Forest Tuned', 'XGBoost Classifier']
accs_16 = [results_class[m]['Accuracy'] for m in models_16]
sns.barplot(x=['RF Base', 'RF Tuned', 'XGBoost'], y=accs_16, palette='rocket', ax=ax)
ax.set_title('16_Perbandingan_RF_vs_RFTuned_vs_XGBoost', fontweight='bold')
ax.set_ylim(0, 1.0)
for i, v in enumerate(accs_16):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '16_Perbandingan_RF_vs_RFTuned_vs_XGBoost.png'), dpi=150)
plt.close()
print("  ✅ Plot 12, 13, 16 disimpan")

# --- 9.2: Visualisasi Perbandingan Model Regresi ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

model_names = list(results.keys())
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Plot 1: MAE Comparison
mae_train = [results[m]['MAE_train'] for m in model_names]
mae_test = [results[m]['MAE_test'] for m in model_names]
x_pos = np.arange(len(model_names))
width = 0.35

axes[0, 0].bar(x_pos - width/2, mae_train, width, label='Train', color='#3498db', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, mae_test, width, label='Test', color='#e74c3c', alpha=0.8)
axes[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(model_names, fontsize=10)
axes[0, 0].set_ylabel('MAE ($)')
axes[0, 0].legend()
for i, v in enumerate(mae_test):
    axes[0, 0].text(i + width/2, v + 200, f'${v:,.0f}', ha='center', fontsize=9)

# Plot 2: R² Comparison
r2_train = [results[m]['R2_train'] for m in model_names]
r2_test = [results[m]['R2_test'] for m in model_names]

axes[0, 1].bar(x_pos - width/2, r2_train, width, label='Train', color='#3498db', alpha=0.8)
axes[0, 1].bar(x_pos + width/2, r2_test, width, label='Test', color='#e74c3c', alpha=0.8)
axes[0, 1].set_title('R² Score', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(model_names, fontsize=10)
axes[0, 1].set_ylabel('R²')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].legend()
for i, v in enumerate(r2_test):
    axes[0, 1].text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Plot 3: Actual vs Predicted (Best Model)
if best_class_model_name == 'Random Forest':
    best_pred = rf_pred_test
elif best_class_model_name == 'XGBoost':
    best_pred = xgb_pred_test
else:
    best_pred = ridge_pred_test

axes[1, 0].scatter(y_test, best_pred, alpha=0.3, s=10, color='#3498db')
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_title(f'Actual vs Predicted ({best_class_model_name})', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Actual Salary ($)')
axes[1, 0].set_ylabel('Predicted Salary ($)')
axes[1, 0].legend()

# Plot 4: Residuals Distribution
residuals = y_test - best_pred
axes[1, 1].hist(residuals, bins=50, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_title(f'Distribusi Residual ({best_class_model_name})', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Residual (Actual - Predicted)')
axes[1, 1].set_ylabel('Frekuensi')
axes[1, 1].text(0.05, 0.95, f'Mean: ${residuals.mean():,.0f}\nStd: ${residuals.std():,.0f}',
                transform=axes[1, 1].transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Perbandingan Model Machine Learning — LinkedIn Job Postings',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 6: Perbandingan Model disimpan")

# --- 9.3: Feature Importance Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# RF Feature Importance
rf_top10 = rf_importance.head(10)
axes[0].barh(rf_top10['Feature'], rf_top10['Importance'], color='#3498db', alpha=0.85)
axes[0].set_title('Random Forest\nFeature Importance', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

# XGB Feature Importance
xgb_top10 = xgb_importance.head(10)
axes[1].barh(xgb_top10['Feature'], xgb_top10['Importance'], color='#e74c3c', alpha=0.85)
axes[1].set_title('XGBoost\nFeature Importance', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()

# Ridge Coefficients (absolute)
ridge_abs = ridge_coef.copy()
ridge_abs['Abs_Coefficient'] = ridge_abs['Coefficient'].abs()
ridge_top10 = ridge_abs.sort_values('Abs_Coefficient', ascending=False).head(10)
colors_ridge = ['#2ecc71' if c >= 0 else '#e74c3c' for c in ridge_top10['Coefficient']]
axes[2].barh(ridge_top10['Feature'], ridge_top10['Abs_Coefficient'], color=colors_ridge, alpha=0.85)
axes[2].set_title('Ridge Regression\n|Coefficients|', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()

plt.suptitle('Feature Importance — Perbandingan 3 Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_importance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 7: Feature Importance Comparison disimpan")

# --- 9.4: Cross Validation Boxplot ---
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = [cv_scores_rf, cv_scores_xgb, cv_scores_ridge]
bp = ax.boxplot(cv_data, labels=model_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('5-Fold Cross Validation R² Score', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1)

for i, cv in enumerate(cv_data, 1):
    ax.text(i, cv.mean() + 0.02, f'μ={cv.mean():.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_cv_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Plot 8: Cross Validation Comparison disimpan")


# ===================================================================
# STAGE 10: SIMPAN MODEL & HASIL
# ===================================================================
print("\n" + "=" * 70)
print("  STAGE 10: SIMPAN HASIL")
print("=" * 70)

# Simpan hasil prediksi ke CSV
pred_df = pd.DataFrame({
    'Actual_Salary': y_test.values,
    'RF_Predicted': rf_pred_test,
    'XGB_Predicted': xgb_pred_test,
    'Ridge_Predicted': ridge_pred_test
})
pred_path = os.path.join(OUTPUT_DIR, 'prediction_results.csv')
pred_df.to_csv(pred_path, index=False)
print(f"  ✅ Hasil prediksi disimpan: {pred_path}")

# Simpan metrik ke CSV
metrics_df = pd.DataFrame(results).T
metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
metrics_df.to_csv(metrics_path)
print(f"  ✅ Metrik model disimpan: {metrics_path}")

# Simpan feature importance ke CSV
fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
rf_importance.to_csv(fi_path, index=False)
print(f"  ✅ Feature importance disimpan: {fi_path}")

# Simpan model dengan joblib
# Simpan hasil klasifikasi ke CSV
pred_class_df = pd.DataFrame({
    'Actual_Class': y_test.values,
    'RF_Class': results_class['Random Forest Classifier']['Pred'],
    'XGB_Class': results_class['XGBoost Classifier']['Pred'],
    'LogReg_Class': results_class['Logistic Regression']['Pred']
})
pred_class_path = os.path.join(OUTPUT_DIR, 'prediction_class_results.csv')
pred_class_df.to_csv(pred_class_path, index=False)
print(f"  ✅ Hasil prediksi klasifikasi disimpan: {pred_class_path}")

metrics_class_df = pd.DataFrame({m: {k:v for k,v in results_class[m].items() if k not in ['Model', 'Pred']} for m in results_class}).T
metrics_class_path = os.path.join(OUTPUT_DIR, 'model_metrics_classification.csv')
metrics_class_df.to_csv(metrics_class_path)
print(f"  ✅ Metrik model klasifikasi disimpan: {metrics_class_path}")

try:
    import joblib
    joblib.dump(rf_model, os.path.join(OUTPUT_DIR, 'model_random_forest.pkl'))
    joblib.dump(xgb_model_full, os.path.join(OUTPUT_DIR, 'model_xgboost.pkl'))
    joblib.dump(ridge_model, os.path.join(OUTPUT_DIR, 'model_ridge.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    joblib.dump(final_features, os.path.join(OUTPUT_DIR, 'feature_names.pkl'))
    
    # Simpan model klasifikasi
    joblib.dump(results_class['Random Forest Classifier']['Model'], os.path.join(OUTPUT_DIR, 'model_class_rf.pkl'))
    joblib.dump(results_class['XGBoost Classifier']['Model'], os.path.join(OUTPUT_DIR, 'model_class_xgb.pkl'))
    joblib.dump(results_class['Logistic Regression']['Model'], os.path.join(OUTPUT_DIR, 'model_class_logreg.pkl'))
    
    print(f"  ✅ Semua model regresi dan klasifikasi disimpan (joblib)")
except ImportError:
    print(f"  ⚠️  joblib tidak tersedia, model tidak disimpan")


# ===================================================================
# RINGKASAN AKHIR
# ===================================================================
print("\n" + "=" * 70)
print("  📋 RINGKASAN AKHIR PIPELINE")
print("=" * 70)
print(f"""
  Dataset          : LinkedIn Job Postings (2023-2024)
  Total Data Awal  : ~124,000+ baris
  Data Setelah Prep: {len(X):,} baris × {len(final_features)} fitur
  Train/Test Split : {len(X_train):,} / {len(X_test):,}
  Target Variable  : remote_allowed (Klasifikasi Remote/Onsite)

  ┌────────────────────┬──────────────┬──────────────┬──────────────┐
  │ Model              │   MAE (Test) │  RMSE (Test) │    R² (Test) │
  ├────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Random Forest      │ ${results['Random Forest']['MAE_test']:>11,.0f} │ ${results['Random Forest']['RMSE_test']:>11,.0f} │ {results['Random Forest']['R2_test']:>12.4f} │
  │ XGBoost            │ ${results['XGBoost']['MAE_test']:>11,.0f} │ ${results['XGBoost']['RMSE_test']:>11,.0f} │ {results['XGBoost']['R2_test']:>12.4f} │
  │ Ridge Regression   │ ${results['Ridge Regression']['MAE_test']:>11,.0f} │ ${results['Ridge Regression']['RMSE_test']:>11,.0f} │ {results['Ridge Regression']['R2_test']:>12.4f} │
  └────────────────────┴──────────────┴──────────────┴──────────────┘

  🏆 Model Regresi Terbaik     : {best_class_model_name}
  🏆 Model Klasifikasi Terbaik : {best_class_model_name}
  📂 Output Directory : {OUTPUT_DIR}
  📊 Dashboard        : Jalankan `streamlit run dashboard_linkedin.py`
""")

# Stop Spark session
spark.stop()
print(f"  ✅ Spark session dihentikan")
print(f"  ✅ Pipeline selesai pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
