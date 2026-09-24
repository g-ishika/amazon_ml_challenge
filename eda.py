import pandas as pd
import os



FILE = "student_resources.zip"   # change this


# =========================
# 2. BASIC FILE INFO
# =========================

file_size_gb = os.path.getsize(FILE) / (1024 ** 3)

print("=" * 60)
print("FILE INFORMATION")
print("=" * 60)

print(f"File: {FILE}")
print(f"Size: {file_size_gb:.2f} GB")


# =========================
# 3. LOAD A SAMPLE
# =========================

# Only read 100,000 rows
df = pd.read_csv(FILE, nrows=100_000)

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# =========================
# 4. COLUMN INFORMATION
# =========================

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, col in enumerate(df.columns):
    print(f"{i}: {col}")


# =========================
# 5. DATA TYPES
# =========================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# =========================
# 6. FIRST 5 ROWS
# =========================

print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)

print(df.head())


# =========================
# 7. MISSING VALUES
# =========================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

print(missing_table.sort_values(
    "missing_percent",
    ascending=False
))


# =========================
# 8. DUPLICATES
# =========================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())


# =========================
# 9. UNIQUE VALUES
# =========================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for col in df.columns:
    print(f"\n{col}")
    print("Unique:", df[col].nunique(dropna=True))


# =========================
# 10. DATA TYPES SUMMARY
# =========================

print("\n" + "=" * 60)
print("NUMERIC SUMMARY")
print("=" * 60)

print(df.describe(include="all").T)