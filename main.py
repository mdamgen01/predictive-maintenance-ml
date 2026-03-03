import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/predictive_maintenance.csv")

# Drop missing values
df = df.dropna()

# Set target
TARGET_COLUMN = "Target"

# Drop columns we should not use
df = df.drop(columns=["UDI", "Product ID", "Failure Type"])

# One-hot encode the Type column
df = pd.get_dummies(df, columns=["Type"])

# Split features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Stratified split (important for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Apply SMOTE ONLY to training data ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:")
print(y_train.value_counts())
print("\nAfter SMOTE:")
print(y_train_res.value_counts())

# --- Train model on balanced data ---
model = RandomForestClassifier(
    random_state=42,
    n_estimators=200
)

model.fit(X_train_res, y_train_res)

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

for t in thresholds:
    preds_t = (probs >= t).astype(int)
    print(f"\nThreshold: {t}")
    print(classification_report(y_test, preds_t))
