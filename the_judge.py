import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

print("Loading data and simulating Teammate's NLP model...")

# 1. Load Data
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Extract Your Metadata (Same as before)
df['title_length'] = df['title'].apply(len)
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['title_uppercase_ratio'] = df['title'].apply(
    lambda t: sum(1 for w in str(t).split() if w.isupper()) / max(len(str(t).split()), 1)
)

# 3. MOCKING TEAMMATE 1'S OUTPUT (The Text Stream)
# We are pretending your teammate's NLP model gave us a "Text Risk Score" (0.0 to 1.0)
# Fake articles generally get a higher score, True articles get a lower score.
np.random.seed(42)
df['text_risk_score'] = np.where(df['label'] == 1, 
                                 np.random.normal(0.8, 0.1, len(df)),  # Fake ~ 80% risk
                                 np.random.normal(0.2, 0.1, len(df)))  # True ~ 20% risk
df['text_risk_score'] = df['text_risk_score'].clip(0, 1) # Keep between 0 and 1

# 4. Prepare the Fusion Dataset
# We are using ONLY numerical features here for simplicity to build the final Judge
fusion_features = ['text_risk_score', 'title_length', 'text_word_count', 'title_uppercase_ratio']
X = df[fusion_features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Final Judge (XGBoost)
print("Training the Final Fusion Judge...")
judge_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
judge_model.fit(X_train, y_train)

print(f"Final Judge Accuracy: {judge_model.score(X_test, y_test) * 100:.2f}%")

# ==========================================
# 6. THE EXPLAINABLE AI (XAI) - The Waterfall Plot
# ==========================================
print("\nGenerating the final XAI Report for the User...")

# Set up the SHAP explainer
explainer = shap.TreeExplainer(judge_model)
shap_values = explainer(X_test)

# Let's explain Article #5 in our test set
sample_idx = 5
actual_label = "Fake" if y_test.iloc[sample_idx] == 1 else "True"
predicted_label = "Fake" if judge_model.predict(X_test.iloc[[sample_idx]])[0] == 1 else "True"

print(f"\n--- Final Verdict for Article #{sample_idx} ---")
print(f"Actual: {actual_label} | Judge Predicted: {predicted_label}")

# Plot the Waterfall Chart for this specific article
plt.figure(figsize=(10, 6))
plt.title(f"Why The Judge Ruled This Article as '{predicted_label}'")
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.tight_layout()
plt.show()