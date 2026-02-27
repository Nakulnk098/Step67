import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt
import joblib
# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
print("Loading datasets...")
# Ensure Fake.csv and True.csv are in the same folder as this script
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels: 1 for Fake, 0 for True
fake_df['label'] = 1
true_df['label'] = 0

# Combine and shuffle the dataset
df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING (YOUR MAIN JOB)
# ==========================================
print("Extracting Metadata Features...")

# A. Date Features (Fake news often spikes at certain times)
# Handle messy dates using errors='coerce' to turn bad data into NaT (Not a Time)
df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date_parsed'].dt.month.fillna(0) # 0 for missing months
df['year'] = df['date_parsed'].dt.year.fillna(0)

# B. Meta-Textual Features (Structure, not meaning)
# 1. Title length
df['title_length'] = df['title'].apply(len)
# 2. Text word count
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))
# 3. Ratio of uppercase words in the title (Clickbait detection)
def uppercase_ratio(title):
    words = str(title).split()
    if len(words) == 0: return 0
    upper_words = sum(1 for word in words if word.isupper())
    return upper_words / len(words)

df['title_uppercase_ratio'] = df['title'].apply(uppercase_ratio)

# Define our final metadata feature columns
metadata_features = ['subject', 'month', 'year', 'title_length', 'text_word_count', 'title_uppercase_ratio']
X = df[metadata_features]
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD AND TRAIN THE METADATA JUDGE
# ==========================================
print("Training the Metadata Model (XGBoost)...")

# We need to One-Hot Encode the 'subject' column because it is text (e.g., 'politicsNews')
# The other features are already numbers, so we pass them through unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['subject'])
    ],
    remainder='passthrough'
)

# Build the pipeline with XGBoost
metadata_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'))
])

# Train the model
metadata_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = metadata_pipeline.predict(X_test)
print("\n--- Model Performance (Metadata Only) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['True', 'Fake']))

# ==========================================
# 4. EXPLAINABLE AI (XAI) - THE "WHY"
# ==========================================
print("\nGenerating visual explanations using SHAP...")

# Transform the test data to get the exact features XGBoost used
X_test_transformed = metadata_pipeline.named_steps['preprocessor'].transform(X_test)

# Get feature names back from the preprocessor
cat_features = metadata_pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(['subject'])
num_features = ['month', 'year', 'title_length', 'text_word_count', 'title_uppercase_ratio']
all_feature_names = list(cat_features) + num_features

# Create the SHAP Explainer
explainer = shap.TreeExplainer(metadata_pipeline.named_steps['classifier'])
# We use a small sample (100 rows) to make the visualization render quickly
shap_values = explainer.shap_values(X_test_transformed[:100]) 

# Plot 1: Summary Plot (Shows overall feature importance for the whole dataset)
plt.figure()
plt.title("What makes an article Fake vs. Real? (Metadata overall)")
shap.summary_plot(shap_values, X_test_transformed[:100], feature_names=all_feature_names, show=False)
plt.tight_layout()
plt.show()

# Plot 2: Waterfall Plot for a SINGLE article (Explaining ONE specific prediction)
sample_idx = 0  # Change this number to explain different articles
shap_explanation = shap.Explanation(values=shap_values[sample_idx], 
                                    base_values=explainer.expected_value, 
                                    data=X_test_transformed[sample_idx], 
                                    feature_names=all_feature_names)

plt.figure()
plt.title(f"Why did the model predict {'Fake' if y_pred[sample_idx] == 1 else 'True'} for Article #{sample_idx}?")
shap.plots.waterfall(shap_explanation, show=False)
plt.tight_layout()
plt.show()


# ==========================================
# 5. SAVE THE MODEL FOR THE TEAM
# ==========================================
print("\nSaving the Metadata Pipeline...")

# Define the filename
filename = 'metadata_judge_pipeline.pkl'

# Save the entire pipeline (Preprocessor + XGBoost Classifier)
joblib.dump(metadata_pipeline, filename)

print(f"✅ Success! Model saved as '{filename}'.")
print("You can now share this file with your team!")