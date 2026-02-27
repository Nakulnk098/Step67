import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

print("Initializing the Multimodal System and Training the Judge...")

# 1. QUICK TRAINING (Simulating the Fusion Dataset)
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df]).sample(frac=0.5, random_state=42).reset_index(drop=True)

# Engineer Metadata Features
df['title_length'] = df['title'].apply(len)
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['title_uppercase_ratio'] = df['title'].apply(
    lambda t: sum(1 for w in str(t).split() if w.isupper()) / max(len(str(t).split()), 1)
)

# ---------------------------------------------------------
# 🌟 THE MOCK MULTIMODAL ENGINES 🌟
# ---------------------------------------------------------
np.random.seed(42)
# Mocking Teammate 1 (NLP/Text Risk)
df['text_risk_score'] = np.where(df['label'] == 1, 
                                 np.random.normal(0.8, 0.1, len(df)),  
                                 np.random.normal(0.2, 0.1, len(df)))
df['text_risk_score'] = df['text_risk_score'].clip(0, 1)

# Mocking Teammate 2 (Computer Vision/Image Risk) -> NEW!
# Fake news images often have manipulation (higher risk score)
df['image_risk_score'] = np.where(df['label'] == 1, 
                                 np.random.normal(0.75, 0.15, len(df)),  
                                 np.random.normal(0.15, 0.1, len(df)))
df['image_risk_score'] = df['image_risk_score'].clip(0, 1)
# ---------------------------------------------------------

# Define all 5 features for the Judge
features = ['text_risk_score', 'image_risk_score', 'title_length', 'text_word_count', 'title_uppercase_ratio']
X = df[features]
y = df['label']

# Train the Multimodal Judge
judge_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
judge_model.fit(X, y)
explainer = shap.TreeExplainer(judge_model)

print("✅ Multimodal System Ready!\n")
print("="*60)
print(" 🕵️‍♂️ WELCOME TO THE MULTIMODAL FAKE NEWS DETECTOR ")
print("="*60)

# 2. INTERACTIVE LOOP
while True:
    print("\n--- Enter New Post Data ---")
    
    tweet_text = input("1. Enter the Tweet/Headline (or 'quit' to exit): ")
    if tweet_text.lower() in ['quit', 'exit']:
        break
        
    word_count_input = input("2. How many words in the full article? (e.g. 300): ")
    
    print("\n[Simulating the Deep Learning Models]")
    mock_nlp_score = input("3. Enter the Text Risk Score (0.0 to 1.0): ")
    mock_img_score = input("4. Enter the Image Risk Score (0.0 to 1.0): ")

    # Calculate engineered features
    title_len = len(tweet_text)
    word_count = int(word_count_input)
    words = tweet_text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)

    # Create a DataFrame for the new input
    new_data = pd.DataFrame({
        'text_risk_score': [float(mock_nlp_score)],
        'image_risk_score': [float(mock_img_score)],
        'title_length': [title_len],
        'text_word_count': [word_count],
        'title_uppercase_ratio': [uppercase_ratio]
    })

    # 3. MAKE PREDICTION
    prediction = judge_model.predict(new_data)[0]
    probability = judge_model.predict_proba(new_data)[0]
    
    result = "🔴 FAKE NEWS" if prediction == 1 else "🟢 REAL NEWS"
    confidence = probability[1] * 100 if prediction == 1 else probability[0] * 100

    print("\n" + "="*40)
    print(f" THE JUDGE'S VERDICT: {result}")
    print(f" CONFIDENCE: {confidence:.2f}%")
    print("="*40)
    
    # 4. EXPLAIN THE DECISION (SHAP)
    print("Generating Explanation Chart... (Close the chart to test another post)")
    shap_values = explainer(new_data)
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Why The Judge Ruled This Post as {result}")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.show()