import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

print("Initializing the System and Training the Judge...")

# 1. QUICK TRAINING (So the script is self-contained)
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df]).sample(frac=0.5, random_state=42).reset_index(drop=True)

df['title_length'] = df['title'].apply(len)
df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['title_uppercase_ratio'] = df['title'].apply(
    lambda t: sum(1 for w in str(t).split() if w.isupper()) / max(len(str(t).split()), 1)
)
np.random.seed(42)
df['text_risk_score'] = np.where(df['label'] == 1, 
                                 np.random.normal(0.8, 0.1, len(df)),  
                                 np.random.normal(0.2, 0.1, len(df)))
df['text_risk_score'] = df['text_risk_score'].clip(0, 1)

features = ['text_risk_score', 'title_length', 'text_word_count', 'title_uppercase_ratio']
X = df[features]
y = df['label']

judge_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
judge_model.fit(X, y)
explainer = shap.TreeExplainer(judge_model)

print("✅ System Ready!\n")
print("="*50)
print(" 🕵️‍♂️ WELCOME TO THE FAKE NEWS DETECTOR TEST RUN ")
print("="*50)

# 2. INTERACTIVE LOOP
while True:
    print("\n--- Enter New Post Data ---")
    
    # Get user inputs
    tweet_text = input("1. Enter the Tweet/Headline: ")
    if tweet_text.lower() in ['quit', 'exit']:
        print("Exiting test run. Good luck at the hackathon!")
        break
        
    word_count_input = input("2. How many words in the full article? (e.g. 300): ")
    
    print("\n[Simulating Teammate's NLP Model]")
    mock_nlp_score = input("3. Enter the Teammate's Text Risk Score (0.0 to 1.0, e.g. 0.85): ")

    # Calculate engineered features exactly like your pipeline does
    title_len = len(tweet_text)
    word_count = int(word_count_input)
    words = tweet_text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(len(words), 1)
    nlp_score = float(mock_nlp_score)

    # Create a DataFrame for the new input
    new_data = pd.DataFrame({
        'text_risk_score': [nlp_score],
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
    plt.show() # Code pauses here until you close the graph window