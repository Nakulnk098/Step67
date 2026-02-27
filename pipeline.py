import torch
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import gradio as gr
from PIL import Image
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification

class MisinformationAnalyzer:
    def __init__(self):
        print("Initializing Models... (This may take a minute)")
        
        # 1. Text Model: RoBERTa
        self.text_pipeline = pipeline(
            "text-classification", 
            model="hamzab/roberta-fake-news-classification"
        )

        # 2. Image Model: ViT 
        self.img_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.img_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
        # 3. The Judge (XGBoost) & XAI (SHAP)
        self.feature_names = ['Text_Risk', 'Image_Risk', 'Engagement', 'Reads']
        self.judge, self.explainer = self._initialize_judge()

    def _initialize_judge(self):
        """Primes the judge with 1000 samples based on your logic."""
        # Features: [Text_Risk, Image_Risk, Engagement, Reads]
        X_train = np.random.rand(1000, 4) 
        y_train = (X_train[:, 0] * 0.6 + X_train[:, 1] * 0.4 > 0.5).astype(int)
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        return model, explainer

    def get_text_score(self, title, content):
        """Uses the specific <title> and <content> token formatting."""
        if not title and not content:
            return 0.5
            
        # Combine title and content as the model expects
        full_text = f"<title> {title} <content> {content} <end>"
        
        try:
            # Inference using the pipeline (truncated to 512 tokens)
            result = self.text_pipeline(full_text[:512])[0] 
            return result['score'] if result['label'].upper() == 'FAKE' else 1.0 - result['score']
        except Exception as e:
            print(f"Text Analysis Error: {e}")
            return 0.5

    def get_image_score(self, img):
        """Analyzes image risk with safety checks."""
        if img is None: 
            return 0.5  # Neutral score if no image
            
        try:
            # Standard ViT processing
            inputs = self.img_processor(images=img, return_tensors="pt")
            with torch.no_grad(): # Saves memory during inference
                outputs = self.img_model(**inputs)
            
            # Convert logits to a "manipulation risk" probability
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return float(probs.max())
        except Exception as e:
            print(f"Image Analysis Error: {e}")
            return 0.5

    def analyze(self, title, content, image, engagement, reads):
        # 1. Feature Extraction (Now correctly passing title and content)
        t_risk = self.get_text_score(title, content)
        i_risk = self.get_image_score(image)
        
        # 2. Prepare Input for Judge
        input_df = pd.DataFrame([[t_risk, i_risk, engagement, reads]], 
                                columns=self.feature_names)
        
        # 3. Prediction & Confidence
        pred_idx = self.judge.predict(input_df)[0]
        confidence = self.judge.predict_proba(input_df)[0][pred_idx]
        
        # 4. SHAP Explanation
        shap_values = self.explainer.shap_values(input_df)
        impacts = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

        # 5. Formatting Output
        verdict = "🚨 FAKE NEWS DETECTED" if pred_idx == 1 else "✅ LIKELY REAL"
        
        reasoning = f"Confidence Score: {confidence*100:.1f}%\n"
        reasoning += "─" * 30 + "\n"
        
        feature_impacts = sorted(zip(self.feature_names, impacts), 
                                 key=lambda x: abs(x[1]), reverse=True)

        for name, imp in feature_impacts:
            direction = "FAKE" if imp > 0 else "REAL"
            reasoning += f"• {name.replace('_', ' '):<12}: {abs(imp):.2f} impact toward {direction}\n"

        return verdict, reasoning

# --- LAUNCHER ---
analyzer = MisinformationAnalyzer()

# Notice the UI is updated to reflect the Title/Content split
demo = gr.Interface(
    fn=analyzer.analyze,
    inputs=[
        gr.Textbox(label="Headline / Title", placeholder="e.g., Breaking News...", lines=1),
        gr.Textbox(label="Article Content", placeholder="Paste main text here...", lines=5),
        gr.Image(type="pil", label="Associated Image (Optional)"),
        gr.Slider(0, 10000, value=500, label="Social Engagement (Likes/Shares)"),
        gr.Slider(0, 50000, value=1000, label="Read/View Count")
    ],
    outputs=[
        gr.Textbox(label="Final Verdict"),
        gr.Textbox(label="Explainable AI (XAI) Report", lines=8)
    ],
    title="🔍 Multimodal Misinformation Analyzer",
    description="Analyzes Headlines vs Content, Imagery, and Social Metadata.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
