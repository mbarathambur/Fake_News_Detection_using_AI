import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- MODEL PATH ----------------
MODEL_PATH = "fake_news_bert_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- CSS ----------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.result {
    padding: 20px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
}
.real {
    background-color: #e6ffe6;
    color: #006600;
}
.fake {
    background-color: #ffe6e6;
    color: #b30000;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown("<div class='title'>📰 Fake News Detection</div>", unsafe_allow_html=True)

news_text = st.text_area(
    "Enter News Article Text",
    height=200,
    placeholder="Paste the news content here..."
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        confidence = confidence.item() * 100

        # ✅ CORRECT LABEL MAPPING
        # 0 → REAL, 1 → FAKE
        if prediction.item() == 0:
            st.markdown(
                f"<div class='result real'>✅ Real News<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result fake'>🚨 Fake News<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
