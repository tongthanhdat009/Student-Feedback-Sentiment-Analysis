"""
Streamlit Dashboard - Student Feedback Sentiment Analysis
==========================================================

Dashboard này hiển thị:
- Model Comparison (Accuracy/F1)
- Sentiment Distribution
- Aspect-wise Sentiment
- Demo Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Thêm thư mục gốc vào path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Page config
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Paths
SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "acsa"
RAW_DIR = ROOT_DIR / "data" / "raw" / "uit_vsfc"

# Labels
SENTIMENT_LABELS = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}
TOPIC_LABELS = {0: "Giảng viên", 1: "Chương trình đào tạo", 2: "Cơ sở vật chất", 3: "Khác"}


@st.cache_data
def load_model_results():
    """Load kết quả từ các models"""
    results = {}
    
    for model_name in ["svm", "lstm", "phobert"]:
        results_path = SAVED_MODELS_DIR / model_name / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results[model_name] = json.load(f)
    
    return results


@st.cache_data
def load_dataset():
    """Load dataset"""
    data = {}
    for split in ["train", "validation", "test"]:
        csv_path = RAW_DIR / f"{split}.csv"
        if csv_path.exists():
            data[split] = pd.read_csv(csv_path)
    return data


def plot_model_comparison(results):
    """Biểu đồ so sánh các models"""
    st.subheader("📊 Model Comparison")
    
    if not results:
        st.warning("Chưa có kết quả. Hãy train các models trước!")
        return
    
    # Chuẩn bị data
    model_names = []
    accuracies = []
    f1_scores = []
    
    for model, data in results.items():
        if "test" in data:
            model_names.append(model.upper())
            accuracies.append(data["test"].get("accuracy", 0))
            f1_scores.append(data["test"].get("f1", 0))
    
    if not model_names:
        st.warning("Không có dữ liệu test. Kiểm tra lại kết quả training.")
        return
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Thêm labels trên bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Bảng kết quả
    st.subheader("📋 Detailed Results")
    df_results = pd.DataFrame({
        "Model": model_names,
        "Accuracy": [f"{a:.4f}" for a in accuracies],
        "F1-Score": [f"{f:.4f}" for f in f1_scores]
    })
    st.dataframe(df_results, use_container_width=True)


def plot_sentiment_distribution(data):
    """Biểu đồ phân bố cảm xúc"""
    st.subheader("📈 Sentiment Distribution")
    
    if not data:
        st.warning("Không có dữ liệu!")
        return
    
    # Combine all splits
    all_df = pd.concat([df for df in data.values()])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        sentiment_counts = all_df['sentiment'].value_counts().sort_index()
        labels = [SENTIMENT_LABELS[i] for i in sentiment_counts.index]
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        
        ax1.pie(sentiment_counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90, explode=[0.02]*len(labels))
        ax1.set_title("Sentiment Distribution (Pie Chart)")
        st.pyplot(fig1)
    
    with col2:
        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        bars = ax2.bar(labels, sentiment_counts.values, color=colors)
        ax2.set_xlabel("Sentiment")
        ax2.set_ylabel("Count")
        ax2.set_title("Sentiment Distribution (Bar Chart)")
        
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig2)


def plot_aspect_sentiment(data):
    """Biểu đồ cảm xúc theo khía cạnh"""
    st.subheader("📊 Aspect-wise Sentiment Analysis")
    
    if not data:
        st.warning("Không có dữ liệu!")
        return
    
    # Combine all splits
    all_df = pd.concat([df for df in data.values()])
    
    # Cross-tabulation
    cross_tab = pd.crosstab(all_df['topic'], all_df['sentiment'], normalize='index')
    
    # Rename
    cross_tab.index = [TOPIC_LABELS.get(i, f"Topic {i}") for i in cross_tab.index]
    cross_tab.columns = [SENTIMENT_LABELS.get(i, f"Sentiment {i}") for i in cross_tab.columns]
    
    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    cross_tab.plot(kind='bar', stacked=True, ax=ax, 
                   color=['#e74c3c', '#f39c12', '#27ae60'])
    ax.set_xlabel("Topic/Aspect")
    ax.set_ylabel("Proportion")
    ax.set_title("Sentiment Distribution by Topic")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Heatmap
    st.subheader("🔥 Sentiment Heatmap by Topic")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    cross_tab_counts = pd.crosstab(all_df['topic'], all_df['sentiment'])
    cross_tab_counts.index = [TOPIC_LABELS.get(i, f"Topic {i}") for i in cross_tab_counts.index]
    cross_tab_counts.columns = [SENTIMENT_LABELS.get(i, f"Sentiment {i}") for i in cross_tab_counts.columns]
    
    sns.heatmap(cross_tab_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax2)
    ax2.set_title("Sentiment Count by Topic")
    plt.tight_layout()
    st.pyplot(fig2)


def demo_prediction():
    """Demo dự đoán sentiment"""
    st.subheader("🎮 Demo Prediction")
    
    # Load SVM model (fastest)
    svm_model_path = SAVED_MODELS_DIR / "svm" / "svm_sentiment.pkl"
    tfidf_path = PROCESSED_DIR / "tfidf_vectorizer.pkl"
    
    if not svm_model_path.exists() or not tfidf_path.exists():
        st.warning("⚠️ Model chưa được train. Hãy train SVM model trước!")
        st.code("python training/train_svm.py", language="bash")
        return
    
    try:
        with open(svm_model_path, "rb") as f:
            svm_model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return
    
    # Input
    st.write("Nhập phản hồi của sinh viên để phân tích cảm xúc:")
    
    # Sample texts
    samples = [
        "Giảng viên rất nhiệt tình và tận tâm với sinh viên",
        "Phòng học chật hẹp, máy chiếu hay bị hỏng",
        "Chương trình học bình thường, không có gì đặc biệt",
        "Thầy cô giảng bài khó hiểu, cần cải thiện phương pháp",
        "Thư viện có nhiều sách hay, không gian yên tĩnh"
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area("📝 Phản hồi:", value=samples[0], height=100)
    with col2:
        st.write("**Mẫu thử:**")
        for i, sample in enumerate(samples):
            if st.button(f"Mẫu {i+1}", key=f"sample_{i}"):
                user_input = sample
    
    if st.button("🔮 Phân tích", type="primary"):
        if user_input.strip():
            # Predict
            import re
            clean_text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', '', user_input.lower())
            X = tfidf.transform([clean_text])
            prediction = svm_model.predict(X)[0]
            proba = svm_model.predict_proba(X)[0]
            
            # Display
            st.write("---")
            st.subheader("📊 Kết quả phân tích:")
            
            col1, col2, col3 = st.columns(3)
            
            sentiment_colors = ["#e74c3c", "#f39c12", "#27ae60"]
            sentiment_emoji = ["😞", "😐", "😊"]
            
            for i, (col, label) in enumerate(zip([col1, col2, col3], ["Negative", "Neutral", "Positive"])):
                with col:
                    color = sentiment_colors[i]
                    prob = proba[i] * 100
                    emoji = sentiment_emoji[i]
                    
                    if i == prediction:
                        st.markdown(f"""
                        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
                            <h2>{emoji}</h2>
                            <h3>{label}</h3>
                            <h1>{prob:.1f}%</h1>
                            <p>✓ PREDICTED</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color:#ecf0f1; padding:20px; border-radius:10px; text-align:center;">
                            <h2>{emoji}</h2>
                            <h3>{label}</h3>
                            <h1>{prob:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhập văn bản!")


def main():
    """Main dashboard"""
    st.title("📊 Student Feedback Sentiment Analysis")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio(
        "Chọn trang:",
        ["🏠 Overview", "📊 Model Comparison", "📈 Sentiment Distribution", 
         "📊 Aspect Analysis", "🎮 Demo Prediction"]
    )
    
    # Load data
    results = load_model_results()
    data = load_dataset()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Status")
    st.sidebar.write(f"Models trained: {len(results)}")
    if data:
        total_samples = sum(len(df) for df in data.values())
        st.sidebar.write(f"Total samples: {total_samples}")
    
    # Pages
    if page == "🏠 Overview":
        st.header("🎓 Phân tích cảm xúc phản hồi sinh viên")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 Models Trained", len(results))
        with col2:
            if data:
                st.metric("📊 Total Samples", sum(len(df) for df in data.values()))
        with col3:
            if results:
                best_model = max(results.keys(), 
                               key=lambda k: results[k].get("test", {}).get("accuracy", 0))
                best_acc = results[best_model].get("test", {}).get("accuracy", 0)
                st.metric("🏆 Best Accuracy", f"{best_acc:.2%}", best_model.upper())
        
        st.markdown("---")
        st.markdown("""
        ### 📋 Tính năng Dashboard
        
        - **Model Comparison**: So sánh hiệu suất giữa SVM, LSTM và PhoBERT
        - **Sentiment Distribution**: Phân bố cảm xúc Positive/Neutral/Negative
        - **Aspect Analysis**: Phân tích cảm xúc theo từng khía cạnh (Giảng viên, CTDT, CSVC...)
        - **Demo Prediction**: Thử nghiệm dự đoán cảm xúc với văn bản mới
        
        ### 🚀 Quick Start
        
        ```bash
        # 1. Train models
        python training/train_svm.py
        python training/train_lstm.py
        python training/train_phobert.py
        
        # 2. Run dashboard
        streamlit run dashboard/app.py
        ```
        """)
    
    elif page == "📊 Model Comparison":
        plot_model_comparison(results)
    
    elif page == "📈 Sentiment Distribution":
        plot_sentiment_distribution(data)
    
    elif page == "📊 Aspect Analysis":
        plot_aspect_sentiment(data)
    
    elif page == "🎮 Demo Prediction":
        demo_prediction()


if __name__ == "__main__":
    main()
