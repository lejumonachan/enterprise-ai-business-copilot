import streamlit as st
import pandas as pd

from utils.file_loader import load_file, get_dataframe_summary
from utils.ai_engine import ask_ai_about_data, generate_business_insights
from utils.visualization import (
    get_column_types,
    create_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_histogram,
    create_box_plot,
    create_scatter_plot,
    create_heatmap
)
from utils.automl_engine import run_automl, predict_user_input
from utils.pdf_generator import generate_pdf_report
from utils.helpers import dataframe_to_context, ensure_directories
from utils.rag_engine import chunk_text, build_faiss_index, rag_answer

ensure_directories()

st.set_page_config(
    page_title="Enterprise AI Business Copilot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f5f3ff 100%);
    color: #0f172a;
}

.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 45%, #1e1b4b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 8px 0 30px rgba(15,23,42,0.35);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.sidebar-brand {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}

.sidebar-brand h2 {
    margin: 0;
    font-size: 24px;
}

.sidebar-brand p {
    margin-top: 8px;
    font-size: 14px;
    color: #cbd5e1;
}

section[data-testid="stSidebar"] label {
    background: rgba(255,255,255,0.06);
    padding: 12px 14px;
    border-radius: 14px;
    margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] label:hover {
    background: rgba(59,130,246,0.25);
}

.hero {
    background: linear-gradient(135deg, #020617 0%, #2563eb 45%, #7c3aed 100%);
    padding: 46px;
    border-radius: 32px;
    color: white;
    box-shadow: 0 28px 70px rgba(37,99,235,0.35);
    margin-bottom: 32px;
    border: 1px solid rgba(255,255,255,0.18);
}

.hero-title {
    font-size: 50px;
    font-weight: 900;
    letter-spacing: -1.4px;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.92;
    margin-top: 14px;
    max-width: 950px;
    line-height: 1.6;
}

.glass-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(18px);
    padding: 28px;
    border-radius: 26px;
    border: 1px solid rgba(226,232,240,0.95);
    box-shadow: 0 14px 40px rgba(15,23,42,0.08);
    margin-bottom: 22px;
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    padding: 26px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 34px rgba(15,23,42,0.09);
}

.metric-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 700;
}

.metric-value {
    color: #0f172a;
    font-size: 30px;
    font-weight: 900;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border-radius: 14px;
    border: none;
    font-weight: 800;
    padding: 0.75rem 1.45rem;
    box-shadow: 0 10px 25px rgba(37,99,235,0.35);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    color: white;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #020617, #334155);
    color: white;
    border-radius: 14px;
    border: none;
    font-weight: 800;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(15,23,42,0.08);
}

textarea {
    border-radius: 18px !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-brand">
    <h2>🤖 AI Copilot</h2>
    <p>Enterprise AI Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload & Preview",
        "AI Chat With Data",
        "Business Dashboard",
        "AutoML Engine",
        "Executive Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.markdown("**Leju Monachan**")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/leju-monachan757/)")
st.sidebar.markdown("[💻 GitHub](https://github.com/lejumonachan)")

st.markdown("""
<div class="hero">
    <div class="hero-title">Enterprise AI Business Copilot</div>
    <div class="hero-subtitle">
        A premium AI-powered analytics workspace for document intelligence, local RAG,
        business dashboards, AutoML modeling, and executive report generation.
    </div>
</div>
""", unsafe_allow_html=True)

if page == "Home":
    st.markdown("## 🚀 Platform Overview")

    c1, c2, c3, c4 = st.columns(4)

    items = [
        ("AI Chat", "Data + PDF"),
        ("RAG Engine", "Local FAISS"),
        ("ML Engine", "AutoML"),
        ("Reports", "PDF Export")
    ]

    for col, item in zip([c1, c2, c3, c4], items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{item[0]}</div>
                <div class="metric-value">{item[1]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>✨ Enterprise Capabilities</h3>
        <ul>
            <li>Upload CSV, Excel, and PDF files</li>
            <li>Analyze business datasets with interactive dashboards</li>
            <li>Chat with uploaded datasets and PDF documents</li>
            <li>Use local RAG-powered PDF intelligence with FAISS</li>
            <li>Run AutoML model comparison and live prediction</li>
            <li>Generate downloadable executive PDF reports</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Upload & Preview":
    st.markdown("## 📂 Upload & Preview")

    st.markdown("""
    <div class="glass-card">
        Upload CSV, Excel, or PDF files for enterprise analysis.
        CSV/Excel files support dashboards and AutoML. PDFs support local RAG intelligence.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["csv", "xlsx", "xls", "pdf"]
    )

    if uploaded_file:
        file_type, data = load_file(uploaded_file)

        if file_type == "dataframe":
            st.success("✅ Data file uploaded successfully")

            st.session_state["uploaded_data"] = data
            st.session_state["file_type"] = "dataframe"

            summary = get_dataframe_summary(data)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", summary["rows"])
            c2.metric("Columns", summary["columns"])
            c3.metric("Missing Values", summary["missing_values"])
            c4.metric("Duplicate Rows", summary["duplicate_rows"])

            st.markdown("### 📄 Data Preview")
            st.dataframe(data.head(20), use_container_width=True)

            st.markdown("### 📊 Statistical Summary")
            st.dataframe(data.describe(include="all"), use_container_width=True)

        elif file_type == "pdf":
            st.success("✅ PDF uploaded and text extracted successfully")

            st.session_state["uploaded_text"] = data
            st.session_state["file_type"] = "pdf"

            if data.strip():
                try:
                    with st.spinner("Building local RAG vector index..."):
                        chunks = chunk_text(data)
                        index, stored_chunks = build_faiss_index(chunks)

                    st.session_state["rag_index"] = index
                    st.session_state["rag_chunks"] = stored_chunks

                    st.success("✅ RAG index created successfully")

                except Exception as e:
                    st.error(f"RAG Index Error: {e}")

            st.markdown("### 📄 Extracted PDF Text Preview")
            st.text_area("PDF Text", data[:5000], height=300)
            st.metric("Extracted Characters", len(data))

        else:
            st.error("Unsupported file type")

elif page == "AI Chat With Data":
    st.markdown("## 💬 AI Chat With Data")

    st.markdown("""
    <div class="glass-card">
        Ask natural language business questions about your uploaded dataset or PDF.
        For PDF documents, the system uses local RAG retrieval before answering.
    </div>
    """, unsafe_allow_html=True)

    if "file_type" not in st.session_state:
        st.warning("Please upload a CSV, Excel, or PDF first from Upload & Preview.")

    else:
        file_type = st.session_state["file_type"]

        if file_type == "dataframe":
            df = st.session_state["uploaded_data"]
            st.success("Dataset is ready for AI chat.")
            context = dataframe_to_context(df)

        else:
            st.success("PDF is ready for RAG-based AI chat.")
            context = st.session_state["uploaded_text"][:12000]

        question = st.text_input(
            "Ask a question",
            placeholder="Example: Summarize this dataset"
        )

        if st.button("Ask AI"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("AI is analyzing..."):
                    if file_type == "pdf" and "rag_index" in st.session_state:
                        answer = rag_answer(
                            question,
                            st.session_state["rag_index"],
                            st.session_state["rag_chunks"]
                        )
                    else:
                        answer = ask_ai_about_data(question, context)

                st.markdown("### 🤖 AI Answer")
                st.markdown(answer)

elif page == "Business Dashboard":
    st.markdown("## 📊 Business Dashboard")

    if "file_type" not in st.session_state or st.session_state["file_type"] != "dataframe":
        st.warning("Please upload a CSV or Excel file first from Upload & Preview.")

    else:
        df = st.session_state["uploaded_data"]
        numeric_cols, categorical_cols, all_cols = get_column_types(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.markdown("### 📄 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        if len(numeric_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            tabs = st.tabs([
                "Bar Chart",
                "Line Chart",
                "Pie Chart",
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Heatmap"
            ])

            with tabs[0]:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("X-axis", all_cols, key="bar_x")
                y_col = c2.selectbox("Y-axis", [None] + numeric_cols, key="bar_y")
                color_col = st.selectbox("Color Group", [None] + all_cols, key="bar_color")
                st.plotly_chart(create_bar_chart(df, x_col, y_col, color_col), use_container_width=True)

            with tabs[1]:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("X-axis", all_cols, key="line_x")
                y_col = c2.selectbox("Y-axis", numeric_cols, key="line_y")
                color_col = st.selectbox("Color Group", [None] + all_cols, key="line_color")
                st.plotly_chart(create_line_chart(df, x_col, y_col, color_col), use_container_width=True)

            with tabs[2]:
                pie_col = st.selectbox("Category Column", all_cols, key="pie_col")
                st.plotly_chart(create_pie_chart(df, pie_col), use_container_width=True)

            with tabs[3]:
                hist_col = st.selectbox("Histogram Column", numeric_cols, key="hist_col")
                bins = st.slider("Bins", 10, 100, 30)
                st.plotly_chart(create_histogram(df, hist_col, bins), use_container_width=True)

            with tabs[4]:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("X-axis Optional", [None] + all_cols, key="box_x")
                y_col = c2.selectbox("Y-axis", numeric_cols, key="box_y")
                st.plotly_chart(create_box_plot(df, y_col, x_col), use_container_width=True)

            with tabs[5]:
                c1, c2 = st.columns(2)
                x_col = c1.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = c2.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color_col = st.selectbox("Color Group", [None] + all_cols, key="scatter_color")
                st.plotly_chart(create_scatter_plot(df, x_col, y_col, color_col), use_container_width=True)

            with tabs[6]:
                if len(numeric_cols) > 1:
                    st.plotly_chart(create_heatmap(df, numeric_cols), use_container_width=True)
                else:
                    st.info("Need at least two numeric columns.")

elif page == "AutoML Engine":
    st.markdown("## 🤖 AutoML Engine")

    if "file_type" not in st.session_state or st.session_state["file_type"] != "dataframe":
        st.warning("Please upload a CSV or Excel file first.")

    else:
        df = st.session_state["uploaded_data"]

        target = st.selectbox("Select Target Column", df.columns)
        numeric_strategy = st.selectbox("Numeric Missing Value Strategy", ["mean", "median"])
        feature_k = st.selectbox("Feature Selection", ["all", 5, 10, 15, 20])

        if "automl_output" not in st.session_state:
            st.session_state["automl_output"] = None

        if st.button("🚀 Run AutoML Pipeline"):
            try:
                with st.spinner("Training and comparing models..."):
                    output = run_automl(
                        df=df,
                        target_column=target,
                        numeric_strategy=numeric_strategy,
                        feature_k=feature_k
                    )

                st.session_state["automl_output"] = output
                st.success("✅ AutoML completed successfully")

            except Exception as e:
                st.error(f"AutoML Error: {e}")

        if st.session_state["automl_output"] is not None:
            output = st.session_state["automl_output"]

            st.markdown("### Detected Problem Type")
            st.info(output["problem_type"])

            st.markdown("### 📊 Model Comparison")
            st.dataframe(output["results"], use_container_width=True)

            st.markdown("### 🏆 Best Model")
            st.success(output["best_model_name"])

            pred_df = pd.DataFrame({
                "Actual": output["actual"],
                "Predicted": output["predictions"]
            })

            st.markdown("### 🔮 Sample Predictions")
            st.dataframe(pred_df, use_container_width=True)

            st.divider()
            st.markdown("## 🚀 Live Prediction Console")

            user_input = {}
            X_sample = output["X_sample"]
            feature_columns = output["feature_columns"]

            for col in feature_columns:
                if pd.api.types.is_numeric_dtype(X_sample[col]):
                    user_input[col] = st.number_input(col, value=float(X_sample[col].mean()))
                else:
                    options = X_sample[col].dropna().astype(str).unique().tolist()
                    if len(options) == 0:
                        options = ["Unknown"]
                    user_input[col] = st.selectbox(col, options)

            if st.button("🎯 Predict Target Value"):
                try:
                    prediction = predict_user_input(
                        model=output["best_model"],
                        input_data=user_input,
                        feature_columns=feature_columns,
                        target_encoder=output["target_encoder"]
                    )
                    st.success(f"✅ Predicted {target}: {prediction}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

elif page == "Executive Report":
    st.markdown("## 📄 Executive Report")

    if "file_type" not in st.session_state:
        st.warning("Please upload a file first.")

    else:
        if st.session_state["file_type"] == "dataframe":
            df = st.session_state["uploaded_data"]
            context = dataframe_to_context(df)
        else:
            context = st.session_state["uploaded_text"][:12000]

        if st.button("Generate Executive Report"):
            with st.spinner("Generating report..."):
                insights = generate_business_insights(context)

            st.markdown("### 🧠 Executive Insights")
            st.markdown(insights)

            report_path = generate_pdf_report(
                title="Enterprise AI Business Copilot Report",
                content=insights
            )

            with open(report_path, "rb") as file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=file,
                    file_name="executive_report.pdf",
                    mime="application/pdf"
                )