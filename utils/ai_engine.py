from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def generate_local_answer(prompt, max_tokens=300):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def ask_ai_about_data(question, context):
    prompt = f"""
You are an enterprise business analyst.

Analyze the dataset context and answer the user question.

Dataset Context:
{context}

Question:
{question}

Answer format:
- Summary
- Key insights
- Business meaning
- Recommendations
"""

    try:
        answer = generate_local_answer(prompt)

        if len(answer.strip()) < 40:
            return f"""
### Dataset-Based Answer

The uploaded dataset has been analyzed using the available context.

### User Question
{question}

### Key Context
{context[:2000]}

### Recommendation
Review the column structure, missing values, numerical summaries, and categorical distributions to identify useful business trends.
"""

        return answer

    except Exception as e:
        return f"Local AI Error: {e}"


def generate_business_insights(context):
    prompt = f"""
You are a senior enterprise business intelligence consultant.

Analyze this dataset/document context.

Context:
{context}

Generate:
- Executive Summary
- Key Insights
- Risks
- Opportunities
- Recommendations
"""

    try:
        answer = generate_local_answer(prompt)

        if len(answer.strip()) < 40:
            return f"""
### Executive Summary

The uploaded data/document has been processed successfully.

### Key Insights
- The available context can be used for business analysis.
- The dataset/document contains structured information useful for decision-making.
- Further analysis should focus on trends, missing values, outliers, and business KPIs.

### Risks
- Missing or inconsistent values may affect decision quality.
- Limited context may reduce the depth of automated insights.

### Recommendations
- Review data quality before modeling.
- Use dashboards to explore patterns.
- Apply AutoML if a prediction target is available.
"""

        return answer

    except Exception as e:
        return f"Local AI Error: {e}"