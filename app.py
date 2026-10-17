import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("P&L Financial Chatbot")

# Read existing Excel file (never create it automatically)
try:
    df = pd.read_excel("sample_pnl.xlsx")
    df["Account"] = df["Account"].astype(str)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
except FileNotFoundError:
    st.error("Upload sample_pnl.xlsx to your GitHub repository.")
    st.stop()

# Machine learning model
model = TfidfVectorizer()
X = model.fit_transform(df["Account"])

question = st.text_input("Ask about your P&L")

if question:
    q = question.lower()

    if "net income" in q or "net profit" in q or q == "profit":
        answer = f"Net income: ${df['Amount'].sum():,.2f}"

    elif "revenue" in q or "sales" in q:
        mask = df["Account"].str.contains(
            r"Revenue|Sales Returns|Discounts Given",
            case=False, regex=True
        )
        answer = f"Net revenue: ${df.loc[mask, 'Amount'].sum():,.2f}"

    elif "total expenses" in q or "all expenses" in q:
        mask = ~df["Account"].str.contains(
            r"Revenue|Sales Returns|Discounts Given",
            case=False, regex=True
        )
        answer = f"Total expenses: ${-df.loc[mask, 'Amount'].sum():,.2f}"

    else:
        similarity = cosine_similarity(
            model.transform([question]), X
        ).flatten()
        i = similarity.argmax()

        if similarity[i] < 0.1:
            answer = "Sorry, I couldn't find a matching P&L item."
        else:
            answer = (
                f"{df.iloc[i]['Account']}: "
                f"${df.iloc[i]['Amount']:,.2f}"
            )

    st.write(answer)
