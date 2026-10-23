
import re
import random
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="P&L Financial Chatbot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 P&L Financial Assistant")
st.caption(
    "Your conversational assistant for revenue, expenses, "
    "profitability, and account balances. No AI API required."
)


# =========================================================
# 2. LOAD EXCEL DATA
# =========================================================

FILE_PATH = Path(__file__).parent / "sample_pnl.xlsx"

try:
    df = pd.read_excel(FILE_PATH)

except FileNotFoundError:
    st.error(
        "I couldn't find sample_pnl.xlsx. "
        "Please add it to the same folder as app.py."
    )
    st.stop()

except Exception as exc:
    st.error(f"Unable to read the Excel file: {exc}")
    st.stop()


required_columns = {"Account", "Amount"}

if not required_columns.issubset(df.columns):
    st.error(
        "Your Excel file must contain these columns: "
        "'Account' and 'Amount'."
    )
    st.stop()


df = df.copy()

df["Account"] = (
    df["Account"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Amount"] = pd.to_numeric(
    df["Amount"], errors="coerce"
)

df = df.dropna(subset=["Amount"])
df = df[df["Account"] != ""]

if df.empty:
    st.warning("No valid financial records were found.")
    st.stop()


# =========================================================
# 3. ACCOUNT CLASSIFICATION
# =========================================================

REVENUE_RE = re.compile(
    r"\b(revenue|sales|turnover|service income|"
    r"sales income|sales returns?|discounts? given|"
    r"sales discounts?|allowances)\b",
    re.I,
)

CONTRA_REVENUE_RE = re.compile(
    r"\b(sales returns?|returns and allowances|"
    r"discounts? given|sales discounts?)\b",
    re.I,
)

COGS_RE = re.compile(
    r"\b(cost of sales|cost of goods sold|cogs)\b",
    re.I,
)

EXPENSE_RE = re.compile(
    r"\b(expense|expenses|salary|salaries|wages|"
    r"rent|utilities|marketing|advertising|insurance|"
    r"depreciation|professional fees|office supplies|"
    r"travel|repairs|maintenance|freight|payroll|"
    r"interest expense|bank charges|commission expense)\b",
    re.I,
)


def classify_account(account):
    name = account.lower()

    if CONTRA_REVENUE_RE.search(name):
        return "Contra-revenue"

    if COGS_RE.search(name):
        return "Cost of sales"

    if REVENUE_RE.search(name):
        return "Revenue"

    if EXPENSE_RE.search(name):
        return "Expense"

    return "Other"


df["Category"] = df["Account"].apply(classify_account)


# =========================================================
# 4. FINANCIAL CALCULATIONS
# =========================================================

# Default assumptions:
# Revenue and contra-revenue are stored as signed amounts.
# Revenue is normally positive; returns may be negative.
# Expenses and cost of sales are normally negative.
#
# If your Excel uses a different sign convention, change
# these functions to match your source accounting system.


def revenue_total():
    """Net revenue, including signed contra-revenue."""
    return df.loc[
        df["Category"].isin(
            ["Revenue", "Contra-revenue"]
        ),
        "Amount",
    ].sum()


def expense_total():
    """Return classified expenses as a positive figure."""
    return -df.loc[
        df["Category"].isin(
            ["Expense", "Cost of sales"]
        ),
        "Amount",
    ].sum()


def cost_of_sales():
    return -df.loc[
        df["Category"] == "Cost of sales",
        "Amount",
    ].sum()


def net_income():
    """Sum all source rows; validate the P&L scope first."""
    return df["Amount"].sum()


def gross_profit():
    return revenue_total() - cost_of_sales()


def money(value):
    return f"${value:,.2f}"


def percent(value):
    return f"{value:.2f}%"


def net_margin():
    revenue = revenue_total()

    if revenue == 0:
        return None

    return net_income() / revenue * 100


# =========================================================
# 5. NATURAL-LANGUAGE NORMALIZATION
# =========================================================

def normalize(text):
    text = str(text).lower().strip()
    text = text.replace("&", " and ")

    text = re.sub(
        r"\bwhat's\b|\bwhats\b",
        "what is",
        text,
    )

    text = re.sub(
        r"\bhow's\b|\bhows\b",
        "how is",
        text,
    )

    text = re.sub(r"[$,?!]", " ", text)

    replacements = {
        r"\bmoney coming in\b": "revenue",
        r"\bsales income\b": "revenue",
        r"\bturnover\b": "revenue",
        r"\bbottom line\b": "net income",
        r"\bnet profit\b": "net income",
        r"\bbusiness profit\b": "net income",
        r"\bprofit margin\b": "net margin",
        r"\bhow much did we spend\b": "total expenses",
        r"\bwhere is our money going\b": "expense breakdown",
        r"\bbiggest costs\b": "top expenses",
        r"\blargest costs\b": "top expenses",
        r"\bdoing financially\b": "financial performance",
        r"\bshow me the top three\b": "show me the top 3",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return re.sub(r"\s+", " ", text).strip()


# =========================================================
# 6. INTENT DETECTION
# =========================================================

def detect_intent(question):
    q = normalize(question)

    # Specific intents must be checked before general intents.

    if re.search(
        r"\b(top|largest|biggest|highest|"
        r"most expensive|rank|ranking)\b",
        q,
    ) and re.search(
        r"\b(expense|expenses|account|cost|spend)\b",
        q,
    ):
        return "top_expenses"

    if re.search(
        r"\b(gross profit|gross margin)\b", q
    ):
        return "gross_profit"

    if re.search(
        r"\b(net margin|margin percentage|"
        r"net profit margin|profitability ratio)\b",
        q,
    ):
        return "profit_margin"

    if re.search(
        r"\b(net income|net profit|overall profit|"
        r"business profit|profitability|"
        r"are we profitable|is the business profitable|"
        r"making a profit|losing money|"
        r"negative profit|break even|break-even)\b",
        q,
    ) or q in {"profit", "profit?"}:
        return "net_income"

    if re.search(
        r"\b(expense breakdown|expense categories|"
        r"breakdown|distribution|by category|"
        r"compare categories)\b",
        q,
    ):
        return "breakdown"

    if re.search(
        r"\b(summarize|summary|overview|analyze|analyse|"
        r"financial performance|business performance|"
        r"how is the business doing|"
        r"how are we doing|explain the pnl|"
        r"explain the p and l)\b",
        q,
    ):
        return "summary"

    if re.search(
        r"\b(total expenses|all expenses|"
        r"operating expenses|total expense|"
        r"expense total|overall expenses|"
        r"how much did we spend)\b",
        q,
    ):
        return "expenses"

    if re.search(
        r"\b(total revenue|net revenue|revenue|"
        r"sales|turnover|income from sales)\b",
        q,
    ):
        return "revenue"

    if re.search(r"\b(hello|hi|hey|good morning|"
                 r"good afternoon|good evening)\b", q):
        return "greeting"

    if re.search(
        r"\b(thank you|thanks|thank you very much)\b",
        q,
    ):
        return "thanks"

    return "account"


# =========================================================
# 7. ACCOUNT MATCHING
# =========================================================

account_names = df["Account"].drop_duplicates().tolist()

account_text = [
    name.lower() + " " +
    re.sub(r"[^a-z0-9 ]", " ", name.lower())
    for name in account_names
]

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1,
)

account_matrix = vectorizer.fit_transform(account_text)


def find_account(question):
    q = normalize(question)

    # Remove common conversational words before matching.
    q_clean = re.sub(r"[^a-z0-9 ]", " ", q)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    ignored_words = {
        "what", "is", "the", "our", "my", "how",
        "much", "about", "please", "show", "me",
        "tell", "give", "account", "amount", "balance",
        "in", "for", "of", "do", "we", "have",
        "can", "you", "check", "look", "up",
    }

    meaningful_words = [
        word for word in q_clean.split()
        if word not in ignored_words
    ]

    search_text = " ".join(meaningful_words)

    if not search_text:
        return None, 0.0, []

    # Exact matching first.
    exact = [
        name for name in account_names
        if name.lower() == search_text
    ]

    if exact:
        return exact[0], 1.0, exact

    # Partial matching.
    partial = [
        name for name in account_names
        if len(search_text) >= 4
        and search_text in name.lower()
    ]

    if len(partial) == 1:
        return partial[0], 0.95, partial

    # Fuzzy matching using TF-IDF.
    scores = cosine_similarity(
        vectorizer.transform([search_text]),
        account_matrix,
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    best_index = ranked_indices[0]
    best_score = float(scores[best_index])

    candidates = [
        account_names[i]
        for i in ranked_indices[:3]
        if scores[i] >= 0.15
    ]

    # Similarity scores are not calibrated probabilities.
    return (
        account_names[best_index],
        best_score,
        candidates,
    )


# =========================================================
# 8. EXPENSE ANALYSIS
# =========================================================

def top_expense_accounts(limit=5):
    expenses = df[
        df["Category"].isin(
            ["Expense", "Cost of sales"]
        )
    ].copy()

    if expenses.empty:
        return expenses

    # Group repeated account names before ranking.
    expenses["Expense amount"] = -expenses["Amount"]

    expenses = (
        expenses.groupby("Account", as_index=False)
        .agg({"Expense amount": "sum"})
        .sort_values(
            "Expense amount",
            ascending=False,
        )
    )

    return expenses.head(limit)


# =========================================================
# 9. CONVERSATIONAL RESPONSES
# =========================================================

def conversational_answer(intent):
    revenue = revenue_total()
    expenses = expense_total()
    profit = net_income()

    if intent == "greeting":
        return (
            "Hi there! 👋 I'm your P&L assistant. "
            "What would you like to explore today? "
            "I can help with revenue, expenses, profit, "
            "margins, and individual accounts."
        )

    if intent == "thanks":
        return (
            "You're welcome! 😊 Feel free to ask me "
            "anything else about your P&L."
        )

    if intent == "revenue":
        return random.choice([
            f"Your net classified revenue is {money(revenue)}.",
            f"Looking at the P&L, net revenue comes to {money(revenue)}.",
            f"We have {money(revenue)} in net classified revenue.",
        ])

    if intent == "expenses":
        return random.choice([
            f"Your classified expenses total {money(expenses)}.",
            f"Based on the current data, expenses come to {money(expenses)}.",
            f"We've recorded {money(expenses)} in classified expenses.",
        ])

    if intent == "net_income":
        if profit > 0:
            explanation = (
                "The overall source balance is positive, "
                "assuming the workbook contains the relevant "
                "P&L accounts and consistent accounting signs."
            )
        elif profit < 0:
            explanation = (
                "The overall source balance is negative. "
                "It is worth checking the expense and revenue "
                "signs and confirming that all relevant accounts "
                "are included."
            )
        else:
            explanation = (
                "The overall source balance is zero. "
                "This indicates break-even only if the workbook "
                "contains the complete P&L."
            )

        return (
            f"Your current net P&L balance is {money(profit)}.\n\n"
            f"{explanation}\n\n"
            "Would you like to look at the biggest expenses next?"
        )

    if intent == "profit_margin":
        margin = net_margin()

        if margin is None:
            return (
                "I can't calculate the net margin because "
                "classified revenue is zero."
            )

        return (
            f"Your estimated net margin is {percent(margin)}.\n\n"
            f"That's calculated as the net P&L balance "
            f"({money(profit)}) divided by classified revenue "
            f"({money(revenue)}).\n\n"
            "This estimate assumes your workbook contains the "
            "relevant P&L accounts and consistent accounting signs."
        )

    if intent == "gross_profit":
        gross = gross_profit()

        if revenue == 0:
            margin_text = "Not available because revenue is zero."
        else:
            margin_text = percent(gross / revenue * 100)

        return (
            f"Your estimated gross profit is {money(gross)}.\n\n"
            f"Your estimated gross margin is {margin_text}.\n\n"
            "I calculated gross profit as net classified revenue "
            "less classified cost of sales. Please verify the "
            "account classifications before relying on this figure."
        )

    if intent == "top_expenses":
        top = top_expense_accounts(5)

        if top.empty:
            return (
                "I couldn't identify any classified expense accounts. "
                "You may need to update the account classification rules."
            )

        lines = [
            "Here are your five largest classified expense accounts:"
        ]

        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                f"{rank}. **{row['Account']}** — "
                f"{money(row['Expense amount'])}"
            )

        lines.append(
            "\nThese amounts assume expenses are stored as negative "
            "numbers in your source workbook."
        )

        return "\n\n".join(lines)

    if intent == "breakdown":
        grouped = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        lines = ["Here's the P&L breakdown by account category:"]

        for category, amount in grouped.items():
            lines.append(
                f"- **{category}:** {money(amount)}"
            )

        lines.append(
            "\nCategories are inferred from account names. "
            "Review the 'Other' category and check your account "
            "classifications before using this breakdown."
        )

        return "\n\n".join(lines)

    if intent == "summary":
        margin = net_margin()

        lines = [
            "Here's a quick overview of your P&L:",
            f"- **Net classified revenue:** {money(revenue)}",
            f"- **Classified expenses:** {money(expenses)}",
            f"- **Net P&L balance:** {money(profit)}",
        ]

        if margin is not None:
            lines.append(
                f"- **Estimated net margin:** {percent(margin)}"
            )

        top = top_expense_accounts(3)

        if not top.empty:
            names = ", ".join(top["Account"].tolist())
            lines.append(
                f"\nYour largest classified expense accounts "
                f"include {names}."
            )

        lines.append(
            "\nKeep in mind that these figures depend on your "
            "account classification and source sign convention."
        )

        return "\n\n".join(lines)

    return None


# =========================================================
# 10. CONTEXT AND FOLLOW-UP QUESTIONS
# =========================================================

def resolve_followup(question):
    q = normalize(question)
    previous_intent = st.session_state.get("last_intent")

    # Short follow-ups inherit the previous topic.
    if re.search(
        r"\b(show me the top 3|show me the top three|"
        r"show me more|tell me more|break it down|"
        r"break that down|give me details|"
        r"which ones|yes please|go ahead)\b",
        q,
    ):
        if previous_intent in {"expenses", "summary"}:
            return "top_expenses"

    if re.search(
        r"\b(what about expenses|and expenses|"
        r"how about expenses)\b",
        q,
    ):
        return "expenses"

    if re.search(
        r"\b(what about revenue|and revenue|"
        r"how about revenue)\b",
        q,
    ):
        return "revenue"

    if re.search(
        r"\b(what about profit|and profit|"
        r"how about profit)\b",
        q,
    ):
        return "net_income"

    if re.search(
        r"\b(top 3|top three|top 5|top five)\b",
        q,
    ) and previous_intent in {"expenses", "summary"}:
        return "top_expenses"

    return None


# =========================================================
# 11. MAIN QUESTION ANSWER FUNCTION
# =========================================================

def answer_question(question):
    followup_intent = resolve_followup(question)

    if followup_intent:
        intent = followup_intent
    else:
        intent = detect_intent(question)

    st.session_state.last_intent = intent

    # Friendly small talk and known financial intents.
    answer = conversational_answer(intent)

    if answer is not None:
        return answer

    # Search for an account.
    account, score, candidates = find_account(question)

    if account is None or score < 0.20:
        return (
            "I'm not quite sure which account you're asking about. "
            "Could you try using the account name from your Excel file?\n\n"
            "You can also ask about revenue, expenses, net profit, "
            "gross profit, margin, or financial performance."
        )

    if score < 0.35:
        if candidates:
            suggestions = "\n".join(
                f"- {name}" for name in candidates
            )

            return (
                f"I found a possible match, **{account}**, "
                "but I'm not confident enough to assume it's correct.\n\n"
                f"Here are a few accounts you might mean:\n{suggestions}\n\n"
                "Could you clarify which one you're looking for?"
            )

        return (
            "I found a possible match, but I'm not confident "
            "enough to give you a reliable figure. "
            "Please try the exact account name."
        )

    matches = df[df["Account"] == account]

    amount = matches["Amount"].sum()
    category = matches["Category"].iloc[0]

    return (
        f"I found **{account}** in your P&L.\n\n"
        f"- **Amount:** {money(amount)}\n"
        f"- **Category:** {category}\n"
        f"- **Rows found:** {len(matches)}\n\n"
        "This amount is the sum of the matching source rows, "
        "using their original signs."
    )


# =========================================================
# 12. CHAT INTERFACE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None


# Sidebar
with st.sidebar:
    st.subheader("💬 Try asking")

    examples = [
        "How much revenue did we make?",
        "Are we profitable?",
        "What are our biggest expenses?",
        "What is our net profit margin?",
        "Calculate gross profit",
        "Summarize financial performance",
        "Show the expense breakdown",
        "How much did we spend on salaries?",
    ]

    for example in examples:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_intent = None
        st.session_state.pending_question = None
        st.rerun()

    st.caption(
        "Answers are generated locally using Python rules. "
        "No OpenAI API is used."
    )


# First-time greeting
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi there! 👋 I'm your P&L financial assistant.\n\n"
            "I can help you understand your revenue, expenses, "
            "profitability, and account balances.\n\n"
            "What would you like to explore first?"
        )


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process a question from either the chat box or a sidebar example.
typed_question = st.chat_input(
    "Ask me about your P&L...",
    key="chat_input",
)

pending_question = st.session_state.pop(
    "pending_question", None
)

question = typed_question or pending_question

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking into your P&L..."):
            try:
                answer = answer_question(question)
            except Exception:
                answer = (
                    "Sorry, I couldn't process that question. "
                    "Please try rephrasing it or asking about "
                    "revenue, expenses, profit, or an account name."
                )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })


# =========================================================
# 13. DATA INSPECTION
# =========================================================

with st.expander("📂 View loaded P&L data"):
    st.dataframe(df, use_container_width=True)

    st.caption(f"{len(df)} financial rows loaded.")

    st.subheader("Account classification review")

    classification_summary = (
        df.groupby("Category")
        .agg(
            Account_count=("Account", "nunique"),
            Row_count=("Account", "size"),
            Source_amount=("Amount", "sum"),
        )
        .reset_index()
    )

    st.dataframe(
        classification_summary,
        use_container_width=True,
    )

    unclassified = df[df["Category"] == "Other"]

    if not unclassified.empty:
        st.warning(
            f"{unclassified['Account'].nunique()} unique account(s) "
            "are classified as Other. Review these accounts before "
            "relying on the financial summaries."
        )

        st.dataframe(
            unclassified[["Account", "Amount"]],
            use_container_width=True,
        )
