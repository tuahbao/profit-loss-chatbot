
import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="P&L Financial Chatbot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 P&L Financial Chatbot")
st.caption(
    "Ask questions about revenue, expenses, profitability, "
    "account balances, and financial performance."
)

# --------------------------------------------------
# LOAD EXCEL DATA
# --------------------------------------------------

FILE_PATH = "sample_pnl.xlsx"

try:
    df = pd.read_excel(FILE_PATH)
except FileNotFoundError:
    st.error(
        "sample_pnl.xlsx was not found. "
        "Upload it to your project directory."
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
df["Account"] = df["Account"].fillna("").astype(str).str.strip()
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df = df.dropna(subset=["Amount"])
df = df[df["Account"] != ""]

if df.empty:
    st.warning("No valid financial records were found.")
    st.stop()

# --------------------------------------------------
# ACCOUNT CLASSIFICATION
# Customize these patterns to match your chart of accounts.
# --------------------------------------------------

REVENUE_RE = re.compile(
    r"\b(revenue|sales|turnover|service income|"
    r"sales returns?|discounts? given|allowances)\b",
    re.I
)

EXPENSE_RE = re.compile(
    r"\b(expense|expenses|cost of sales|cost of goods sold|"
    r"cogs|salary|salaries|wages|rent|utilities|"
    r"marketing|advertising|insurance|depreciation|"
    r"professional fees|office supplies|travel|"
    r"repairs|maintenance|freight|payroll|"
    r"interest expense|bank charges|commission expense)\b",
    re.I
)

CONTRA_REVENUE_RE = re.compile(
    r"\b(sales returns?|returns and allowances|"
    r"discounts? given|sales discounts?)\b",
    re.I
)

COGS_RE = re.compile(
    r"\b(cost of sales|cost of goods sold|cogs)\b",
    re.I
)


def classify_account(account):
    """Assign a broad financial category using account names."""
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

# Assumption:
# Revenue is positive and expenses are negative.
# If your workbook uses the opposite convention, configure the
# normalization functions below to match your accounting system.

def revenue_total():
    return df.loc[
        df["Category"].isin(["Revenue", "Contra-revenue"]),
        "Amount"
    ].sum()


def expense_total():
    # Return a positive expense figure for display.
    return -df.loc[
        df["Category"].isin(["Expense", "Cost of sales"]),
        "Amount"
    ].sum()


def cost_of_sales():
    return -df.loc[
        df["Category"] == "Cost of sales", "Amount"
    ].sum()


def net_income():
    # Assumes this worksheet contains only relevant P&L accounts
    # and the source amounts already have their correct signs.
    return df["Amount"].sum()


def money(value):
    return f"${value:,.2f}"


def percent(value):
    return f"{value:.2f}%"


# --------------------------------------------------
# NATURAL-LANGUAGE INTENT DETECTION
# --------------------------------------------------

def normalize(text):
    """Normalize common conversational wording."""
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[$,?!.]", " ", text)
    text = re.sub(r"\b(what's|whats)\b", "what is", text)
    text = re.sub(r"\b(how's|hows)\b", "how is", text)
    text = re.sub(r"\b(made|earned)\b", "profit", text)
    text = re.sub(r"\b(spent|spending)\b", "expense", text)
    text = re.sub(r"\b(costs|cost)\b", "expense", text)
    text = re.sub(r"\b(money coming in)\b", "revenue", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_intent(question):
    q = normalize(question)

    # Order matters: check specific questions before general ones.

    if re.search(
        r"\b(top|largest|biggest|highest|most expensive|"
        r"rank|ranking)\b", q
    ) and re.search(r"\b(expense|account|cost|spend)\b", q):
        return "top_expenses"

    if re.search(
        r"\b(profit margin|net margin|margin percentage|"
        r"profitability ratio|percentage of revenue)\b", q
    ):
        return "profit_margin"

    if re.search(
        r"\b(gross profit|gross margin)\b", q
    ):
        return "gross_profit"

    if re.search(
        r"\b(net income|net profit|bottom line|"
        r"business profit|overall profit|"
        r"how profitable|making a profit|"
        r"are we profitable|is the business profitable)\b", q
    ) or q in {"profit", "profitability"}:
        return "net_income"

    if re.search(
        r"\b(total revenue|net revenue|revenue|sales|"
        r"turnover|income from sales)\b", q
    ):
        return "revenue"

    if re.search(
        r"\b(total expenses|all expenses|operating expenses|"
        r"total expense|how much did we spend|"
        r"expense total|overall expenses)\b", q
    ):
        return "expenses"

    if re.search(
        r"\b(summarize|summary|overview|analyze|analyse|"
        r"financial performance|business performance|"
        r"how is the business doing|"
        r"explain the pnl|explain the p and l)\b", q
    ):
        return "summary"

    if re.search(
        r"\b(compare|breakdown|distribution|"
        r"by category|expense categories)\b", q
    ):
        return "breakdown"

    if re.search(
        r"\b(loss|losing money|negative profit)\b", q
    ):
        return "net_income"

    return "account"


# --------------------------------------------------
# ACCOUNT SEARCH
# --------------------------------------------------

# Build the model from account names and individual words.
# Character n-grams help match wording variations and typos.

account_text = (
    df["Account"].str.lower()
    + " "
    + df["Account"].str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True)
)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5),
    min_df=1
)

account_matrix = vectorizer.fit_transform(account_text)


def find_account(question):
    q = normalize(question)

    # Try exact or partial account-name matching first.
    q_clean = re.sub(r"[^a-z0-9 ]", " ", q)
    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    exact_matches = df[
        df["Account"].str.lower().apply(
            lambda name: (
                name == q_clean
                or (len(q_clean) >= 4 and q_clean in name)
            )
        )
    ]

    if not exact_matches.empty:
        return exact_matches.iloc[0]["Account"], 1.0

    scores = cosine_similarity(
        vectorizer.transform([q]), account_matrix
    ).flatten()

    best_index = scores.argmax()
    return df.iloc[best_index]["Account"], scores[best_index]


# --------------------------------------------------
# FINANCIAL ANALYSIS
# --------------------------------------------------

def top_expense_accounts(limit=5):
    expenses = df[
        df["Category"].isin(["Expense", "Cost of sales"])
    ].copy()

    if expenses.empty:
        return expenses

    expenses["Expense amount"] = -expenses["Amount"]
    expenses = expenses.sort_values(
        "Expense amount", ascending=False
    )

    return expenses.head(limit)


def financial_summary():
    revenue = revenue_total()
    expenses = expense_total()
    profit = net_income()

    if revenue != 0:
        margin = profit / revenue * 100
    else:
        margin = None

    lines = [
        f"**Revenue:** {money(revenue)}",
        f"**Expenses (excluding separately classified "
        f"contra-revenue):** {money(expenses)}",
        f"**Net P&L balance:** {money(profit)}",
    ]

    if margin is not None:
        lines.append(f"**Net margin:** {percent(margin)}")
    else:
        lines.append(
            "**Net margin:** Not available because revenue is zero."
        )

    if profit > 0:
        lines.append("The net P&L balance is positive.")
    elif profit < 0:
        lines.append("The net P&L balance is negative.")
    else:
        lines.append("The net P&L balance is zero.")

    if expenses > 0:
        top = top_expense_accounts(3)
        if not top.empty:
            names = ", ".join(top["Account"].tolist())
            lines.append(f"**Largest classified expense accounts:** {names}")

    lines.append(
        "_Check that all P&L accounts are included and that "
        "revenue and expense signs match your workbook's convention._"
    )

    return "\n\n".join(lines)


def answer_question(question):
    intent = detect_intent(question)
    revenue = revenue_total()
    expenses = expense_total()
    profit = net_income()

    if intent == "revenue":
        return f"**Total revenue:** {money(revenue)}"

    if intent == "expenses":
        return f"**Total classified expenses:** {money(expenses)}"

    if intent == "net_income":
        if profit > 0:
            status = "positive net P&L balance"
        elif profit < 0:
            status = "negative net P&L balance"
        else:
            status = "break-even net P&L balance"

        return (
            f"**Net P&L balance:** {money(profit)}\n\n"
            f"The current data indicates a {status}. "
            "This assumes all rows are relevant P&L accounts "
            "with consistent accounting signs."
        )

    if intent == "profit_margin":
        if revenue == 0:
            return (
                "I cannot calculate the margin because "
                "classified revenue is zero."
            )

        margin = profit / revenue * 100

        return (
            f"**Net margin:** {percent(margin)}\n\n"
            f"Calculated as net P&L balance ({money(profit)}) "
            f"divided by classified revenue ({money(revenue)}). "
            "This is only reliable when the workbook contains "
            "the relevant P&L accounts and consistent signs."
        )

    if intent == "gross_profit":
        gross = revenue - cost_of_sales()

        if revenue == 0:
            gross_margin_text = "Not available because revenue is zero."
        else:
            gross_margin_text = percent(gross / revenue * 100)

        return (
            f"**Estimated gross profit:** {money(gross)}\n\n"
            f"**Estimated gross margin:** {gross_margin_text}\n\n"
            "This calculation uses classified revenue less "
            "classified cost of sales. Verify that all relevant "
            "accounts have been classified correctly."
        )

    if intent == "top_expenses":
        top = top_expense_accounts(5)

        if top.empty:
            return (
                "No expense accounts were identified. "
                "Review the account-classification patterns."
            )

        lines = ["**Largest classified expense accounts:**"]

        for _, row in top.iterrows():
            lines.append(
                f"- {row['Account']}: "
                f"{money(row['Expense amount'])}"
            )

        return "\n".join(lines)

    if intent == "summary":
        return financial_summary()

    if intent == "breakdown":
        grouped = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        lines = ["**P&L category breakdown (source signs):**"]

        for category, amount in grouped.items():
            lines.append(f"- {category}: {money(amount)}")

        lines.append(
            "\nCategories are inferred from account names. "
            "Review unclassified accounts before relying on this breakdown."
        )

        return "\n".join(lines)

    # Default: search for a matching account.
    account, score = find_account(question)

    if score < 0.15:
        return (
            "I couldn't confidently match that question to an account. "
            "Try asking about revenue, expenses, profit, gross margin, "
            "or use the exact account name from your Excel file."
        )

    matches = df[df["Account"] == account]

    if len(matches) == 1:
        amount = matches["Amount"].sum()
        category = matches["Category"].iloc[0]

        return (
            f"**{account}**\n\n"
            f"Amount: {money(amount)}\n\n"
            f"Category: {category}\n\n"
            f"Match confidence (similarity score): {score:.2f}"
        )

    total = matches["Amount"].sum()

    return (
        f"**{account}**\n\n"
        f"Combined amount across {len(matches)} rows: {money(total)}\n\n"
        "Multiple rows share this account name."
    )


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Example questions")
    st.markdown(
        """
        - How much revenue did we make?
        - Are we profitable?
        - What are our biggest expenses?
        - What is our net profit margin?
        - Calculate gross profit.
        - Summarize financial performance.
        - Show the expense breakdown.
        - How much did we spend on salaries?
        """
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your P&L...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    answer = answer_question(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

# Optional: inspect source data.
with st.expander("View loaded P&L data"):
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df)} financial rows loaded.")
