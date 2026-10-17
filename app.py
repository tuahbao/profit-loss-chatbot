import os
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Profit & Loss AI Chatbot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Profit & Loss Financial Chatbot")
st.caption(
    "Explore a sample US company's financial results "
    "and ask questions in plain English."
)

COMPANY = "Acme Technology Solutions, Inc."
PERIOD = "Year ended December 31, 2025"
CURRENCY = "USD"

# Positive amounts are stored for both revenue and expenses.
# Expense signs are applied when calculating profit.

REVENUE = {
    "Product sales": 850_000,
    "Consulting services revenue": 420_000,
    "Subscription revenue": 280_000,
    "Other revenue": 25_000,
}

COST_OF_REVENUE = {
    "Product costs": 310_000,
    "Direct labor": 180_000,
    "Hosting and infrastructure": 45_000,
}

OPERATING_EXPENSES = {
    "Salaries and wages": 300_000,
    "Payroll taxes and benefits": 55_000,
    "Rent": 36_000,
    "Utilities": 8_000,
    "Marketing and advertising": 65_000,
    "Software subscriptions": 18_000,
    "Insurance": 12_000,
    "Professional fees": 20_000,
    "Travel and entertainment": 15_000,
    "Office supplies": 5_000,
    "Depreciation": 25_000,
    "Research and development": 40_000,
    "Repairs and maintenance": 4_000,
    "Training": 6_000,
    "Telephone and internet": 7_000,
}

INTEREST_EXPENSE = 12_000
TAX_RATE = 0.21  # Illustrative assumption, not tax advice


def calculate_financials():
    revenue = sum(REVENUE.values())
    cost_of_revenue = sum(COST_OF_REVENUE.values())
    gross_profit = revenue - cost_of_revenue

    operating_expenses = sum(OPERATING_EXPENSES.values())
    operating_income = gross_profit - operating_expenses

    pretax_income = operating_income - INTEREST_EXPENSE
    tax_expense = round(pretax_income * TAX_RATE)
    net_income = pretax_income - tax_expense

    return {
        "Total revenue": revenue,
        "Total cost of revenue": cost_of_revenue,
        "Gross profit": gross_profit,
        "Total operating expenses": operating_expenses,
        "Operating income": operating_income,
        "Interest expense": INTEREST_EXPENSE,
        "Income before tax": pretax_income,
        "Income tax expense": tax_expense,
        "Net income": net_income,
        "Gross profit margin": (
            gross_profit / revenue if revenue else 0
        ),
        "Operating margin": (
            operating_income / revenue if revenue else 0
        ),
        "Net profit margin": (
            net_income / revenue if revenue else 0
        ),
    }


FINANCIALS = calculate_financials()


def money(value):
    return f"${value:,.2f}"


def percentage(value):
    return f"{value:.2%}"


# --------------------------------------------------
# 2. CREATE THE EXCEL WORKBOOK
# --------------------------------------------------

def create_excel():
    wb = Workbook()
    ws = wb.active
    ws.title = "Profit and Loss"

    ws.append([COMPANY])
    ws.append([f"Income Statement | {PERIOD}"])
    ws.append(["All amounts in US dollars (USD)"])
    ws.append([])
    ws.append(["Section", "Financial line item", "Amount (USD)"])

    # Revenue
    for name, amount in REVENUE.items():
        ws.append(["Revenue", name, amount])

    revenue_start = 6
    revenue_end = revenue_start + len(REVENUE) - 1
    ws.append([
        "Revenue",
        "TOTAL REVENUE",
        f"=SUM(C{revenue_start}:C{revenue_end})",
    ])
    total_revenue_row = ws.max_row
    ws.append([])

    # Cost of revenue
    cost_start = ws.max_row + 1
    for name, amount in COST_OF_REVENUE.items():
        ws.append(["Cost of revenue", name, amount])

    cost_end = ws.max_row
    ws.append([
        "Cost of revenue",
        "TOTAL COST OF REVENUE",
        f"=SUM(C{cost_start}:C{cost_end})",
    ])
    total_cost_row = ws.max_row
    ws.append([
        "Profitability",
        "GROSS PROFIT",
        f"=C{total_revenue_row}-C{total_cost_row}",
    ])
    gross_profit_row = ws.max_row
    ws.append([])

    # Operating expenses
    expense_start = ws.max_row + 1
    for name, amount in OPERATING_EXPENSES.items():
        ws.append(["Operating expenses", name, amount])

    expense_end = ws.max_row
    ws.append([
        "Operating expenses",
        "TOTAL OPERATING EXPENSES",
        f"=SUM(C{expense_start}:C{expense_end})",
    ])
    total_expenses_row = ws.max_row
    ws.append([
        "Profitability",
        "OPERATING INCOME",
        f"=C{gross_profit_row}-C{total_expenses_row}",
    ])
    operating_income_row = ws.max_row
    ws.append(["Other expenses", "Interest expense", INTEREST_EXPENSE])

    interest_row = ws.max_row
    ws.append([
        "Profitability",
        "INCOME BEFORE TAX",
        f"=C{operating_income_row}-C{interest_row}",
    ])
    pretax_row = ws.max_row

    # Tax and net income
    ws.append([
        "Other expenses",
        "Income tax expense (illustrative 21%)",
        f"=C{pretax_row}*21%",
    ])
    tax_row = ws.max_row

    ws.append([
        "Profitability",
        "NET INCOME",
        f"=C{pretax_row}-C{tax_row}",
    ])
    net_income_row = ws.max_row

    # Margins
    ws.append([])
    ws.append([
        "Profitability",
        "Gross profit margin",
        f"=C{gross_profit_row}/C{total_revenue_row}",
    ])
    ws.append([
        "Profitability",
        "Operating margin",
        f"=C{operating_income_row}/C{total_revenue_row}",
    ])
    ws.append([
        "Profitability",
        "Net profit margin",
        f"=C{net_income_row}/C{total_revenue_row}",
    ])

    # Formatting
    ws.freeze_panes = "A6"
    ws.auto_filter.ref = f"A5:C{net_income_row}"
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 43
    ws.column_dimensions["C"].width = 20

    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="center")

            if cell.row == 1:
                cell.font = Font(
                    bold=True, size=16, color="FFFFFF"
                )
                cell.fill = PatternFill(
                    "solid", fgColor="17365D"
                )

            elif cell.row == 2:
                cell.font = Font(italic=True, color="17365D")

            elif cell.row == 5:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill(
                    "solid", fgColor="4472C4"
                )

            if cell.column == 3 and cell.row >= 6:
                cell.number_format = '$#,##0.00;[Red]($#,##0.00)'

    for row_num in [
        total_revenue_row,
        total_cost_row,
        gross_profit_row,
        total_expenses_row,
        operating_income_row,
        pretax_row,
        net_income_row,
    ]:
        for cell in ws[row_num]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(
                "solid", fgColor="D9EAF7"
            )

    for row_num in range(net_income_row + 2, ws.max_row + 1):
        ws.cell(row_num, 3).number_format = "0.00%"

    # Add a notes sheet
    notes = wb.create_sheet("Notes")
    notes.append(["Item", "Description"])
    notes.append([
        "Company",
        "Fictional US technology services company",
    ])
    notes.append([
        "Period",
        PERIOD,
    ])
    notes.append([
        "Currency",
        "USD",
    ])
    notes.append([
        "Tax assumption",
        "Illustrative 21% of pretax income; not tax advice",
    ])
    notes.append([
        "Purpose",
        "Educational sample; not actual financial results",
    ])

    for cell in notes[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")

    notes.column_dimensions["A"].width = 22
    notes.column_dimensions["B"].width = 75

    output = BytesIO()
    wb.save(output)
    return output.getvalue()


EXCEL_BYTES = create_excel()

st.download_button(
    label="📥 Download sample P&L Excel workbook",
    data=EXCEL_BYTES,
    file_name="sample_profit_loss.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
)


# --------------------------------------------------
# 3. MACHINE LEARNING QUESTION MATCHING
# --------------------------------------------------

FAQS = [
    {
        "question": "What is total revenue?",
        "answer": (
            f"Total revenue is {money(FINANCIALS['Total revenue'])}. "
            "It includes product sales, consulting, subscriptions, "
            "and other revenue."
        ),
    },
    {
        "question": "What is gross profit?",
        "answer": (
            f"Gross profit is {money(FINANCIALS['Gross profit'])}. "
            "It equals revenue minus cost of revenue."
        ),
    },
    {
        "question": "What is net income or net profit?",
        "answer": (
            f"Net income is {money(FINANCIALS['Net income'])}. "
            "It is income after operating expenses, interest, "
            "and the illustrative income tax expense."
        ),
    },
    {
        "question": "What is the gross profit margin?",
        "answer": (
            "Gross profit margin is "
            f"{percentage(FINANCIALS['Gross profit margin'])}. "
            "It is gross profit divided by revenue."
        ),
    },
    {
        "question": "What is the operating profit margin?",
        "answer": (
            "Operating margin is "
            f"{percentage(FINANCIALS['Operating margin'])}. "
            "It is operating income divided by revenue."
        ),
    },
    {
        "question": "What is the net profit margin?",
        "answer": (
            "Net profit margin is "
            f"{percentage(FINANCIALS['Net profit margin'])}. "
            "It is net income divided by revenue."
        ),
    },
    {
        "question": "What are the total operating expenses?",
        "answer": (
            "Total operating expenses are "
            f"{money(FINANCIALS['Total operating expenses'])}."
        ),
    },
    {
        "question": "What is cost of revenue?",
        "answer": (
            "Total cost of revenue is "
            f"{money(FINANCIALS['Total cost of revenue'])}."
        ),
    },
    {
        "question": "What is operating income?",
        "answer": (
            "Operating income is "
            f"{money(FINANCIALS['Operating income'])}. "
            "It is gross profit minus operating expenses."
        ),
    },
    {
        "question": "Is the company profitable?",
        "answer": (
            f"Yes. The sample company reports net income of "
            f"{money(FINANCIALS['Net income'])} and a net margin of "
            f"{percentage(FINANCIALS['Net profit margin'])}."
        ),
    },
    {
        "question": "What are the largest operating expenses?",
        "answer": (
            "The three largest operating expenses are "
            + ", ".join(
                f"{name} ({money(amount)})"
                for name, amount in sorted(
                    OPERATING_EXPENSES.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            )
            + "."
        ),
    },
    {
        "question": "What is the income tax expense?",
        "answer": (
            "The illustrative income tax expense is "
            f"{money(FINANCIALS['Income tax expense'])}. "
            "The 21% assumption is for demonstration only."
        ),
    },
    {
        "question": "What is the reporting period?",
        "answer": f"The reporting period is {PERIOD}.",
    },
    {
        "question": "What currency does the company use?",
        "answer": "The statement uses US dollars (USD).",
    },
]

# Include each question and its answer in the text used for matching.
ML_TEXTS = [
    faq["question"] + " " + faq["answer"]
    for faq in FAQS
]

VECTORIZER = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)

FAQ_MATRIX = VECTORIZER.fit_transform(ML_TEXTS)


def ml_answer(question, threshold=0.12):
    """Return the closest matching FAQ answer."""
    question_vector = VECTORIZER.transform([question])
    scores = cosine_similarity(
        question_vector, FAQ_MATRIX
    ).flatten()

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < threshold:
        return None

    return FAQS[best_index]["answer"]


# --------------------------------------------------
# 4. DYNAMIC FINANCIAL QUESTIONS
# --------------------------------------------------

def answer_question(question):
    q = question.lower().strip()

    # Ask for a financial line item by name.
    all_items = {
        **REVENUE,
        **COST_OF_REVENUE,
        **OPERATING_EXPENSES,
        "Interest expense": INTEREST_EXPENSE,
        "Income tax expense": FINANCIALS["Income tax expense"],
    }

    for name, amount in sorted(
        all_items.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if name.lower() in q:
            if any(word in q for word in [
                "what", "how much", "amount", "expense",
                "revenue", "cost", "salary", "rent",
            ]):
                return f"{name}: {money(amount)}."

    # Scenario analysis: expense increase.
    increase = re.search(
        r"(?:increase|rise|grow|up)\D{0,20}(\d+(?:\.\d+)?)\s*%",
        q,
    )

    if increase and any(
        word in q for word in ["expense", "cost", "operating"]
    ):
        rate = float(increase.group(1)) / 100
        added_cost = FINANCIALS["Total operating expenses"] * rate
        new_net_income = FINANCIALS["Net income"] - added_cost

        return (
            f"If total operating expenses increase by {rate:.1%}, "
            f"expenses rise by {money(added_cost)}. Assuming revenue, "
            "cost of revenue, interest, and taxes remain unchanged, "
            f"illustrative net income becomes {money(new_net_income)}. "
            "This simplified scenario does not recalculate income tax."
        )

    # Revenue growth scenario.
    growth = re.search(
        r"(?:revenue|sales)\D{0,20}(?:increase|grow|growth|rise)"
        r"\D{0,15}(\d+(?:\.\d+)?)\s*%",
        q,
    )

    if growth:
        rate = float(growth.group(1)) / 100
        new_revenue = FINANCIALS["Total revenue"] * (1 + rate)

        return (
            f"At {rate:.1%} revenue growth, projected revenue is "
            f"{money(new_revenue)}, assuming the other revenue "
            "items scale proportionally."
        )

    # Direct calculations.
    if "revenue" in q and any(
        word in q for word in ["total", "overall", "sales"]
    ):
        return f"Total revenue is {money(FINANCIALS['Total revenue'])}."

    if "gross profit" in q:
        return (
            f"Gross profit is {money(FINANCIALS['Gross profit'])}, "
            f"with a margin of "
            f"{percentage(FINANCIALS['Gross profit margin'])}."
        )

    if "net income" in q or "net profit" in q:
        return (
            f"Net income is {money(FINANCIALS['Net income'])}, "
            f"with a margin of "
            f"{percentage(FINANCIALS['Net profit margin'])}."
        )

    if "operating income" in q or "operating profit" in q:
        return (
            f"Operating income is "
            f"{money(FINANCIALS['Operating income'])}."
        )

    if "margin" in q:
        return ml_answer(q)

    if any(word in q for word in [
        "largest expense", "biggest expense", "highest expense",
    ]):
        return ml_answer(q)

    # ML fallback for natural-language questions.
    answer = ml_answer(q)

    if answer:
        return answer

    return (
        "I couldn't confidently match that question to the sample "
        "financial data. Try asking about revenue, gross profit, "
        "net income, operating expenses, margins, or a specific "
        "line item. You can also extend the FAQ training examples."
    )


# --------------------------------------------------
# 5. FINANCIAL DASHBOARD
# --------------------------------------------------

st.subheader(COMPANY)
st.caption(f"{PERIOD} | Illustrative financial data | USD")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Revenue", money(FINANCIALS["Total revenue"]))
m2.metric("Gross profit", money(FINANCIALS["Gross profit"]))
m3.metric("Operating income", money(FINANCIALS["Operating income"]))
m4.metric("Net income", money(FINANCIALS["Net income"]))

st.subheader("Financial statement")

rows = [
    {"Section": "Revenue", "Item": k, "Amount (USD)": v}
    for k, v in REVENUE.items()
]
rows += [
    {"Section": "Cost of revenue", "Item": k, "Amount (USD)": v}
    for k, v in COST_OF_REVENUE.items()
]
rows += [
    {"Section": "Operating expenses", "Item": k, "Amount (USD)": v}
    for k, v in OPERATING_EXPENSES.items()
]

statement = pd.DataFrame(rows)
st.dataframe(statement, use_container_width=True, hide_index=True)

expense_df = pd.DataFrame(
    list(OPERATING_EXPENSES.items()),
    columns=["Expense", "Amount (USD)"],
).sort_values("Amount (USD)", ascending=False)

st.subheader("Operating expense breakdown")
st.bar_chart(expense_df.set_index("Expense"))

# Excel upload for a future/customized workflow.
# This demo chatbot uses its built-in sample data.
with st.expander("Upload another Excel workbook"):
    uploaded = st.file_uploader(
        "Choose an .xlsx file",
        type=["xlsx"],
    )

    if uploaded:
        try:
            preview = pd.read_excel(uploaded)
            st.write("Workbook preview:")
            st.dataframe(preview, use_container_width=True)
            st.info(
                "This starter version previews uploaded workbooks. "
                "The chatbot still answers from the built-in sample. "
                "Connect the uploaded data to the calculation engine "
                "before using it for custom company questions."
            )
        except Exception as exc:
            st.error(f"Could not read the workbook: {exc}")


# --------------------------------------------------
# 6. CHAT INTERFACE
# --------------------------------------------------

st.subheader("💬 Ask the financial chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can answer questions about the sample "
                "company's revenue, expenses, profits, and margins. "
                "Try: What is the net profit margin?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask a question about the profit and loss statement..."
)

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = answer_question(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
