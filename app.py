
import streamlit as st

st.set_page_config(
    page_title="P&L Finance Chatbot",
    page_icon="💰",
    layout="centered",
)

# Sample data in Malaysian ringgit (RM)
revenue = 100_000.00
cost_of_goods_sold = 60_000.00

expenses = {
    "Salaries and wages": 12_000.00,
    "Rental": 3_000.00,
    "Utilities": 800.00,
    "Marketing": 1_200.00,
    "Transportation": 1_000.00,
    "Office and other expenses": 1_000.00,
}

gross_profit = revenue - cost_of_goods_sold
total_expenses = sum(expenses.values())
net_profit = gross_profit - total_expenses
profit_margin = net_profit / revenue * 100 if revenue else 0

st.title("💰 P&L Finance Chatbot")
st.caption("Ask questions about your sample business finances.")

with st.sidebar:
    st.header("Sample P&L Summary")
    st.metric("Revenue", f"RM {revenue:,.2f}")
    st.metric("Gross profit", f"RM {gross_profit:,.2f}")
    st.metric("Operating expenses", f"RM {total_expenses:,.2f}")
    st.metric("Net profit before tax", f"RM {net_profit:,.2f}")
    st.metric("Net profit margin", f"{profit_margin:.2f}%")

    st.subheader("Try asking")
    st.markdown(
        "- What is my revenue?\n"
        "- How much is my net profit?\n"
        "- What are my total expenses?\n"
        "- Which expense is the highest?\n"
        "- Show me my P&L statement"
    )


def answer_question(question):
    q = question.lower().strip()

    # Use explicit keyword rules for this starter version.
    if any(word in q for word in ["hello", "hi", "hey"]):
        return (
            "Hello! I can answer questions about your "
            "sample profit and loss statement."
        )

    if "highest expense" in q or "biggest expense" in q:
        name, amount = max(
            expenses.items(), key=lambda item: item[1]
        )
        return f"Your highest expense is {name}: RM {amount:,.2f}."

    if "margin" in q:
        return f"Your net profit margin is {profit_margin:.2f}%."

    if "revenue" in q or "sales" in q:
        return f"Your revenue is RM {revenue:,.2f}."

    if "cost of goods" in q or "cogs" in q:
        return (
            f"Your cost of goods sold is "
            f"RM {cost_of_goods_sold:,.2f}."
        )

    if "gross profit" in q:
        return f"Your gross profit is RM {gross_profit:,.2f}."

    if "net profit" in q or "profit before tax" in q:
        return f"Your net profit before tax is RM {net_profit:,.2f}."

    if "expense" in q or "expenditure" in q or "spending" in q:
        details = "\n".join(
            f"- {name}: RM {amount:,.2f}"
            for name, amount in expenses.items()
        )
        return (
            f"Total operating expenses: RM {total_expenses:,.2f}\n\n"
            f"{details}"
        )

    if "profit and loss" in q or "p&l" in q:
        return (
            f"Revenue: RM {revenue:,.2f}\n\n"
            f"Cost of goods sold: RM {cost_of_goods_sold:,.2f}\n\n"
            f"Gross profit: RM {gross_profit:,.2f}\n\n"
            f"Operating expenses: RM {total_expenses:,.2f}\n\n"
            f"Net profit before tax: RM {net_profit:,.2f}"
        )

    return (
        "I couldn't identify that question. Try asking about "
        "revenue, gross profit, cost of goods sold, expenses, "
        "net profit, profit margin, or your P&L statement."
    )


# Keep the conversation in Streamlit session state.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome! I'm your P&L chatbot. "
                "What would you like to know about your finances?"
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your P&L..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    response = answer_question(prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)
