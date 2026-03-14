
import streamlit as st
import pandas as pd
import os
import sys

# allow imports from copilot/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPILOT_DIR = os.path.join(BASE_DIR, "copilot")
if COPILOT_DIR not in sys.path:
    sys.path.append(COPILOT_DIR)

from query_router import answer_question

st.set_page_config(
    page_title="SaaS Churn Copilot",
    page_icon="📉",
    layout="wide"
)

st.title("📉 SaaS Churn Copilot")
st.markdown("Ask a churn analytics question and get SQL-backed answers from the database.")

st.subheader("Example questions")
st.markdown("""
- show me top 10 high risk accounts  
- average churn risk by region  
- which plan has the most high risk accounts  
- how many accounts are in the top 5%
""")

question = st.text_input("Ask your question")

if st.button("Run Query"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        output = answer_question(question)

        if output["matched_query"] is None:
            st.error(output["message"])
        else:
            st.success("Query matched successfully.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Matched Query")
                st.code(output["matched_query"], language="text")

            with col2:
                st.markdown("### SQL Used")
                st.code(output["sql"], language="sql")

            st.markdown("### Result")
            st.dataframe(output["result"], use_container_width=True)