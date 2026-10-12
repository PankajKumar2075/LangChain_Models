from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=ChatOpenAI()

st.header('Research Tool')

paper_input=st.selectbox('Select a paper', ['Paper 1', 'Paper 2', 'Paper 3'])

style_input=st.selectbox('Select a style', ['Style 1', 'Style 2', 'Style 3'])

length_input=st.selectbox('Select a length', ['Short', 'Medium', 'Long'])

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"]
)

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button('Summarize'):
    response = model.generate(prompt)
    st.write(response)
