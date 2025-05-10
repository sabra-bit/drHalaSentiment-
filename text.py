import streamlit as st

text_to_copy = "This is the text you want to be copyable."

st.write("Here is some text in a copyable box:")

st.code(text_to_copy)