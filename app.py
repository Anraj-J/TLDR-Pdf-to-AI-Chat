from PIL import Image
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

import os
import openai

st.set_page_config(page_title="TLDR")

penai_api_key = os.getenv("OPENAI_API_KEY")

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():


  with st.container():
    st.title("TLDR | Win Back Your Time")
    lottie_coding = load_lottieurl("https://lottie.host/5a3a650d-704e-48b5-8a5a-e356cd4409a4/D4HGHdjMdS.json")
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Dont have time to read a whole document? Just upload it as a pdf and ask what you need to know!")
        st.write("Whether it be that chapter from your textbook that's just a tad too long or the 1000th resume that you don't want to read, TLDR can help you get the exact info you need; nothing more, nothing less.")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")
    
  
  with st.container():
    img_open_ai = Image.open("Images/chatgptlogo.png")
    st.subheader("Powered By:")
    st.image(img_open_ai)
    st.write("TLDR is powered by ChatGPT, an advanced AI developed by OpenAI. ChatGPT brings a whole new level of intelligence and responsiveness, check it out!")
    # upload filep
    pdf = st.file_uploader("Upload PDF Here", type="pdf")
    
    # extract text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()
