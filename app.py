import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))





# extraire le text de chaque page du pdf 
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


# diviser ce texte en chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# embeddings and vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# creer prompt et conversational chain
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# definir la generation de la réponse par le chatbot

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error("Erreur lors de la désérialisation: " + str(e))
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

# définir la fonction principale main (celle de l'interaction avec l'utilisateur)


def main():
    st.set_page_config(
        page_title="Aitkheri_PDF_Chatbot",
        layout="centered",
        initial_sidebar_state="expanded",
        page_icon="📝"
    )


    st.markdown(
        """
        <style>
        html, body {
            margin: 0px;
            width: 100vw;
            height: 100vh;
            font-family: arial;
            overflow: hidden;
        }

        .sidebar .block-container {
            text-align: center;
        }
        
        .sidebar .stButton > button {
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("📚 🤖 PDF Chatbot ")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", placeholder="Type your question here...")

    if user_question:
        response = user_input(user_question)
 

    with st.sidebar:
        st.title("Documentation:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Add Button", accept_multiple_files=True)
        if st.button("Add"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()




