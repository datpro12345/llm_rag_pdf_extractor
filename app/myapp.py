import streamlit as st
import base64
import tempfile
import uuid
import json            # CHANGED: dùng để parse JSON
import re              # CHANGED: dùng để tìm kiếm JSON trong response
import pandas as pd    # CHANGED: dùng để hiển thị dữ liệu dạng bảng
from pydantic import BaseModel, Field  # CHANGED: định nghĩa model để parse JSON

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize API key in session state if not provided
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

def display_pdf(uploaded_file):
    """
    Display the uploaded PDF in an iframe.
    """
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_pdf_text(uploaded_file):
    """
    Save the uploaded PDF to a temp file and extract text using PyPDFLoader.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name
    loader = PyPDFLoader(tmp_filename)
    pages = loader.load()
    return pages

def create_vectorstore_from_texts(documents, api_key, file_name):
    """
    Split document text into chunks, generate embeddings using Google API and create a Chroma vector store.
    """
    embedding_function = GoogleGenerativeAIEmbeddings(
         model="models/text-embedding-004",
         api_key=api_key
    )
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=1500,
         chunk_overlap=200,
         length_function=len,
         separators=["\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(documents)
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    vectorstore = Chroma.from_documents(
         documents=chunks,
         ids=ids,
         embedding=embedding_function,
         persist_directory=f"./vectorstore_{file_name}"
    )
    vectorstore.persist()
    return vectorstore

# ============================
# CHANGED: Định nghĩa các model dùng để parse kết quả JSON trả về từ LLM.
class AnswerWithSources(BaseModel):
    answer: str = Field(..., description="Answer to question")
    sources: str = Field(..., description="Exact text snippet from context")
    reasoning: str = Field(..., description="Brief explanation on how the answer was derived")

class ExtractedInfo(BaseModel):
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources
# ============================

# ============================
# CHANGED: Cập nhật hàm query_document với prompt mới để ép output ra JSON hoàn chỉnh.
def query_document(vectorstore, query, api_key):
    """
    Retrieve relevant text chunks from the vector store, build a prompt that instructs the LLM to output a 
    structured JSON response and parse it to return an ExtractedInfo object.
    
    Yêu cầu:
      - Trích xuất tiêu đề của bài báo chính xác như trong PDF.
      - Không thêm text bình luận ngoài JSON.
    """
    # Tạo retriever dựa trên similarity search
    retriever = vectorstore.as_retriever(search_type="similarity")
    relevant_docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # CHANGED: Prompt mới được cập nhật với hướng dẫn chặt chẽ về JSON
    prompt = f"""
You are an expert in analyzing academic papers. Based solely on the context provided, extract the following information exactly and accurately:
1. paper_title: The complete title of the research paper, exactly as it appears in the paper.
2. paper_summary: A brief summary of the research paper.
3. publication_year: The publication year of the paper.
4. paper_authors: The author(s) of the paper.

Return your answer as a valid JSON object with no additional text, in the following format (ensure it starts with {{ and ends with }} exactly):

{{
  "paper_title": {{"answer": "<paper title>", "sources": "<exact text snippet>", "reasoning": "<brief explanation>"}},
  "paper_summary": {{"answer": "<summary>", "sources": "<exact text snippet>", "reasoning": "<brief explanation>"}},
  "publication_year": {{"answer": "<year>", "sources": "<exact text snippet>", "reasoning": "<brief explanation>"}},
  "paper_authors": {{"answer": "<author(s)>", "sources": "<exact text snippet>", "reasoning": "<brief explanation>"}}
}}

Do not include any extra commentary or bibliography.
Context:
{context}

Question: {query}
"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
    response = llm.invoke(prompt)
    # CHANGED: Nếu response không phải string, chuyển nó thành string
    if hasattr(response, 'content'):
        response_text = response.content
    else:
        response_text = str(response)
    
    # CHANGED: Dùng regex để ép lấy phần JSON nếu LLM trả thêm text ngoài JSON.
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    else:
        json_str = response_text

    try:
        structured_dict = json.loads(json_str)
        info = ExtractedInfo(**structured_dict)
    except Exception as e:
        st.error("Failed to parse structured response. Raw response:")
        st.write(response_text)
        return None
    return info
# ============================

def load_streamlit_page():
    """
    Set up the Streamlit page with two columns:
      - Left: Input Google API key and upload PDF.
      - Right: Display the PDF.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1:
        st.header("Input your Google API key")
        st.text_input('Google API key', type='password', key='api_key',
                      label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")
    return col1, col2, uploaded_file

# ------------------------
# Main page logic:
col1, col2, uploaded_file = load_streamlit_page()

if uploaded_file is not None:
    with col2:
        display_pdf(uploaded_file)
        
    documents = get_pdf_text(uploaded_file)
    st.session_state.vector_store = create_vectorstore_from_texts(
        documents, 
        api_key=st.session_state.api_key,
        file_name=uploaded_file.name
    )
    st.write("Input Processed")

# ------------------------
# CHANGED: Khi nhấn "Generate table", gọi query_document và chuyển đổi kết quả thành bảng.
with col1:
    if st.button("Generate table"):
        with st.spinner("Generating answer..."):
            info = query_document(
                vectorstore=st.session_state.vector_store, 
                query="Give me the title, summary, publication date, and authors of the research paper.",
                api_key=st.session_state.api_key
            )
            if info:
                # Chuyển ExtractedInfo thành dict và tạo bảng
                info_dict = info.dict()
                # Các hàng: answer, sources, reasoning
                rows = ["answer", "sources", "reasoning"]
                data = {key: [info_dict[key]['answer'], info_dict[key]['sources'], info_dict[key]['reasoning']]
                        for key in info_dict}
                df_table = pd.DataFrame(data, index=rows)
                st.table(df_table)