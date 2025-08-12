# pip install streamlit rank_bm25 faiss-cpu sentence-transformers langchain-huggingface PyPDF2
# pip install langchain langchain-community langchain-openai python-dotenv

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI


# 0) 환경 변수 로드
PDF_PATH = "토끼와_거북이_줄거리.pdf"

# 1) PDF에서 문서 로드
def load_documents_from_pdf(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} 파일이 없습니다.")

    reader = PdfReader(pdf_path)
    texts, metadatas = [], []

    for page_num, page in enumerate(reader.pages, start=1):
        content = page.extract_text()
        if not content or not content.strip():
            continue
        texts.append(content.strip())
        metadatas.append({
            "page": page_num,
            "source": f"pdf_page_{page_num}",
            "title": "토끼와 거북이"
        })

    return texts, metadatas

# 2) 앙상블 Retriever 구성 (BM25 + FAISS)
def build_ensemble_retriever(texts, metadatas):
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas)
    bm25.k = 2

    embedding = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    faiss_store = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    faiss = faiss_store.as_retriever(search_kwargs={"k": 2})

    return EnsembleRetriever(retrievers=[bm25, faiss], weights=[0.3, 0.7])

# 3) OpenAI LLM 로드
@st.cache_resource(show_spinner=False)
def load_openai_llm(model_name="gpt-4o-mini", temperature=0.0):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경변수 설정 필요")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

# 4) 검색 함수
def search(query, retriever):
    return retriever.invoke(query) or []

# 5) 프롬프트 구성
def build_prompt(query, docs):
    lines = []
    lines.append("아래 '자료'만 근거로 한국어로 간결히 답하세요.")
    lines.append("- 자료 밖 정보를 추측하지 마세요.")
    lines.append("- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.\n")
    lines.append(f"질문:\n{query}\n")
    lines.append("자료:")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        lines.append(f"[문서{i}] (source={m.get('source')}, page={m.get('page')}, title={m.get('title')})\n{d.page_content}\n")
    lines.append("답변:")
    return "\n".join(lines)

# 6) LLM 호출
def generate_with_llm(llm, prompt):
    resp = llm.invoke(prompt)
    return resp.content.strip()

# 7) Streamlit 앱
def main():
    st.set_page_config(page_title="📄 PDF RAG 어시스턴트", page_icon="📄", layout="centered")
    st.title("📄 토끼와 거북이 줄거리 Q&A")

    try:
        texts, metadatas = load_documents_from_pdf(PDF_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if "retriever" not in st.session_state:
        st.session_state.retriever = build_ensemble_retriever(texts, metadatas)
    if "llm" not in st.session_state:
        st.session_state.llm = load_openai_llm()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("PDF 내용에 대해 질문하세요.")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        docs = search(user_input.strip(), st.session_state.retriever)

        if not docs:
            answer = "제공된 문서에서 찾지 못했습니다."
        else:
            prompt = build_prompt(user_input.strip(), docs)
            answer = generate_with_llm(st.session_state.llm, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("🔎 사용한 자료 보기"):
            if not docs:
                st.markdown("_검색 결과 없음_")
            else:
                for i, d in enumerate(docs, 1):
                    m = d.metadata
                    st.markdown(f"**[문서{i}]** (source={m.get('source')}, page={m.get('page')}, title={m.get('title')})\n\n{d.page_content}")

if __name__ == "__main__":
    main()
