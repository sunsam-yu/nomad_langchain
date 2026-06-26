import streamlit as st

from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_name, file_content, openai_api_key):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")

    files_dir = Path("./.cache/files")
    files_dir.mkdir(parents=True, exist_ok=True)

    file_path = files_dir / file_name

    with open(file_path, "wb") as file:
        file.write(file_content)

    loader = TextLoader(file_path, encoding="utf-8")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=350,
        chunk_overlap=80,
    )

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )

    vectorstore = FAISS.from_documents(
        docs,
        cached_embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 12}
    )

    return retriever

def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save = False,
        )
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chain(retriever, openai_api_key):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        openai_api_key=openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant.

Answer the user's question using only the context below.
If the answer is not in the context, say you don't know.
Do not make up an answer.

Context:
{context}
""",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return chain

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("📄 DocumentGPT")

with st.sidebar:
    st.header("Settings")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    st.markdown("---")

    st.markdown(
        "[GitHub Repository](https://github.com/sunsam-yu/nomad_langchain)"
    )

st.markdown(
    """
Upload a document and ask questions about it.

Your OpenAI API key is used only for this session.
    """

)

uploaded_file = st.file_uploader(
    "Upload a .txt file",
    type=["txt"],
)

if uploaded_file is not None:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        retriever = embed_file(
            uploaded_file.name,
            uploaded_file.read(),
            openai_api_key,
        )

        st.success(f"Uploaded and embedded: {uploaded_file.name}")

        paint_history()

        message = st.chat_input("Ask anything about your document...")

        if message:
            send_message(message, "human")

            chain = get_chain(
                retriever,
                openai_api_key,
            )

            with st.chat_message("ai"):
                with st.spinner("Thinking..."):
                    response = chain.invoke(message)
                    st.markdown(response.content)

            save_message(response.content, "ai")
else:
    st.info("Please upload a .txt file to start chatting.")