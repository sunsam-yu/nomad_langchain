import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

CLOUDFLARE_SITEMAP_URL = "https://developers.cloudflare.com/sitemap-index.xml"

PRODUCT_URLS = [
    "https://developers.cloudflare.com/ai-gateway/",
    "https://developers.cloudflare.com/vectorize/",
    "https://developers.cloudflare.com/workers-ai/",
]


def parse_page(soup: BeautifulSoup):
    for tag_name in [
        "nav",
        "header",
        "footer",
        "script",
        "style",
    ]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    main = soup.find("main")

    if main:
        text = main.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    return text


@st.cache_data(show_spinner="Loading Cloudflare documentation...")
def load_cloudflare_docs():
    loader = SitemapLoader(
        CLOUDFLARE_SITEMAP_URL,
        filter_urls=PRODUCT_URLS,
        parsing_function=parse_page,
    )

    loader.requests_per_second = 2

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    splits = splitter.split_documents(docs)

    return splits


@st.cache_resource(show_spinner="Creating vector store...")
def create_retriever(openai_api_key):
    splits = load_cloudflare_docs()

    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 8,
        }
    )

    return retriever


def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source') or doc.metadata.get('loc')}\nContent: {doc.page_content}"
        for doc in docs
    )


def get_sources(docs):
    sources = []

    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("loc")

        if source and source not in sources:
            sources.append(source)

    return sources[:5]


def create_chain(openai_api_key):
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
You are Cloudflare SiteGPT.

Answer the user's question using ONLY the Cloudflare documentation context below.

Rules:
- Only answer from the provided context.
- If the answer is not in the context, say you don't know.
- Be concise but complete.
- If possible, mention the relevant Cloudflare product.
- Do not make up prices, limits, or product capabilities.

Context:
{context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm

    return chain


def answer_question(question, openai_api_key):
    retriever = create_retriever(openai_api_key)
    docs = retriever.invoke(question)

    context = format_docs(docs)
    sources = get_sources(docs)

    chain = create_chain(openai_api_key)
    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content, sources


def save_message(role, content, sources=None):
    st.session_state["messages"].append(
        {
            "role": role,
            "content": content,
            "sources": sources or [],
        }
    )


def send_message(role, content, sources=None, save=True):
    with st.chat_message(role):
        st.markdown(content)

        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(source)

    if save:
        save_message(role, content, sources)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["role"],
            message["content"],
            message.get("sources", []),
            save=False,
        )


if "messages" not in st.session_state:
    st.session_state["messages"] = []


st.title("🌐 SiteGPT")

st.markdown(
    """
Ask questions about Cloudflare's AI documentation.

This chatbot answers questions using documentation for:

- AI Gateway
- Cloudflare Vectorize
- Workers AI
    """
)


with st.sidebar:
    st.header("Settings")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "[GitHub Repository](https://github.com/sunsam-yu/nomad_langchain)"
    )


paint_history()


if not openai_api_key:
    st.info("Enter your OpenAI API Key in the sidebar.")
else:
    message = st.chat_input(
        "Ask a question about Cloudflare AI docs..."
    )

    if message:
        send_message(
            "user",
            message,
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching Cloudflare docs and generating an answer..."):
                answer, sources = answer_question(
                    message,
                    openai_api_key,
                )

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source)

        save_message(
            "assistant",
            answer,
            sources,
        )