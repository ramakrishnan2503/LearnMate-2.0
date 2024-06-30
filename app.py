import os
from pydantic import BaseModel
from dotenv import load_dotenv
from pytube import YouTube
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
import streamlit as st
from newapp import (
    download_video,
    video_to_images,
    video_to_audio,
    audio_to_text,
    retrieve
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure page layout and styling
st.set_page_config(page_title="Learnmate", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .main {
        background: #1e1e1e;
    }
    .sidebar .sidebar-content {
        background: #1e1e1e;
    }
    h1 {
        color: #3b3b3b;
        text-align: center;
        margin-top: 20px;
    }
    .stButton>button {
        background-color: transparent;
        color: white;
        border: 2px solid #3b3b3b;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #6f6f6f;
    }
    .stTextInput>div>input {
        background-color: #2e2e2e;
        color: white;
        border: 1px solid #3b3b3b;
        border-radius: 10px;
    }
    .stSpinner>div>div {
        border-top-color: #3b3b3b;

    }
    .chat-box {
        background-color: #2e2e2e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-box.user {
        border-left: 5px solid #3b3b3b;
    }
    .chat-box.model {
        border-left: 5px solid #6f6f6f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main application code
def main():

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for user input
    st.sidebar.header("Learnmate")
    youtube_url = st.sidebar.text_input("Enter YouTube URL:", key="youtube_url")
    submit_button = st.sidebar.button("Submit", key="submit_button")

    if submit_button and youtube_url:
        st.session_state.youtube_url_input = youtube_url
        st.session_state.processed = False
        st.session_state.query_str = ""

    if "youtube_url_input" in st.session_state:
        youtube_url = st.session_state.youtube_url_input
        st.subheader("YouTube Video")
        try:
            yt = YouTube(youtube_url)
            st.video(yt.streams.filter(file_extension='mp4').first().url)
        except Exception as e:
            st.warning("Failed to load video. Please check the URL.")

        if not st.session_state.get("processed", False):
            output_video_path = "video_data/"
            output_folder = "mixed_data2/"
            output_audio_path = os.path.join(output_folder, "output_audio.wav")

            # Ensure output directories are clear and recreated
            if os.path.exists(output_folder):
                for file in os.listdir(output_folder):
                    os.remove(os.path.join(output_folder, file))
            os.makedirs(output_folder, exist_ok=True)

            filepath = os.path.join(output_video_path, "input_vid.mp4")

            with st.spinner("Downloading video..."):
                metadata_vid = download_video(youtube_url, output_video_path)
            st.success("Video downloaded successfully!")

            with st.spinner("Extracting images from video..."):
                video_to_images(filepath, output_folder)
            st.success("Images extracted successfully!")

            with st.spinner("Extracting audio from video..."):
                video_to_audio(filepath, output_audio_path)
            st.success("Audio extracted successfully!")

            with st.spinner("Transcribing audio to text..."):
                text_data = audio_to_text(output_audio_path)
            st.success("Audio transcribed successfully!")

            with st.spinner("Saving text data to file..."):
                with open(os.path.join(output_folder, "output_text.txt"), "w") as file:
                    file.write(text_data)
                os.remove(output_audio_path)
            st.success("Text data saved and audio file removed!")

            with st.spinner("Initializing vector store..."):
                text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
                image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
                storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
            st.success("Vector store initialized!")

            with st.spinner("Embedding documents..."):
                model_name = "models/embedding-001"
                embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document")

                documents = SimpleDirectoryReader(output_folder).load_data()
                index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
                st.session_state.retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=5)
            st.success("Documents embedded successfully!")

            st.session_state.processed = True

        st.subheader("Question Answering")
        query_str = st.text_input("Ask a question about the video:", key="query_str_input")
        ask_button = st.button("Ask", key="ask_button")

        if ask_button and query_str:
            st.session_state.query_str = query_str
            st.write(f"**Question:** {query_str}")

            with st.spinner("Retrieving relevant information..."):
                img, text = retrieve(st.session_state.retriever_engine, query_str)

            context_str = "".join(text)
            image_documents = SimpleDirectoryReader(input_files=img).load_data()

            class OutputClass(BaseModel):
                text_result: str

            gemini_llm = GeminiMultiModal(api_key=GOOGLE_API_KEY, model_name="models/gemini-pro-vision")

            qa_tmpl_str = (
                "Based on the provided information, including relevant images and retrieved context from the video, "
                "accurately and precisely answer the query without any additional prior knowledge.\n"
                "---------------------\n"
                "Context: {context_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            prompt = qa_tmpl_str.format(query_str=query_str, context_str=context_str)

            with st.spinner("Generating answer..."):
                llm_program = MultiModalLLMCompletionProgram.from_defaults(
                    image_documents=image_documents,
                    prompt_template_str=prompt,
                    multi_modal_llm=gemini_llm,
                    output_cls=OutputClass,
                    verbose=True,
                )

                response = llm_program(prompt=prompt)

            st.session_state.chat_history.append({"user": query_str, "model": response.text_result})

        elif ask_button:
            st.warning("Please enter a question.")

        # Display chat history
        st.subheader("Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"""
            <div class='chat-box user'>
                <strong>User:</strong> {chat['user']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='chat-box model'>
                <strong>Model:</strong> {chat['model']}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
