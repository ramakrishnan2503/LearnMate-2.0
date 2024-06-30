from pathlib import Path
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv

import speech_recognition as sr
from pytube import YouTube
from moviepy.editor import VideoFileClip

from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def download_video(url, output_path):
    yt = YouTube(url)
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(output_path=output_path, filename="input_vid.mp4")
    return metadata

# Extract images from video
def video_to_images(video_path, output_folder):
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(os.path.join(output_folder, "frame%04d.png"), fps=0.2)

# Extract audio from video
def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

# Convert audio to text
def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
            text = ""
    return text



video_url = "https://youtu.be/3dhcmeOTZ_Q"

output_video_path = "D:/multimodal rag/video_data/"
output_folder = "D:/multimodal rag/mixed_data2/"
output_audio_path = "D:/multimodal rag/mixed_data2/output_audio.wav"
os.makedirs(output_folder, exist_ok=True)
filepath = os.path.join(output_video_path, "./input_vid.mp4")



metadata_vid = download_video(video_url, output_video_path)
video_to_images(filepath, output_folder)
video_to_audio(filepath, output_audio_path)
text_data = audio_to_text(output_audio_path)



with open(os.path.join(output_folder, "output_text.txt"), "w") as file:
    file.write(text_data)
print("Text data saved to file")
os.remove(output_audio_path)
print("Audio file removed")



text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)




model_name = "models/embedding-001"

embed_model = GeminiEmbedding(
    model_name=model_name, api_key=GOOGLE_API_KEY, title="this is a document"
)



documents = SimpleDirectoryReader(output_folder).load_data()
index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context,embed_model=embed_model)



retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=5)



def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)
    return retrieved_image, retrieved_text






query = "who is elon musk?"
img, text = retrieve(retriever_engine, query)



def plot_images(images_path):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in images_path:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 5:
                break


plot_images(img)


qa_tmpl_str=(
    "Based on the provided information, including relevant images and retrieved context from the video, \
    accurately and precisely answer the query without any additional prior knowledge.\n"

    "---------------------\n"
    "Context: {context_str}\n"

    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)

img

import json
metadata_str=json.dumps(metadata_vid)

metadata_str

query_str="who is elon musk?"

context_str = "".join(text)

image_documents = SimpleDirectoryReader( input_files=img).load_data()

image_documents



gemini_llm = GeminiMultiModal(
        api_key=GOOGLE_API_KEY, model_name="models/gemini-pro-vision"
    )


class OutputClass(BaseModel):
    text_result : str

output_cls = OutputClass

prompt = qa_tmpl_str.format(query_str=query_str, context_str=context_str)

llm_program = MultiModalLLMCompletionProgram.from_defaults(
        image_documents=image_documents,
        prompt_template_str=prompt,
        multi_modal_llm=gemini_llm,
        output_cls=output_cls,
        verbose=True,
    )


print(prompt)


response = llm_program(prompt=prompt)

pprint(response.text_result)





