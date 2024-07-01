# Learnmate: YouTube Video Viewer and Question Answering

Learnmate is a web application built using Streamlit that allows users to input a YouTube URL and ask questions about the video. The application processes the video, extracts images and audio, transcribes the audio to text, and uses a multimodal retrieval-augmented generation (RAG) approach to answer questions based on the video content.

## Features

- Download and process YouTube videos.
- Extract images and audio from the video.
- Transcribe audio to text.
- Embed documents and images using a vector store.
- Retrieve relevant information and generate answers to user queries.
- Display chat history of questions and answers.

## Getting Started

### Prerequisites

- Python 3.11
- Git
- A Google Developer's API Key

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ramakrishnan2503/LearnMate-2.0.git
    cd LearnMate-20.0
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory of the project.
    - Add your Google API Key to the `.env` file:
      ```
      GOOGLE_API_KEY=your-google-api-key
      ```

### Running the Application

1. **Start the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

2. **Open your browser and navigate to the URL specified to use the application**.

## Usage

1. **Enter a YouTube URL**:
    - Paste the URL of the YouTube video you want to process in the sidebar and click "Submit".

2. **Ask a Question**:
    - After the video is processed, you can ask as many question as you wish about the video content in the provided input box.

3. **View Chat History**:
    - The chat history will display the questions asked by the user and the corresponding answers provided by the application.

