# 🚀 MultiModal-RAG-with-Gemini
 

## Project Overview

This is a powerful multimodal Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. It provides an intelligent chat interface for interacting with both PDF documents and images using state-of-the-art AI models. The application can:

- 📄 Process PDF documents and extract both text and images
- 🖼️ Handle direct image uploads for visual question answering
- 🔍 Create and query searchable vector databases
- 💬 Generate accurate, context-aware responses using Google's Gemini model
- 🎯 Provide precise answers based on both textual and visual content

## 🎥 Demo

https://github.com/Amulyakantamneni/MultiModal-RAG-with-Gemini/assets/demo.mp4

Watch the demo video above to see the application in action! The video showcases:
- PDF document processing and analysis
- Image upload and visual question answering
- Real-time chat interaction
- User interface navigation

## Key Features

- **📚 Document Processing**
  - PDF text and image extraction
  - Automatic content indexing
  - Vector database creation for efficient retrieval
  - Embedded PDF viewer for easy reference

- **🎨 Image Analysis**
  - Support for PNG, JPG, and JPEG formats
  - Visual question answering with Gemini
  - Smart image resizing and optimization
  - Direct image upload capability

- **🤖 Advanced AI Integration**
  - Google Gemini model for text and vision
  - CLIP model for image embeddings
  - Context-aware response generation
  - Multi-turn conversation support

- **👥 User Experience**
  - Intuitive chat interface
  - Real-time response generation
  - Clean and responsive design
  - Session state management

## How It Works

1. **Data Processing:**
   - PDF documents are processed to extract text and images
   - Images are optimized and prepared for analysis
   - Content is converted into searchable embeddings

2. **Vector Database:**
   - Extracted content is stored in FAISS vector database
   - Efficient indexing for fast retrieval
   - Separate handling of text and image embeddings

3. **Query Processing:**
   - User questions are analyzed for intent
   - Relevant context is retrieved from the vector database
   - Both text and image context are considered

4. **Response Generation:**
   - Retrieved context is sent to Gemini model
   - AI generates natural language responses
   - Responses include references to source content

## Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- Google API Key for Gemini access

### Quick Start

1. **Clone the Repository:**
```bash
git clone https://github.com/Amulyakantamneni/MultiModal-RAG-with-Gemini.git
cd MultiModal-RAG-with-Gemini
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API Keys:**
Create `.env`:
```toml
GOOGLE_API_KEY = "your_api_key_here"
```

4. **Run the Application:**
```bash
streamlit run app.py
```

## Usage Guide

### PDF Mode
1. Upload your PDF through the sidebar
2. Wait for processing completion
3. Ask questions about the document
4. Use the PDF viewer for reference

### Image Mode
1. Upload an image through the sidebar
2. Type your question about the image
3. Press Enter to get AI analysis
4. Continue the conversation as needed

### General Chat
- Use the chat interface naturally
- Reference previous conversations
- Get AI-powered responses
- Switch between documents as needed

## Technical Architecture

```
MultiModal-RAG-app/
├── app.py              # Main Streamlit interface
├── llm_utils.py        # LLM and vector store utilities
├── utils.py            # Core processing functions
├── requirements.txt    # Dependencies
├── outputs/           
│   └── vectorstore/    # Vector database files
└── pdfs/              # Document storage
```

## Dependencies

Key libraries used:
- `streamlit`: Web interface
- `langchain`: LLM framework
- `google-generativeai`: Gemini API
- `PyMuPDF`: PDF processing
- `FAISS`: Vector storage
- `transformers`: CLIP model
- `Pillow`: Image processing

## Contributing

Contributions are welcome! Please feel free to:
- Report issues
- Suggest features
- Submit pull requests
- Improve documentation


## Acknowledgments

- Google Gemini for AI capabilities
- Streamlit for the web framework
- LangChain for LLM integration
- PyMuPDF for PDF processing
- CLIP for image embeddings
- The open-source community
