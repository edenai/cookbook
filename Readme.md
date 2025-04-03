# Eden AI Cookbook 🧑‍🍳

<p align="center">
<img src="assets/logo.jpg" alt="Eden AI Logo" width="200" style="max-width:100%; height:auto;">
</p>

> A delightful collection of Python recipes using Eden AI services. Because cooking up AI solutions should be as easy (and fun) as following a recipe!

## 📋 Table of Contents
- [✨ Overview](#-overview)
- [🗂️ Project Structure](#%EF%B8%8F-project-structure)
- [⚙️ Setup](#%EF%B8%8F-setup)

## ✨ Overview

EDENAI-COOKBOOK is a curated set of examples demonstrating how to leverage Eden AI's powerful APIs for:

- 📄 **Document Parsing** - extract info from PDFs, resumes, etc.
- ⚙️ **EdenAI-OpenAI-Adapter** - shows openAI compatibility with Eden AI's API.
- 🖼️ **Image Analysis** - detect fake content, generate embeddings
- 🎙️ **Speech** - text-to-speech, speech-to-text
- 📝 **Text** - embeddings, sentiment analysis, summarization
- 🧩 **Workflows** - Combination of various AI features built on the Eden AI's Workflows platform.

These code snippets serve as "recipes" to jumpstart your own AI projects. Simply grab a script, tweak a few lines, and integrate advanced AI features into your workflow.

## 🗂️ Project Structure

```
EDENAI-COOKBOOK/
├── Document Parser/
│   └── [ financial-parser.py, resume-parser.py]
├── EdenAI-OpenAI-Adapter/
│   └── [ adapter.py, app.py ]
├── Image/
│   ├── fake_content_detection.py
│   └── image_embeddings.py
├── RAG/
│   └── rag.py
├── Speech/
│   └── text_to_speech_async.py
├── Text/
│   └── text_embeddings.py
├── Workflows/
│   └── [ OCR_LLM.py, webscraping_LLM.py ]
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Setup

### Requirements
- Python 3.8+
- Pip or Poetry

### Environment Variables
Create a `.env` file with:
```bash
EDENAI_API_KEY="your_api_key_here"
```

Happy coding! 🎉
