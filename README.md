# 🤖 Corporate Q&A Bot

A Retrieval-Augmented Generation (RAG) based Corporate Q&A Bot that answers questions using company documents stored in Azure Blob Storage.
The system retrieves relevant document sections using Azure AI Search and generates answers using Azure OpenAI.

---

## 📌 Overview

This project implements a corporate knowledge assistant that allows users to query internal documents. The bot retrieves relevant document chunks from Azure AI Search and uses Azure OpenAI to generate contextual answers with citations.

The solution is designed using a **RAG (Retrieval-Augmented Generation)** architecture.

---

## 🧠 Architecture

```
User Question
      ↓
Python Bot
      ↓
Azure AI Search (Retrieve relevant document chunks)
      ↓
Azure OpenAI (Generate answer)
      ↓
Answer with citations
```

Documents are stored in Azure Blob Storage and indexed using Azure AI Search indexers.

---

## ✨ Features

* 🔎 Retrieval-Augmented Generation (RAG)
* ☁️ Azure AI Search integration
* 🤖 Azure OpenAI response generation
* 📑 Document citation support
* ⚡ Streaming responses
* 🧠 Context-aware conversation memory
* 📂 Folder-based document filtering

---

## 🛠 Technologies Used

* 🐍 Python
* 🔍 Azure AI Search
* 🤖 Azure OpenAI
* ☁️ Azure Blob Storage
* 📦 Python dotenv
* ⚙️ Azure SDK for Python

---

## 📁 Project Structure

```
Corporate_Bot/
│
├── main.py                # Main chatbot script
├── .env                   # Environment variables
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/corporate-qa-bot.git
cd corporate-qa-bot
```

---

### 2️⃣ Create a Python virtual environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Configure environment variables

Create a `.env` file in the project root and add:

```
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_INDEX_NAME=
AZURE_SEARCH_API_KEY=

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=
AZURE_OPENAI_API_VERSION=

AZURE_STORAGE_ACCOUNT_URL=
BLOB_CONTAINER_NAME=
```

---

## ☁️ Azure Setup

### 1️⃣ Upload documents

Upload company documents to Azure Blob Storage.

Example structure:

```
container
 └── Kenneth
      ├── employee_handbook.pdf
      ├── hr_policy.pdf
```

---

### 2️⃣ Create Azure AI Search Index

Create an index with fields such as:

* `content`
* `metadata_storage_path`

These fields are used by the Python bot for document retrieval.

---

### 3️⃣ Configure an Indexer

Create an indexer to:

1. Scan documents from Blob Storage
2. Extract text
3. Store document content in the search index

---

## ▶️ Running the Bot

Start the chatbot:

```
python main.py
```

Example interaction:

```
You: What is the company's leave policy?

Bot: Employees are entitled to annual leave based on their tenure.
[Source: employee_handbook.pdf]
```

Type:

```
exit
```

to stop the chatbot.

---

## 🔄 Example Workflow

1️⃣ User asks a question
2️⃣ Azure AI Search retrieves relevant document chunks
3️⃣ The bot sends retrieved context to Azure OpenAI
4️⃣ Azure OpenAI generates a response with citations

---

## 🔮 Future Improvements

* 📊 Add vector search with embeddings
* 📄 Support PDF page citations
* 🌐 Build a web interface
* 🔐 Add authentication for enterprise use
* 🚀 Deploy as an API service

---

## 👨‍💻 Author

Kenneth
Corporate Q&A Bot Assignment
