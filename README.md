# pdf-chatbot-rag

# 🤖 PDF Chatbot with RAG using LangChain & OpenAI

This project is a Retrieval-Augmented Generation (RAG) based chatbot that reads a PDF file, creates embeddings from its content, and allows you to ask questions about the document using OpenAI's GPT model.

---

## 📚 What It Does

- Loads a PDF (e.g., "Prompt Engineering") 📄
- Converts it into text chunks
- Embeds the chunks using OpenAI embeddings
- Stores them in an in-memory vector store
- Uses a retriever to fetch relevant chunks based on your query
- Uses GPT (`gpt-4o-mini`) to generate answers based on the retrieved context

---

## 🧠 Technologies Used

- Python 3.9+
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://platform.openai.com/)
- [DocArray Vector Store](https://python.langchain.com/docs/modules/data_connection/vectorstores/docarray)
- [dotenv](https://pypi.org/project/python-dotenv/)


