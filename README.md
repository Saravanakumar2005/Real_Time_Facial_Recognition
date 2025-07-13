<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/779dcad3-870b-42a2-904c-2e4b0ee8582f" /># 🧠 Face Recognition Platform with Real-Time AI Q&A using RAG

A browser-based platform for registering and recognizing faces in real-time using the laptop webcam. It supports multi-face detection and includes a chatbot interface powered by Retrieval-Augmented Generation (RAG) to answer queries about face registration events.

---

## 🚀 Features

### 🔐 Face Registration

* Access webcam and detect faces
* Register and assign names to detected faces
* Store facial encodings and metadata (name, timestamp) in MongoDB
* Allow multiple unique face registrations

### 🧠 Live Recognition

* Live webcam feed scanning
* Real-time face recognition from stored encodings
* Bounding boxes and names shown on screen
* Multi-face detection in one frame

### 💬 Chat-Based Query (RAG)

* Embedded chatbot interface (HTML + CSS)
* RAG architecture using Flask (with Python) and WebSockets
* Ask questions like:

  * "Who was the last person registered?"
  * "When was Karthik registered?"
  * "How many people are currently registered?"
* RAG built using LangChain + FAISS + OpenAI LLM API

---

## 🧠 Tech Stack

### 💅 Frontend

* HTML + CSS + JavaScript

### 💅 Backend

* Python (Flask + WebSocket)

### 🤖 Face Recognition & RAG

* Python with the following libraries:

#### 🔍 Face Recognition

* numpy  
* opencv  
* insightface  
* onnxruntime  
* pandas  

#### 📆 Data Processing & ML

* scikit-learn  

#### 📚 Retrieval-Augmented Generation (RAG)

* langchain  
* chromadb  
* faiss-cpu  
* tiktoken  

---


## 🏗 Project Structure (Monorepo)

```
├── backend/
│   ├── app.py
├── frontend/
│   └── template/
│       ├── index.html
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 📦 Prerequisites

* Python 3.10
* MySql-Workbench
* Webcam-enabled device

### 🔧 Backend Setup(Flask + Face Recognition + RAG)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 🧠 Python RAG Notes

To resolve issues with `insightface`, use the following command to manually install the wheel:

```bash
pip install insightface-0.7.3-cp310-cp310-win_amd64.whl
```

Ensure the `.whl` file is in the same directory or provide the full path to the file.

---

## 🧱 Architecture Diagram

![Architecture Diagram](https://chat.openai.com/cdn/pub/file-KtbFJqxtB64hwMWgwALWgA)

---

## 📜 Assumptions

* Using MongoDB MySql-Workbench for the database
* Flask-SocketIO  is used for real-time chat updates
* Basic error handling and logging implemented for face registration
* User identification in chat handled by session context

---

## 📹 Demo Video

https://drive.google.com/file/d/12ULGomluGHXCcVWD4aOrBSewZNCKD5zw/view?usp=sharing

---

## 📜 Logs

* Logging implemented for:

  * Face registration events (name + timestamp)

---

## 🏁 Submission Note

This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com)
