# An-Chatbot-Integration

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)

## 📝 Description

Bring the power of intelligent conversation to your applications with An-Chatbot-Integration, a Python-based solution designed for seamless chatbot integration. This project provides a robust foundation for building and deploying chatbots, allowing you to automate customer interactions, provide instant support, and enhance user engagement. Leverage the flexibility of Python to create a customized chatbot experience tailored to your specific needs.

## 🛠️ Tech Stack

- 🐍 Python


## 📦 Key Dependencies

```
fastapi: latest
uvicorn: latest
pandas: latest
langchain: 0.2.15
langchain-community: 0.2.14
langgraph: latest
chromadb: latest
mysql-connector-python: latest
```

## 📁 Project Structure

```
.
├── app.py
├── chrom.py
├── count.py
├── re.txt
├── requirements.txt
├── static
│   ├── script.js
│   └── style.css
└── templates
    └── index.html
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.9+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt` if needed also install the `pip install -r re.txt`
5. If you dont have Ollama download it and select the model you want to use (in this case (MISTRAL))
6. Then pull the respected model in the terminal using the command
   - `ollama pull mistral` (you can use your own choice of model)
8. Run the chrom.py first to create the chroma DB
9. Then the run the count.py to check how many files are there in the Chroma DB
10. Finally run the app.py file


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/ambrose-kutti/An-Chatbot-Integration.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request
