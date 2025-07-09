# Research Assistant

## Overview
This project is a Django-based web application that enables document ingestion, embedding, and QA using vector search. It provides tools to upload documents, build vector indexes, and query documents with natural language. A Jupyter notebook is also included for experimentation.

## Features
- Upload and process documents into FAISS vector indexes
- Store and manage metadata and embeddings per project
- Document question-answering via similarity search
- Web interface for uploading and querying documents
- Example Jupyter notebook for prototyping and exploring the pipeline
- CLI scripts for batch processing and maintenance

## Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for fast Python package management
- SQLite (default) or another Django-supported database

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yash3056/Research_Assistant.git -b openvino Research_Assistant
   cd Research_Assistant
   ```

2. Install uv if you haven't already For Linux:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Install uv if you haven't already For Windows:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. Install dependencies and create virtual environment:
   ```bash
   uv sync
   ```

4. Apply database migrations:
   ```bash
   uv run python manage.py migrate
   ```

5. (Optional) Create a superuser for the Django admin:
   ```bash
   uv run python create_admin.py
   ```

## Usage
### Web Application
1. Start the Django development server:
   ```bash
   uv run python startup.py
   ```
2. Navigate to `http://127.0.0.1:8000/` to access the web UI.
3. Upload documents and query via the web interface.

### Jupyter Notebook
Open `custom_modal.ipynb` to experiment with data processing, embedding, and querying:
```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

## Project Structure
```
├── ai_notebook/         # Django project and settings
├── document_qa/         # Django app for document QA
├── ov_model/           # OpenVINO model files
├── projects/           # Generated embedding indexes per project
├── static/             # Static assets (CSS, JS, images)
├── templates/          # HTML templates for the web UI
├── manage.py           # Django management tool
├── pyproject.toml      # Project dependencies and configuration
├── uv.lock            # Dependency lock file
├── db.sqlite3         # Default SQLite database
├── create_admin.py    # Script to create Django superuser
├── doctoembed.py      # CLI for document embedding
├── startup.py         # Initialization script
├── custom_modal.ipynb   # Jupyter notebook for experimentation
└── README.md          # Project documentation
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
