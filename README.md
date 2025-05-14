# Research Assitant

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
- Python 3.11+
- [pip](https://pip.pypa.io/en/stable/) for package management
- SQLite (default) or another Django-supported database

## Installation
1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd "Python project"
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Apply database migrations:
   ```bash
   python manage.py migrate
   ```
5. (Optional) Create a superuser for the Django admin:
   ```bash
   python create_admin.py
   ```

## Usage
### Web Application
1. Start the Django development server:
   ```bash
   python manage.py runserver
   ```
2. Navigate to `http://127.0.0.1:8000/` to access the web UI.
3. Upload documents and query via the web interface.

### CLI Scripts
- `doctoembed.py`: Convert documents into embeddings and store vectors.
- `migrate_projects.py`: Reindex or migrate existing project embeddings.
- `run.py`: Helper to run custom pipelines or server.
- `startup.py`: Bootstraps environment and services.
- `tmp.py`: Scratch script for quick tests.

### Jupyter Notebook
Open `openvino export.ipynb` in the `ai_notebook/` folder to experiment with data processing, embedding, and querying.

## Project Structure
```
├── ai_notebook/         # Django project and settings
├── document_qa/         # Django app for document QA
├── projects/            # Generated embedding indexes per project
├── static/              # Static assets (CSS, JS, images)
├── templates/           # HTML templates for the web UI
├── manage.py            # Django management tool
├── requirements.txt     # Python dependencies
├── db.sqlite3           # Default SQLite database
├── create_admin.py      # Script to create Django superuser
├── doctoembed.py        # CLI for document embedding
├── migrate_projects.py  # Script to migrate or reindex projects
├── run.py               # Entrypoint for custom runs
├── startup.py           # Initialization script
└── README.md            # Project documentation
```

## Configuration
- `projects.json`: Tracks all project IDs and metadata.
- `metadata.pkl`: Pickle file for model metadata.

## Testing
Run Django tests:
```bash
python manage.py test
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
