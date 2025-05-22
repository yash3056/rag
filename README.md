# AI Notebook Project

## Overview

This is a Django-based web application that provides an AI-powered notebook interface, document QA, and model serving integration using [vLLM](https://github.com/vllm-project/vllm).

### Features

- **AI Notebook**: Interactive notebook interface embedded within Django.
- **Document QA**: Upload PDF documents and query them using embeddings and FAISS.
- **Model Serving**: Host large language models with vLLM for inference and reasoning.

## Prerequisites

- Python 3.11 or later
- Git
- [vLLM](https://github.com/vllm-project/vllm) installed (see below)

## Setup

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/yash3056/Research_Assistant.git ai-notebook
   cd ai-notebook
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Initialize the database and apply migrations:
   ```bash
   # Run migrations and ensure the projects directory exists
   python startup.py
   ```

5. (Optional) Create the default admin user (`admin` / `admin`):
   ```bash
   python create_admin.py
   ```

## Running the Django Server

By default, the Django server listens on port **8001** to avoid conflicts with the vLLM service.

```bash
python run.py
``` 

Or, using manage.py directly:

```bash
# Explicit port
python manage.py runserver 0.0.0.0:8001
```

Open your browser at [http://localhost:8001](http://localhost:8001).

## Running vLLM Model Server

In a separate terminal (same virtual environment), start the vLLM service:

```bash
vllm serve Qwen/Qwen3-1.7B \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --max_model_len 32768
```

This serves the `Qwen3-1.7B` model on the default port **8000**. Ensure vLLM is installed:

```bash
pip install vllm
```

## Configuration

- Django settings are in `ai_notebook/settings.py`.
- The base projects directory is configured via `PROJECTS_BASE_DIR` in settings.
- Database: `db.sqlite3` by default.

## File Structure

```
/ai-notebook
├── ai_notebook/          # Django project
├── document_qa/          # App for document QA
├── projects/             # Stored project data & FAISS indices
├── templates/            # HTML templates
├── static/               # CSS, JS, images
├── startup.py            # Initialize DB and run server on 8000
├── run.py                # Run Django on port 8001
├── create_admin.py       # Create default admin user
├── requirements.txt      # Python dependencies
├── manage.py
└── README.md
```

## License

This project is licensed under the terms of the [MIT License](LICENSE).
