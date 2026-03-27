# CrewAI Streamlit Multi-Agent App

A demo Streamlit app that uses CrewAI + Langchain + OpenAI to:
- route user questions through dedicated specialist agents (`python` or `general`)
- generate concise answers
- produce a social media post summary
- generate an image prompt
- call OpenAI image generation (`gpt-image-1`) and display result

## 📦 Project Files

- `app.py` – Streamlit app orchestrating agents and UI.
- `Dockerfile` – container setup with Python 3.11 slim and Streamlit config.
- `requirements.txt` – required Python packages.

## 🧠 Architecture

- `Agent` instances:
  - `router_agent`: choose `python` or `general`
  - `python_agent`: technical/Python domain
  - `general_agent`: all other questions
  - `social_media_agent`: craft social copy
  - `image_agent`: write image prompt
  - `image_creation_agent`: conceptual step (image actually from OpenAI client)
- `Crew` and `Task` orchestration for each stage.
- OpenAI `ChatOpenAI` (gpt-4o-mini) and `OpenAI` for image creation.
- Uses `OPENAI_API_KEY` from environment.

## 🧩 Requirements

`requirements.txt`:
- `streamlit==1.41.1`
- `openai==1.59.7`
- `crewai==0.100.0`
- `langchain==0.3`
- `langchain-openai==0.2.0`
- `Pillow==11.0.0`

## 🚀 Quick Start (local)

1. Create and activate virtual env:
   - Windows: `python -m venv venv && .\\venv\\Scripts\\activate`
   - macOS/Linux: `python -m venv venv && source venv/bin/activate`
2. Install:
   - `pip install -r requirements.txt`
3. Set OpenAI API key:
   - Windows: `set OPENAI_API_KEY=your_api_key`
   - macOS/Linux: `export OPENAI_API_KEY=your_api_key`
4. Run:
   - `streamlit run app.py`
5. Open browser at `http://localhost:8501`.

## 🐳 Docker

Build image:

```bash
docker build -t crewai-streamlit-app .