import logging
from pathlib import Path
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from API.data_routers import book_router, data_router
from API.vector_store_check import vector_store_router
from API.create_eval_routes import create_eval_router
from API.orchestration_routes import orchestration_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT_DIR = Path(__file__).parent
config_file = ROOT_DIR / "setup.yaml"
with open(config_file, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

port = config.get("Port", 8000)

app = FastAPI(
    title="Story Generation and Evaluation API",
    description="Book search, data pipeline, vector store, and orchestration (run pipeline, generate/evaluate stories).",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(book_router)
app.include_router(data_router)
app.include_router(vector_store_router)
app.include_router(orchestration_router)
app.include_router(create_eval_router)


orchestration_docs_app = FastAPI(title="Orchestration API", description="Pipeline-only endpoints.")
orchestration_docs_app.include_router(orchestration_router)

create_eval_docs_app = FastAPI(title="Create Eval API", description="Story creation and evaluation endpoints.")
create_eval_docs_app.include_router(create_eval_router)

vector_docs_app = FastAPI(title="Vector Store API", description="Vector store query/check/list/health endpoints.")
vector_docs_app.include_router(vector_store_router)

book_docs_app = FastAPI(title="Book API", description="Book search/download and metadata check/update endpoints.")
book_docs_app.include_router(book_router)

data_docs_app = FastAPI(title="Data API", description="Data merge and status endpoints.")
data_docs_app.include_router(data_router)

@app.get("/")
def root():
    return {
        "message": "Story Generation and Evaluation API",
        "docs": f"http://localhost:{port}/docs",
        "other_docs": {
            "create_eval": f"http://localhost:{port}/create-eval",
            "vector": f"http://localhost:{port}/vector",
            "book": f"http://localhost:{port}/book-docs",
            "data": f"http://localhost:{port}/data-docs",
        },
    }

@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def orchestration_docs():
    return get_swagger_ui_html(
        openapi_url="/orchestration/openapi.json",
        title="Orchestration API Docs",
    )

@app.get("/orchestration/openapi.json", include_in_schema=False)
def orchestration_openapi():
    return orchestration_docs_app.openapi()

@app.get("/vector", response_class=HTMLResponse, include_in_schema=False)
def vector_docs():
    return get_swagger_ui_html(
        openapi_url="/vector/openapi.json",
        title="Vector Store API Docs",
    )

@app.get("/vector/openapi.json", include_in_schema=False)
def vector_openapi():
    return vector_docs_app.openapi()

@app.get("/create-eval", response_class=HTMLResponse, include_in_schema=False)
def create_eval_docs():
    return get_swagger_ui_html(
        openapi_url="/create-eval/openapi.json",
        title="Create Eval API Docs",
    )

@app.get("/create-eval/openapi.json", include_in_schema=False)
def create_eval_openapi():
    return create_eval_docs_app.openapi()


@app.get("/book-docs", response_class=HTMLResponse, include_in_schema=False)
def book_docs():
    return get_swagger_ui_html(
        openapi_url="/book/openapi.json",
        title="Book API Docs",
    )

@app.get("/book/openapi.json", include_in_schema=False)
def book_openapi():
    return book_docs_app.openapi()


@app.get("/data-docs", response_class=HTMLResponse, include_in_schema=False)
def data_docs():
    return get_swagger_ui_html(
        openapi_url="/data/openapi.json",
        title="Data API Docs",
    )

@app.get("/data/openapi.json", include_in_schema=False)
def data_openapi():
    return data_docs_app.openapi()

if __name__ == "__main__":
    logging.info("Starting server on port %s", port)
    base = f"http://localhost:{port}"
    print(f"\n  {base}\n  Docs: {base}/docs\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
