from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import get_settings
from .controllers import health_controller, account_controller, notebook_controller, job_controller, dataset_controller
from .services.job_worker import worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    await worker.recover()
    try:
        yield
    finally:
        await worker.shutdown()


app=FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(health_controller.router)
app.include_router(account_controller.router)
app.include_router(notebook_controller.router)
app.include_router(job_controller.router)
app.include_router(dataset_controller.router)
