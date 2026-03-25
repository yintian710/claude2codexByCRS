from fastapi import FastAPI

from src.claude_crs_proxy.routes.messages import router as messages_router

app = FastAPI()
app.include_router(messages_router)
