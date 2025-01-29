from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from api.routes import router
from database.database import engine
from utils.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()

app = FastAPI(
    title="Trading Analysis API",
    description="API für Trading-Analysen und Marktdaten",
    version="1.0.0"
)

# CORS-Konfiguration
origins = [
    "http://localhost:3000",  # React Development Server
    "http://localhost:5173",  # Vite Development Server
    "http://localhost:4173",  # Vite Preview Server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    logger.info("Server wird gestartet...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server wird heruntergefahren...")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "localhost")
    port = int(os.getenv("API_PORT", "5000"))
    
    logger.info(f"Server wird auf {host}:{port} gestartet...")
    uvicorn.run("main:app", host=host, port=port, reload=True) 