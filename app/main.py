
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.ml.model_manager import ModelManager
from app.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia a inicialização e o encerramento da API.
    """

    # Carrega os artefatos treinados
    manager = ModelManager(settings.model_path)
    manager.load_model()

    # Inicializa o serviço de inferência
    app.state.prediction_service = PredictionService(manager)

    yield

    # Libera a referência ao serviço no encerramento
    del app.state.prediction_service


app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description=(
        "API para previsão de cancelamento de reservas de hotéis "
        "utilizando Random Forest, pré-processamento treinado "
        "e seleção fixa de atributos pelo Boruta."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
