
from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_prediction_service
from app.schemas.reservation import (
    PredictionResponse,
    ReservationRequest,
)
from app.services.prediction_service import PredictionService


router = APIRouter()


@router.get("/health", tags=["Saúde"])
def health():
    """
    Verifica se a API está respondendo.
    """
    return {"status": "ok"}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predição"],
)
def predict(
    payload: ReservationRequest,
    service: PredictionService = Depends(get_prediction_service),
):
    """
    Recebe reservas e retorna as previsões de cancelamento.
    """

    # Converte os objetos Pydantic para dicionários
    reservations = [
        reservation.model_dump()
        for reservation in payload.reservas
    ]

    try:
        predictions = service.predict(reservations)

        return {"predicoes": predictions}

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
