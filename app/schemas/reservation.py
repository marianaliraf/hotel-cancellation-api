
from pydantic import BaseModel, Field, ConfigDict


class ReservationInput(BaseModel):
    """
    Representa os atributos originais de uma reserva
    utilizados para previsão de cancelamento.
    """

    model_config = ConfigDict(extra="forbid")

    # Informações gerais da reserva
    hotel: str
    lead_time: int = Field(ge=0)

    # Data de chegada
    arrival_date_year: int = Field(ge=0)
    arrival_date_month: str
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)

    # Duração da hospedagem
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)

    # Quantidade de hóspedes
    adults: int = Field(ge=0)
    children: float | None = Field(default=None, ge=0)
    babies: int = Field(ge=0)

    # Informações comerciais
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str

    # Histórico do hóspede
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(ge=0)
    previous_bookings_not_canceled: int = Field(ge=0)

    # Características da reserva
    reserved_room_type: str
    deposit_type: str
    agent: float | None = None
    company: float | None = None
    customer_type: str

    # Informações financeiras e solicitações
    adr: float
    required_car_parking_spaces: int = Field(ge=0)
    total_of_special_requests: int = Field(ge=0)


class ReservationRequest(BaseModel):
    """
    Requisição contendo uma ou mais reservas.
    """

    reservas: list[ReservationInput] = Field(
        min_length=1,
        max_length=100,
        description="Lista de reservas para previsão de cancelamento."
    )


class ReservationPrediction(BaseModel):
    """
    Resultado da previsão de uma reserva.
    """

    cancelada: int = Field(ge=0, le=1)
    probabilidade_cancelamento: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    """
    Resposta contendo as previsões realizadas.
    """

    predicoes: list[ReservationPrediction]
