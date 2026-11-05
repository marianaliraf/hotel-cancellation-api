"""Teste HTTP da API em execução: python apiTest.py."""

import argparse
import json
import sys

import httpx


RESERVA = {
    "hotel": "Resort Hotel",
    "lead_time": 14,
    "arrival_date_year": 2016,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": None,
    "babies": 0,
    "meal": "BB",
    "country": None,
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "agent": None,
    "company": None,
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 1,
}


def validar_predicoes(response: httpx.Response, quantidade: int) -> list[dict]:
    response.raise_for_status()
    predicoes = response.json().get("predicoes")
    if not isinstance(predicoes, list) or len(predicoes) != quantidade:
        raise ValueError("Quantidade de predições diferente da esperada.")
    for predicao in predicoes:
        if not isinstance(predicao, dict):
            raise ValueError("Formato de predição inválido.")
        classe = predicao.get("cancelada")
        probabilidade = predicao.get("probabilidade_cancelamento")
        if type(classe) is not int or classe not in (0, 1):
            raise ValueError("A classe deve ser 0 ou 1.")
        if type(probabilidade) not in (int, float) or not 0 <= probabilidade <= 1:
            raise ValueError("A probabilidade deve estar entre 0 e 1.")
    return predicoes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL base da API")
    args = parser.parse_args()

    try:
        with httpx.Client(base_url=args.url.rstrip("/"), timeout=30.0) as client:
            response = client.get("/health")
            response.raise_for_status()
            if response.json() != {"status": "ok"}:
                raise ValueError("Resposta inesperada do endpoint /health.")
            print("[OK] GET /health")

            response = client.post("/predict", json={"reservas": [RESERVA]})
            individual = validar_predicoes(response, 1)
            print("[OK] Predição individual:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))

            response = client.post("/predict", json={"reservas": [RESERVA, RESERVA]})
            lote = validar_predicoes(response, 2)
            if lote != individual * 2:
                raise ValueError("As predições em lote diferem da predição individual.")
            print("[OK] Predição em lote consistente")

            response = client.post("/predict", json={"reservas": []})
            if response.status_code != 422:
                raise ValueError(f"Lista vazia: esperado HTTP 422, recebido {response.status_code}.")
            print("[OK] Lista vazia rejeitada com HTTP 422")

        print("Todos os testes passaram.")
        return 0
    except httpx.ConnectError:
        print(f"[ERRO] Não foi possível conectar a {args.url}. Inicie a API.", file=sys.stderr)
    except httpx.HTTPStatusError as exc:
        print(f"[ERRO] HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
    except (httpx.RequestError, ValueError) as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
