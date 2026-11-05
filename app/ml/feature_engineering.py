"""Criação dos atributos usados no treinamento e na inferência."""

import pandas as pd


class FeatureEngineering:
    """Aplica as mesmas transformações determinísticas do notebook."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retorna uma cópia com os cinco atributos derivados da reserva."""
        x = data.copy()
        x["total_noites"] = (
            x["stays_in_weekend_nights"].fillna(0)
            + x["stays_in_week_nights"].fillna(0)
        )
        x["total_hospedes"] = (
            x["adults"].fillna(0)
            + x["children"].fillna(0)
            + x["babies"].fillna(0)
        )
        x["possui_criancas"] = (
            (x["children"].fillna(0) + x["babies"].fillna(0)) > 0
        ).astype(int)
        x["total_reservas_anteriores"] = (
            x["previous_cancellations"].fillna(0)
            + x["previous_bookings_not_canceled"].fillna(0)
        )
        x["antecedencia_por_noite"] = (
            x["lead_time"].fillna(0) / x["total_noites"].clip(lower=1)
        )
        return x
