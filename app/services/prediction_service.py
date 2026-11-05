
import numpy as np
import pandas as pd

from app.ml.model_manager import ModelManager
from app.ml.feature_engineering import FeatureEngineering


class PredictionService:
    """
    Orquestra o pré-processamento e a inferência
    do modelo de cancelamento de reservas.
    """

    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.feature_engineering = FeatureEngineering()

    def predict(self, reservations: list[dict]) -> list[dict]:
        """
        Recebe reservas e retorna a classificação
        e a probabilidade de cancelamento.
        """

        if not self.manager.is_loaded:
            raise RuntimeError(
                "Os artefatos de Machine Learning não foram carregados."
            )

        if not reservations:
            raise ValueError(
                "É necessário informar pelo menos uma reserva."
            )

        # Colunas utilizadas no treinamento do pré-processador
        required_columns = list(
            self.manager.preprocessor.feature_names_in_
        )

        # Atributos criados pelo Feature Engineering
        derived_columns = {
            "total_noites",
            "total_hospedes",
            "possui_criancas",
            "total_reservas_anteriores",
            "antecedencia_por_noite"
        }

        original_columns = [
            column
            for column in required_columns
            if column not in derived_columns
        ]

        # Verifica campos ausentes em cada reserva
        missing_by_row = [
            {
                "reserva": idx,
                "campos": sorted(
                    set(original_columns) - set(item)
                )
            }
            for idx, item in enumerate(reservations)
            if set(original_columns) - set(item)
        ]

        if missing_by_row:
            raise ValueError(
                f"Campos obrigatórios ausentes: {missing_by_row}"
            )

        # Constrói o DataFrame
        data = pd.DataFrame.from_records(reservations)
        data = data.reindex(columns=original_columns)
        data = data.replace({None: np.nan})

        # Aplica Feature Engineering
        data = self.feature_engineering.transform(data)

        # Garante as colunas e a ordem esperadas
        missing_after_engineering = (
            set(required_columns) - set(data.columns)
        )

        if missing_after_engineering:
            raise ValueError(
                "Atributos ausentes após Feature Engineering: "
                f"{sorted(missing_after_engineering)}"
            )

        data = data.reindex(columns=required_columns)

        # Aplica o pré-processamento treinado
        transformed_data = (
            self.manager.preprocessor.transform(data)
        )

        # Recupera os nomes das colunas transformadas
        transformed_df = pd.DataFrame(
            transformed_data,
            columns=self.manager.feature_names,
            index=data.index
        )

        # Seleção fixa das variáveis definidas pelo Boruta
        selected_data = transformed_df[
            self.manager.selected_features
        ]

        # Executa a inferência
        predictions = self.manager.model.predict(
            selected_data
        )

        probabilities = self.manager.model.predict_proba(
            selected_data
        )

        # Identifica a classe positiva
        classes = list(self.manager.model.classes_)

        if 1 not in classes:
            raise ValueError(
                "O modelo não contém a classe positiva (1)."
            )

        positive_idx = classes.index(1)

        # Organiza as respostas
        return [
            {
                "cancelada": int(label),
                "probabilidade_cancelamento": float(
                    proba[positive_idx]
                )
            }
            for label, proba in zip(
                predictions,
                probabilities
            )
        ]
