
from pathlib import Path

import joblib


class ModelManager:
    """
    Gerencia o carregamento e a disponibilização dos
    artefatos de Machine Learning utilizados pela API.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

        self.model = None
        self.preprocessor = None
        self.selected_features = None
        self.feature_names = None

    def load_model(self) -> None:
        """
        Carrega e valida os artefatos exportados
        durante o treinamento do modelo.
        """

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.model_path}"
            )

        artifacts = joblib.load(self.model_path)

        required_keys = {
            "modelo",
            "preprocessador",
            "variaveis_selecionadas"
        }

        if not isinstance(artifacts, dict):
            raise TypeError(
                "O arquivo deve conter um dicionário de artefatos."
            )

        missing_keys = required_keys - set(artifacts)

        if missing_keys:
            raise ValueError(
                f"Artefatos ausentes: {sorted(missing_keys)}"
            )

        model = artifacts["modelo"]
        preprocessor = artifacts["preprocessador"]
        selected_features = list(
            artifacts["variaveis_selecionadas"]
        )

        # Valida os componentes carregados
        if not hasattr(model, "predict"):
            raise TypeError(
                "O modelo não possui o método predict()."
            )

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "O modelo não possui o método predict_proba()."
            )

        if not hasattr(preprocessor, "transform"):
            raise TypeError(
                "O pré-processador não possui o método transform()."
            )

        # Recupera os nomes dos atributos transformados
        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        # Verifica a compatibilidade com o Boruta
        missing_features = (
            set(selected_features) - set(feature_names)
        )

        if missing_features:
            raise ValueError(
                "Variáveis selecionadas não encontradas: "
                f"{sorted(missing_features)}"
            )

        if len(selected_features) != len(set(selected_features)):
            raise ValueError(
                "A lista de variáveis selecionadas contém duplicatas."
            )

        # Disponibiliza os artefatos após validação
        self.model = model
        self.preprocessor = preprocessor
        self.selected_features = selected_features
        self.feature_names = feature_names

    @property
    def is_loaded(self) -> bool:
        """
        Indica se todos os artefatos foram carregados.
        """
        return all(
            component is not None
            for component in (
                self.model,
                self.preprocessor,
                self.selected_features,
                self.feature_names
            )
        )
