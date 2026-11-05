# Hotel Cancellation Prediction API

API FastAPI orientada a objetos para servir o **pipeline completo** do notebook `Projeto_ML_Cancelamento_Reservas_Hotel.ipynb`.

## 1. Preparação

Execute a célula **11 — Exportação do pipeline completo** do notebook e copie o arquivo `modelo_cancelamento_hotel.joblib` para `models/`.

**Importante:** o modelo treinado **não está incluído** neste projeto. A API não inicializa sem ele. O notebook exporta um `Pipeline` completo com etapas `feature_engineering`, `preprocessamento`, `selecao_fixa` e `classificador`. Não exporte um dicionário de componentes separados para usar esta implementação.

A classe `FeatureEngineering`, em `app/ml/feature_engineering.py`, cria os atributos usados na inferência por meio do método `transform`. As fórmulas devem permanecer iguais às da função `criar_atributos` do notebook; se alterar o treinamento, atualize também essa classe.

## 2. Instalação e execução

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

## 3. Endpoints

- `GET /health`: retorna `{"status": "ok"}` se a aplicação inicializou.
- `POST /predict`: recebe `{"reservas": [{...}]}` e devolve a classe `cancelada` e `probabilidade_cancelamento` para cada reserva.

Para obter um **payload real válido**, execute a célula 12 do notebook que imprime `exemplo_reserva` a partir de `X_teste_bruto`.

A API exige os atributos originais usados no treinamento, inclusive os necessários para criar os atributos derivados. As colunas são obtidas do pré-processador ajustado. Não envie `is_canceled` nem colunas de vazamento removidas no notebook. Valores ausentes podem ser `null`; os campos devem existir no JSON. Envie no máximo 100 reservas por chamada.

## 4. Testes

```bash
pytest -q
```

Os testes unitários não dependem do artefato treinado. O teste de integração com o `.joblib` precisa ser realizado após exportá-lo do notebook.

## 5. Notas

- Carregue arquivos `.joblib` somente de fontes confiáveis; o formato permite execução de código durante desserialização.
- Fixe as versões exatas das bibliotecas de treinamento e inferência para evitar incompatibilidade de serialização.
- Para produção, configure autenticação, limite de requisições, logs e observabilidade; não use `--reload`.
- A seleção de variáveis pelo Boruta é fixa e não é reexecutada na API.
