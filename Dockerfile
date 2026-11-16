# Imagem base oficial leve do Python
FROM python:3.10-slim

# Evita geração de bytecode .pyc e buffer de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho na imagem
WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia os requisitos e instala as dependências Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação, notebook e dados do projeto
COPY app.py /app/
COPY Rice_Cammeo_Osmancik.arff /app/
COPY model_metadata.json /app/
COPY notebook_gmm_rice.ipynb /app/
COPY train_model.py /app/

# Executa o pipeline de treino no próprio build para gerar scaler.joblib e gmm_model.joblib
RUN python train_model.py

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Healthcheck para monitoramento da aplicação no container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando padrão para inicializar o Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
