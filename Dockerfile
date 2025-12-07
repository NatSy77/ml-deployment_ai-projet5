FROM python:3.11-slim

# Créer un utilisateur non-root (bonne pratique)
RUN useradd -m -u 1000 user
USER user

# Dossier de travail
WORKDIR /app

# Ajouter le PATH local pip
ENV PATH="/home/user/.local/bin:$PATH"

# Copier le requirements et installer les dépendances
COPY --chown=user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier tout le projet dans /app
COPY --chown=user . /app

# Lancer FastAPI avec uvicorn
# ⚠️ Hugging Face Spaces attend le port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
