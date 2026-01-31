# Hugging Face Spaces - Ethiopian Crop Recommendation System
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.10-slim

# Create user for HF Spaces (runs as UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy app code and assets
COPY --chown=user src/ src/
COPY --chown=user static/ static/
COPY --chown=user models/ models/

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
