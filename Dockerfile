# The API: CPU-only PyTorch, the two published checkpoints, and a tracking
# database backfilled from experiment1's results while the image builds.
FROM python:3.12-slim

# Hugging Face Spaces runs containers as uid 1000. Building as that user
# means the database written below is still writable when the app runs.
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH" \
    PIP_NO_CACHE_DIR=1

COPY --chown=user requirements-cpu.txt requirements.txt ./
RUN pip install --user -r requirements-cpu.txt && pip install --user -r requirements.txt

COPY --chown=user src ./src
COPY --chown=user runs/multiseed/experiment1 ./runs/multiseed/experiment1
RUN python -m src.backfill --experiment-dir runs/multiseed/experiment1 --db experiments.db

ENV VISION_SIMPLE_RUN_DIR=runs/multiseed/experiment1/simple_s42 \
    VISION_STRONGER_RUN_DIR=runs/multiseed/experiment1/stronger_s42 \
    VISION_DEVICE=cpu \
    VISION_DB_PATH=experiments.db

# 7860 is the port Hugging Face Spaces routes to (app_port in its README).
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
