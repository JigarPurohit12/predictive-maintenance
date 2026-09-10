# Python 3.11 as the spec requires. The development venv is 3.12 because 3.11
# was not available on the development machine; this image is what pins the
# version that actually matters.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # SageMaker script mode looks here for the entry point.
    SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code \
    PYTHONPATH=/opt/ml/code

WORKDIR /opt/ml/code

# libgomp is XGBoost's OpenMP runtime; it is not in python:slim and XGBoost
# fails to import without it, which is a confusing way to lose a training job.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a code change does not reinstall the world.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY config.py mypy.ini pytest.ini ./
COPY sql/ ./sql/
COPY data_layer/ ./data_layer/
COPY features/ ./features/
COPY training/ ./training/
COPY serving/ ./serving/
COPY monitoring/ ./monitoring/

# Fail the build rather than a job if the package graph is broken.
RUN python -c "import config, features.build_features, training.train, serving.inference"

# Training and inference are both driven by SageMaker, which overrides this.
# The default is the training path, so `docker run` does something sensible.
ENTRYPOINT ["python", "-m"]
CMD ["training.train"]
