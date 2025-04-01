FROM nipype/nipype:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --no-dev

COPY . /app

CMD ["python3", "run_workflow.py"]
