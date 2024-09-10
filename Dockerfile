FROM python:3.11.2-slim
RUN useradd --create-home --shell /bin/bash app_user
WORKDIR /home/app_user
#COPY pyproject.toml ./
COPY . .
RUN pip install -e .
#RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
USER app_user
CMD ["bash"]
