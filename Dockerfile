FROM python:3.11.2

RUN python3 -m venv /opt/venv

# Copy all the repo (it will copy everything)
COPY . .

# Install dependencies:
#COPY requirements.txt .
#COPY setup.py .

#RUN . /opt/venv/bin/activate && pip install -e .
RUN . /opt/venv/bin/activate && pip install -r requirements.txt && pip install -e .

# Run the application:
#CMD . /opt/venv/bin/activate && exec pip list
CMD . /opt/venv/bin/activate && exec python src/predict.py
