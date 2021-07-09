FROM python:3.9-slim-buster

COPY . /app

WORKDIR /app

RUN ls -la
RUN ls data
RUN apt-get update
RUN apt-get -y install gcc

COPY requirements.txt requirements.txt 
RUN  pip3 install -r requirements.txt

EXPOSE  8501

CMD ["streamlit", "run", "data_visualization_app.py"]























