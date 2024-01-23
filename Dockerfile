FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install --ignore-installed  Flask joblib pandas scikit-learn numpy


EXPOSE 123

CMD ["python3", "api.py"]
