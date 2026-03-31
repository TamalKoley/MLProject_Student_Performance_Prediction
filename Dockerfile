FROM python:3.8
WORKDIR /app

RUN apt update -y && apt install awscli -y

COPY requirements.txt /app/
COPY setup.py /app/

RUN pip install -r requirements.txt

COPY . /app/

CMD ["python","application.py"]