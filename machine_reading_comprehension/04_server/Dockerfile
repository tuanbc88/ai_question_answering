FROM python:3.9.13 as build

RUN mkdir -p /akagi2106/answer-extractor

WORKDIR /akagi2106/answer-extractor

COPY . /akagi2106/answer-extractor

RUN pip install -U -r requirements.txt

CMD python app.py