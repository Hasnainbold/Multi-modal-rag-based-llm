FROM python:3.9

MAINTAINER pranavrao25

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
  && apt-get -y install tesseract-ocr

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","rag.py"]