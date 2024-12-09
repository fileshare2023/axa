# Dockerfile
FROM python:3.8

WORKDIR /app

COPY . /app

RUN pip install gradio scikit-learn numpy

EXPOSE 7860

CMD ["python", "dbapp.py"]
