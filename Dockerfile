# Create a file named Dockerfile and paste this content
FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit-rag-app.py"]