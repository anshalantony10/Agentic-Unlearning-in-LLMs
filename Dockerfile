# Create a file named Dockerfile and paste this content
FROM python:3.8
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "moa-with-rag.py"]