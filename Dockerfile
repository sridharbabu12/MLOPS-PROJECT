FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Add the current directory to Python path
ENV PYTHONPATH=/app

EXPOSE 5000

CMD ["python", "application.py"]
