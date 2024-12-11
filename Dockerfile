FROM python:3.12.3-slim

# Set working directory
WORKDIR /app
ENV PORT = 3000
# Copy semua file ke dalam container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD ["python", "app.py"]
