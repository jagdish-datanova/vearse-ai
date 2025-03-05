# Use an official Python runtime as a parent image
FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /app
# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application code to the container
COPY . /app/
# Expose the port your Flask app runs on (default is 5001 in your code)
EXPOSE 5001
# Set environment variables if needed
# ENV DATABASE_URL="your_database_url"
# ENV OPENAI_API_KEY="your_openai_api_key"
# Run the application
CMD ["python", "vearse-ai-poc.py"]