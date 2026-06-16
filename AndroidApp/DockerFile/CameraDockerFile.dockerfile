# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY receiver.py .

# Expose the UDP port 9876
EXPOSE 9876/udp

# Run the Python script when the container launches
CMD ["python", "receiver.py"]



