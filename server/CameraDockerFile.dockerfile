# Use the official ultralytics/ultralytics:latest image as base
# This base image typically includes Python and the ultralytics library
FROM ultralytics/ultralytics:latest

# Set the working directory inside the container.
# The ultralytics images often use /usr/src/app as a default, but /app is a common and clean choice.
# Let's explicitly set it to /app for clarity and consistent pathing.
WORKDIR /app

# Copy your Python server script into the container's working directory.
# Replace 'tower_protocol.py' with the actual name of your server file.
COPY receiver.py .

# Copy your trained YOLOv11 model into the container's working directory.
# This assumes 'yolov11best.pt' is in the same directory as your Dockerfile on your host machine.
COPY yolov11sbest.pt .

# Expose the UDP port 9876 that your application listens on
EXPOSE 9876/udp

# Command to run your Python server script when the container launches.
# Make sure 'tower_protocol.py' is the correct filename of your server script.
CMD ["python", "receiver.py"]
