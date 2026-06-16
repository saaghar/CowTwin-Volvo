FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY receiver_new_save_all.py .

COPY yolov11best.pt .

EXPOSE 9876/udp

CMD ["python", "receiver_new_save_all.py"]
