import socket
import json
import threading
import time
import struct
import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from ultralytics import YOLO
import sys

# --- Dataclasses (unchanged) ---
@dataclass
class RequestForSignDetection:
    TotalNumberOfFragments: int
    Lat: float
    Lon: float
    Distance: float
    UniqueSequencedID: int

@dataclass
class ResponseForSignDetection:
    IsSignDetected: bool
    UniqueSequencedID: int

@dataclass
class SendImageMessage:
    UniqueSequencedID: int
    fragmentSequenceID: int
    fragment: bytes

# --- ImageAssembler Class (unchanged) ---
class ImageAssembler:
    def __init__(self):
        self.fragments: Dict[int, List[bytes]] = {}
        self.expected_fragments: Dict[int, int] = {}
        self.start_times: Dict[int, datetime] = {}

    def add_fragment(self, unique_id: int, fragment_id: int, fragment_data: bytes, total_fragments: int):
        if unique_id not in self.fragments:
            self.fragments[unique_id] = [None] * total_fragments
            self.expected_fragments[unique_id] = total_fragments
            self.start_times[unique_id] = datetime.now()

        if 0 <= fragment_id < total_fragments:
            self.fragments[unique_id][fragment_id] = fragment_data
        else:
            pass

    def is_complete(self, unique_id: int) -> bool:
        if unique_id not in self.fragments:
            return False

        fragments_list = self.fragments[unique_id]
        return all(fragment is not None for fragment in fragments_list)

    def get_complete_image(self, unique_id: int) -> Optional[bytes]:
        if not self.is_complete(unique_id):
            return None

        fragments_list = self.fragments[unique_id]
        complete_image = b''.join(fragments_list)

        del self.fragments[unique_id]
        del self.expected_fragments[unique_id]
        del self.start_times[unique_id]

        return complete_image

    def cleanup_expired(self, max_age_seconds: int = 5):
        """Cleans up incomplete image assemblies older than max_age_seconds"""
        current_time = datetime.now()
        expired_ids = []

        for unique_id, start_time in self.start_times.items():
            if (current_time - start_time).total_seconds() > max_age_seconds:
                expired_ids.append(unique_id)

        for unique_id in expired_ids:
            if unique_id in self.fragments:
                print(f"CLEANUP: Expired image assembly for ID {unique_id} removed.")
                del self.fragments[unique_id]
            if unique_id in self.expected_fragments:
                del self.expected_fragments[unique_id]
            if unique_id in self.start_times:
                del self.start_times[unique_id]

# --- SignDetectionEngine Class (unchanged) ---
class SignDetectionEngine:
    """Optimized sign detection engine"""

    def __init__(self, model_path: str = "/app/yolov11best.pt", confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.model = self.load_detection_model(model_path)

    def load_detection_model(self, model_path: str):
        """Loads the detection model (YOLO). This function returns a model object."""

        if not os.path.exists(model_path):
            print(f"MODEL LOAD ERROR: Model file not found at {model_path}. Using mock model.")
            class FallbackMockModel:
                def __call__(self, image_input, verbose=False, imgsz=640):
                    return [type('obj', (object,), {'boxes': [], 'names': {0: "mock_sign"}})]
            return FallbackMockModel()

        try:
            model = YOLO(model_path)
            print(f"MODEL: Successfully loaded YOLO model from {model_path}")
            return model

        except Exception as e:
            print(f"MODEL LOAD ERROR: Failed to load YOLO model from {model_path}: {e}. Using mock model.")
            class FallbackMockModel:
                def __call__(self, image_input, verbose=False, imgsz=640):
                    return [type('obj', (object,), {'boxes': [], 'names': {0: "mock_sign"}})]
            return FallbackMockModel()

    def decode_udp_packet_to_image(self, image_data_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decodes the image bytes from a UDP packet into an OpenCV image np array.
        Returns None if decoding fails or if the input is invalid.
        """
        try:
            np_arr = np.frombuffer(image_data_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                print("IMAGE DECODE ERROR: Could not decode image from bytes.")
            return image
        except Exception as e:
            print(f"IMAGE DECODE ERROR: {e}")
            return None

    def detect_signs(self, image_data_bytes: bytes) -> Tuple[bool, Optional[str], List[Tuple[str, float]]]:
        """
        Decodes the raw image bytes into an image, runs it through the detection model.
        Returns a tuple:
        (is_sign_detected: bool,
         highest_confidence_sign_type: Optional[str],
         all_detected_signs: List[Tuple[str, float]])
        where all_detected_signs contains (sign_type, confidence) for all signs
        detected above the confidence_threshold.
        """
        image = self.decode_udp_packet_to_image(image_data_bytes)

        if image is None:
            return False, None, []

        results = self.model(image, verbose=False, imgsz=640)

        is_sign_detected: bool = False
        highest_confidence_sign_type: Optional[str] = None
        highest_confidence_found = 0.0
        all_detected_signs: List[Tuple[str, float]] = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = box.cls.item()

                    class_name = r.names.get(int(cls), f"Unknown Class {int(cls)}")

                    if conf > self.confidence_threshold:
                        all_detected_signs.append((class_name, conf))

                        if conf > highest_confidence_found:
                            highest_confidence_found = conf
                            highest_confidence_sign_type = class_name
                            is_sign_detected = True

        if is_sign_detected:
            print(f"DETECTION SUMMARY: At least one sign detected. Highest: '{highest_confidence_sign_type}' (Conf: {highest_confidence_found:.2f})")
            if all_detected_signs:
                print("ALL DETECTED SIGNS (above threshold):")
                for sign_type, conf in all_detected_signs:
                    print(f"  - Type: '{sign_type}', Confidence: {conf:.2f}")
        else:
            print(f"DETECTION SUMMARY: No sign detected above threshold (Highest Conf: {highest_confidence_found:.2f})")

        return is_sign_detected, highest_confidence_sign_type, all_detected_signs

# --- TowerProtocol Class (MODIFIED) ---
class TowerProtocol:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.image_assembler = ImageAssembler()
        self.pending_requests: Dict[int, RequestForSignDetection] = {}
        self.request_timestamps: Dict[int, datetime] = {}
        self.max_response_time = 3.0
        self.output_folder = "images"
        self.output_results_folder = "results"

        self.sign_detector = SignDetectionEngine(
            model_path="/app/yolov11best.pt",
            confidence_threshold=0.4
        )

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.output_results_folder, exist_ok=True)
        print(f"SERVER STARTUP: Images (with signs) will be saved to: {os.path.abspath(self.output_folder)}")
        print(f"SERVER STARTUP: Detection results will be saved to: {os.path.abspath(self.output_results_folder)}")

    def start(self):
        self.socket.bind((self.host, self.port))
        self.running = True
        print(f"SERVER: Tower protocol started on {self.host}:{self.port}")

        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()

        self.socket.settimeout(1.0)
        self._message_loop()

    def stop(self):
        self.running = False
        self.socket.close()
        print("SERVER: Tower protocol stopped.")

    def _cleanup_worker(self):
        while self.running:
            time.sleep(1)
            self.image_assembler.cleanup_expired()

    def _message_loop(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                self._process_message_fast(data, addr)
            except socket.timeout:
                pass # No data, just continue loop
            except socket.error as e:
                if self.running:
                    print(f"ERROR: Socket error: {e}")
                break

    def _process_message_fast(self, data: bytes, addr: tuple):
        try:
            try:
                message_json = data.decode('utf-8')
                message_dict = json.loads(message_json)

                if all(key in message_dict for key in ['TotalNumberOfFragments', 'UniqueSequencedID']):
                    threading.Thread(target=self._handle_sign_detection_request, args=(message_dict, addr), daemon=True).start()
                    return
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

            if len(data) >= 12:
                self._handle_raw_bytes_fragment(data, addr)

        except Exception as e:
            print(f"ERROR: Failed to process incoming message from {addr}: {e}")

    def _handle_sign_detection_request(self, message_dict: dict, addr: tuple):
        request = RequestForSignDetection(
            TotalNumberOfFragments=message_dict['TotalNumberOfFragments'],
            Lat=message_dict['Lat'],
            Lon=message_dict['Lon'],
            Distance=message_dict['Distance'],
            UniqueSequencedID=message_dict['UniqueSequencedID']
        )

        unique_id = request.UniqueSequencedID
        self.pending_requests[unique_id] = request
        self.request_timestamps[unique_id] = datetime.now()

        print(f"REQUEST {unique_id}: Received sign detection request. Expecting {request.TotalNumberOfFragments} fragments.")

        timer_thread = threading.Thread(
            target=self._response_timer,
            args=(unique_id, addr),
            daemon=True
        )
        timer_thread.start()

    def _handle_raw_bytes_fragment(self, data: bytes, addr: tuple):
        try:
            unique_id = struct.unpack('>Q', data[0:8])[0]
            fragment_id = struct.unpack('>I', data[8:12])[0]
            fragment_data = data[12:]

            if unique_id not in self.pending_requests:
                return

            request = self.pending_requests[unique_id]

            self.image_assembler.add_fragment(
                unique_id,
                fragment_id,
                fragment_data,
                request.TotalNumberOfFragments
            )

            if self.image_assembler.is_complete(unique_id):
                # Process complete image in a new thread to avoid blocking main receive loop
                threading.Thread(target=self._process_complete_image, args=(unique_id, addr), daemon=True).start()

        except Exception as e:
            print(f"ERROR: Failed to process fragment: {e}")

    def _process_complete_image(self, unique_id: int, addr: tuple):
        request = self.pending_requests.get(unique_id)
        if request is None:
            print(f"ERROR: Request for ID {unique_id} not found during image processing.")
            return

        complete_image_bytes = self.image_assembler.get_complete_image(unique_id)

        if complete_image_bytes is None:
            print(f"IMAGE ASSEMBLY ERROR: Failed to assemble image for ID {unique_id}. Sending negative response.")
            self._send_response(unique_id, False, addr)
            self._clean_request_data(unique_id)
            return

        print(f"IMAGE {unique_id}: Assembled {len(complete_image_bytes)} bytes.")

        # --- Perform sign detection first ---
        is_sign_detected, highest_confidence_sign_type, all_detected_signs = self.sign_detector.detect_signs(
            complete_image_bytes
        )

        # --- ONLY SAVE IMAGE AND RESULTS IF A SIGN IS DETECTED ---
        if is_sign_detected:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"image_{unique_id}_{timestamp}.jpeg"
            image_file_path = os.path.join(self.output_folder, image_filename)

            try:
                with open(image_file_path, "wb") as f:
                    f.write(complete_image_bytes)
                print(f"IMAGE {unique_id}: Saved to {image_file_path} (sign detected).")

            except Exception as e:
                print(f"IMAGE SAVE ERROR {unique_id}: {e}")

            if all_detected_signs: # This check is redundant if `is_sign_detected` is true, but good for clarity
                results_filename = os.path.splitext(image_filename)[0] + ".txt"
                results_file_path = os.path.join(self.output_results_folder, results_filename)

                try:
                    with open(results_file_path, "w") as f:
                        f.write(f"{unique_id}\n")
                        for sign_type, conf in all_detected_signs:
                            f.write(f"{sign_type}, {conf:.2f}\n")
                    print(f"RESULTS {unique_id}: Saved to {results_file_path} (sign detected).")
                except Exception as e:
                    print(f"RESULTS SAVE ERROR {unique_id}: {e}")
        else:
            print(f"IMAGE {unique_id}: No sign detected. Image and results NOT saved.")
        # -----------------------------------------------------------

        self._send_response(unique_id, is_sign_detected, addr)
        self._clean_request_data(unique_id)


    def _response_timer(self, unique_id: int, addr: tuple):
        start_time = self.request_timestamps.get(unique_id)
        if start_time is None:
            return

        elapsed_time = (datetime.now() - start_time).total_seconds()
        time_to_wait = self.max_response_time - elapsed_time

        if time_to_wait > 0:
            time.sleep(time_to_wait)

        if unique_id in self.pending_requests:
            print(f"TIMEOUT {unique_id}: Response not sent within {self.max_response_time}s. Sending negative response.")
            self._send_response(unique_id, False, addr)
            self._clean_request_data(unique_id)
            if unique_id in self.image_assembler.fragments:
                print(f"CLEANUP: Incomplete image assembly for ID {unique_id} removed due to timeout.")
                del self.image_assembler.fragments[unique_id]
                del self.image_assembler.expected_fragments[unique_id]
                del self.image_assembler.start_times[unique_id]

    def _send_response(self, unique_id: int, is_detected: bool, addr: tuple):
        response = ResponseForSignDetection(
            IsSignDetected=is_detected,
            UniqueSequencedID=unique_id
        )

        response_dict = {
            'IsSignDetected': response.IsSignDetected,
            'UniqueSequencedID': response.UniqueSequencedID
        }

        response_json = json.dumps(response_dict)
        response_data = response_json.encode('utf-8')

        try:
            self.socket.sendto(response_data, addr)
            print(f"RESPONSE {unique_id}: Sent. Detection: {is_detected}")
        except Exception as e:
            print(f"ERROR: Failed to send response for ID {unique_id} to {addr}: {e}")


    def _clean_request_data(self, unique_id: int):
        if unique_id in self.pending_requests:
            del self.pending_requests[unique_id]
        if unique_id in self.request_timestamps:
            del self.request_timestamps[unique_id]

# --- Main Function (unchanged) ---
def main():
    """Main function to run the tower protocol"""
    tower = TowerProtocol()

    try:
        tower.start()
    except KeyboardInterrupt:
        print("\nSERVER: Shutting down gracefully...")
        tower.stop()
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled exception in main: {e}")
        tower.stop()

if __name__ == "__main__":
    main()