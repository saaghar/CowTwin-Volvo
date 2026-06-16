import socket
import json
import threading
import time
import struct
import cv2
import numpy as np
import os
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from ultralytics import YOLO
import sys

# --- DATACLASSES ---
@dataclass
class RequestForSignDetection:
    TotalNumberOfFragments: int
    UniqueSequencedID: str # This will be the trucklat_trucklon_distancetosign
    SignGpsPosition: Optional[str] = None # This will be signlat_signlon

@dataclass
class ResponseForSignDetection:
    IsSignDetected: bool
    UniqueSequencedID: str

@dataclass
class SendImageMessage:
    UniqueSequencedID: str
    fragmentSequenceID: int
    fragment: str
# --- END DATACLASSES ---

# --- ImageAssembler Class ---
class ImageAssembler:
    def __init__(self):
        self.fragments: Dict[str, List[str]] = {}
        self.expected_fragments: Dict[str, int] = {}
        self.start_times: Dict[str, datetime] = {}

    def add_fragment(self, unique_id: str, fragment_id: int, fragment_data_base64: str, total_fragments: int):
        if unique_id not in self.fragments:
            self.fragments[unique_id] = [None] * total_fragments
            self.expected_fragments[unique_id] = total_fragments
            self.start_times[unique_id] = datetime.now()

        if 0 <= fragment_id < total_fragments:
            self.fragments[unique_id][fragment_id] = fragment_data_base64
        else:
            print(f"WARNING: Fragment ID {fragment_id} out of bounds for ID {unique_id} (expected {total_fragments})")

    def is_complete(self, unique_id: str) -> bool:
        if unique_id not in self.fragments:
            return False

        fragments_list = self.fragments[unique_id]
        return all(fragment is not None for fragment in fragments_list)

    def get_complete_image(self, unique_id: str) -> Optional[bytes]:
        if not self.is_complete(unique_id):
            return None

        fragments_list = self.fragments[unique_id]
        try:
            complete_image_bytes = b''.join([base64.b64decode(f) for f in fragments_list])
        except Exception as e:
            print(f"BASE64 DECODE ERROR for ID {unique_id}: {e}")
            return None

        # Clean up data for the assembled image
        if unique_id in self.fragments:
            del self.fragments[unique_id]
        if unique_id in self.expected_fragments:
            del self.expected_fragments[unique_id]
        if unique_id in self.start_times:
            del self.start_times[unique_id]

        return complete_image_bytes

    def cleanup_expired(self, max_age_seconds: int = 5):
        """Cleans up incomplete image assemblies older than max_age_seconds"""
        current_time = datetime.now()
        expired_ids = []

        for unique_id, start_time in list(self.start_times.items()): # Use list() to allow modification during iteration
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

# --- SignDetectionEngine Class ---
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

# --- TowerProtocol Class ---
class TowerProtocol:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.image_assembler = ImageAssembler()
        self.pending_requests: Dict[str, RequestForSignDetection] = {}
        self.request_timestamps: Dict[str, datetime] = {}
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

        self.socket.settimeout(1.0) # Short timeout to keep cleanup responsive
        self._message_loop()

    def stop(self):
        self.running = False
        self.socket.close()
        print("SERVER: Tower protocol stopped.")

    def _cleanup_worker(self):
        while self.running:
            time.sleep(1)
            self.image_assembler.cleanup_expired()
            # Also clean up expired pending requests
            current_time = datetime.now()
            expired_request_ids = []
            for unique_id, timestamp in list(self.request_timestamps.items()):
                if (current_time - timestamp).total_seconds() > self.max_response_time:
                    expired_request_ids.append(unique_id)
            for unique_id in expired_request_ids:
                if unique_id in self.pending_requests:
                    print(f"CLEANUP: Expired pending request for ID {unique_id} removed.")
                    del self.pending_requests[unique_id]
                if unique_id in self.request_timestamps:
                    del self.request_timestamps[unique_id]

    def _message_loop(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536) # Max UDP packet size
                self._process_incoming_packet(data, addr)
            except socket.timeout:
                pass # No data, just continue loop
            except socket.error as e:
                if self.running:
                    print(f"ERROR: Socket error: {e}")
                break

    def _process_incoming_packet(self, data: bytes, addr: tuple):
        """
        Parses incoming UDP packets. Both initial requests and fragments are now JSON.
        """
        try:
            message_json = data.decode('utf-8')
            message_dict = json.loads(message_json)

            if 'TotalNumberOfFragments' in message_dict and 'UniqueSequencedID' in message_dict:
                # This is an initial RequestForSignDetection
                threading.Thread(target=self._handle_sign_detection_request, args=(message_dict, addr), daemon=True).start()
            elif 'fragmentSequenceID' in message_dict and 'fragment' in message_dict and 'UniqueSequencedID' in message_dict:
                # This is an image fragment (SendImageMessage)
                threading.Thread(target=self._handle_image_fragment, args=(message_dict, addr), daemon=True).start()
            else:
                print(f"WARNING: Received unknown or malformed JSON packet from {addr}: {message_json}")

        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"ERROR: Received non-JSON or invalid JSON packet from {addr}: {e}. Data: {data[:50]}...")
        except Exception as e:
            print(f"ERROR: Failed to process incoming packet from {addr}: {e}")

    def _handle_sign_detection_request(self, message_dict: dict, addr: tuple):
        sign_gps_position = message_dict.get('SignGpsPosition')

        request = RequestForSignDetection(
            TotalNumberOfFragments=message_dict['TotalNumberOfFragments'],
            UniqueSequencedID=message_dict['UniqueSequencedID'],
            SignGpsPosition=sign_gps_position
        )

        unique_id = request.UniqueSequencedID
        self.pending_requests[unique_id] = request
        self.request_timestamps[unique_id] = datetime.now()

        print(f"REQUEST {unique_id}: Received sign detection request. Expecting {request.TotalNumberOfFragments} fragments.")
        if sign_gps_position:
            print(f"  Calculated Sign GPS: {sign_gps_position}")

        timer_thread = threading.Thread(
            target=self._response_timer,
            args=(unique_id, addr),
            daemon=True
        )
        timer_thread.start()

    def _handle_image_fragment(self, message_dict: dict, addr: tuple):
        try:
            fragment_message = SendImageMessage(
                UniqueSequencedID=message_dict['UniqueSequencedID'],
                fragmentSequenceID=message_dict['fragmentSequenceID'],
                fragment=message_dict['fragment']
            )

            unique_id = fragment_message.UniqueSequencedID
            fragment_id = fragment_message.fragmentSequenceID
            fragment_data_base64 = fragment_message.fragment

            # Only add fragment if there's a pending request for this unique_id
            if unique_id not in self.pending_requests:
                # print(f"WARNING: Received fragment for unknown/expired request ID {unique_id}. Discarding.")
                return

            request = self.pending_requests[unique_id]

            self.image_assembler.add_fragment(
                unique_id,
                fragment_id,
                fragment_data_base64,
                request.TotalNumberOfFragments
            )

            if self.image_assembler.is_complete(unique_id):
                if unique_id in self.pending_requests: # Double check to prevent race conditions
                    print(f"IMAGE ASSEMBLY: All fragments received for ID {unique_id}. Processing image...")
                    threading.Thread(target=self._process_complete_image, args=(unique_id, addr), daemon=True).start()

        except Exception as e:
            print(f"ERROR: Failed to process image fragment from {addr}: {e}")

    def _process_complete_image(self, unique_id: str, addr: tuple):
        request = self.pending_requests.get(unique_id)
        if request is None:
            print(f"ERROR: Request for ID {unique_id} not found during image processing (might have been processed or timed out).")
            return

        complete_image_bytes = self.image_assembler.get_complete_image(unique_id)

        if complete_image_bytes is None:
            print(f"IMAGE ASSEMBLY ERROR: Failed to assemble image for ID {unique_id}. Sending negative response.")
            self._send_response(unique_id, False, addr)
            self._clean_request_data(unique_id)
            return

        print(f"IMAGE {unique_id}: Assembled {len(complete_image_bytes)} bytes.")
        sign_gps_from_request = request.SignGpsPosition
        if sign_gps_from_request:
            print(f"IMAGE {unique_id}: Associated Sign GPS: {sign_gps_from_request}")


        # --- Perform sign detection ---
        is_sign_detected, highest_confidence_sign_type, all_detected_signs = self.sign_detector.detect_signs(
            complete_image_bytes
        )

        # --- Save ALL images for debugging, regardless of detection ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"image_{unique_id}_{timestamp}.jpeg"
        image_file_path = os.path.join(self.output_folder, image_filename)

        try:
            with open(image_file_path, "wb") as f:
                f.write(complete_image_bytes)
            print(f"IMAGE {unique_id}: Saved to {image_file_path} (for debugging).")

        except Exception as e:
            print(f"IMAGE SAVE ERROR {unique_id}: {e}")

        # --- Now, save detection results ONLY if a sign was detected ---
        if is_sign_detected:
            if all_detected_signs:
                results_filename = os.path.splitext(image_filename)[0] + ".txt"
                results_file_path = os.path.join(self.output_results_folder, results_filename)

                try:
                    with open(results_file_path, "w") as f:
                        # Write the unique ID (trucklat_trucklon_distancetosign) on the first line
                        f.write(f"{unique_id}\n")
                        # Write each detected sign with its type, confidence, and the calculated sign GPS
                        for sign_type, conf in all_detected_signs:
                            # Ensure sign_gps_from_request is not None before including it
                            sign_gps_str = sign_gps_from_request if sign_gps_from_request else "N/A"
                            f.write(f"{sign_type}, {conf:.2f}, {sign_gps_str}\n")
                    print(f"RESULTS {unique_id}: Saved to {results_file_path} (sign detected).")
                except Exception as e:
                    print(f"RESULTS SAVE ERROR {unique_id}: {e}")
        else:
            print(f"IMAGE {unique_id}: No sign detected. Results NOT saved.")

        self._send_response(unique_id, is_sign_detected, addr)
        self._clean_request_data(unique_id)
    def _response_timer(self, unique_id: str, addr: tuple):
        start_time = self.request_timestamps.get(unique_id)
        if start_time is None:
            return

        elapsed_time = (datetime.now() - start_time).total_seconds()
        time_to_wait = self.max_response_time - elapsed_time

        if time_to_wait > 0:
            time.sleep(time_to_wait)

        if unique_id in self.pending_requests: # Check if still pending (not processed yet)
            print(f"TIMEOUT {unique_id}: Response not sent within {self.max_response_time}s. Sending negative response.")
            self._send_response(unique_id, False, addr)
            self._clean_request_data(unique_id)
            # Also ensure cleanup of incomplete image data for this ID if it timed out
            if unique_id in self.image_assembler.fragments:
                print(f"CLEANUP: Incomplete image assembly for ID {unique_id} removed due to timeout.")
                # Directly remove from assembler as cleanup_expired is a separate process
                del self.image_assembler.fragments[unique_id]
                if unique_id in self.image_assembler.expected_fragments:
                    del self.image_assembler.expected_fragments[unique_id]
                if unique_id in self.image_assembler.start_times:
                    del self.image_assembler.start_times[unique_id]

    def _send_response(self, unique_id: str, is_detected: bool, addr: tuple):
        response_dict = {
            'IsSignDetected': is_detected,
            'UniqueSequencedID': unique_id
        }

        response_json = json.dumps(response_dict)
        response_data = response_json.encode('utf-8')

        try:
            self.socket.sendto(response_data, addr)
            print(f"RESPONSE {unique_id}: Sent. Detection: {is_detected}")
        except Exception as e:
            print(f"ERROR: Failed to send response for ID {unique_id} to {addr}: {e}")

    def _clean_request_data(self, unique_id: str):
        if unique_id in self.pending_requests:
            del self.pending_requests[unique_id]
        if unique_id in self.request_timestamps:
            del self.request_timestamps[unique_id]

# --- Main Function ---
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