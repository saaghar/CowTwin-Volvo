import socket
import json
import threading
import time
import struct
import cv2
import numpy as np
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from ultralytics import YOLO
import sys

# --- Dataclasses ---
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
    DetectedSignType: Optional[str] = None  # New field for the detected sign type

@dataclass
class SendImageMessage:
    UniqueSequencedID: int
    fragmentSequenceID: int
    fragment: bytes

# --- ImageAssembler Class ---
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
            print(f"Warning: Fragment ID {fragment_id} out of bounds for Unique ID {unique_id} (Total: {total_fragments})")

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
        
        # Clean up immediately after getting the complete image
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
                del self.fragments[unique_id]
            if unique_id in self.expected_fragments:
                del self.expected_fragments[unique_id]
            if unique_id in self.start_times:
                del self.start_times[unique_id]
            print(f"Cleaned up expired image assembly for ID: {unique_id}")

# --- SignDetectionEngine Class ---
class SignDetectionEngine:
    """Optimized mock sign detection engine"""
    
    def __init__(self, model_path: str = "/app/yolov11best.pt", confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.model = self.load_detection_model(model_path)
    
    def load_detection_model(self, model_path: str):
        """Loads the detection model (YOLO). This function returns a model object."""

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}. Please ensure your model (e.g., yolov11best.pt) is in the correct directory.")
            print("Falling back to mock detection model.")
            class FallbackMockModel:
                def __call__(self, image_input, verbose=False, imgsz=1280): # Changed to match YOLO model's behavior
                    if image_input is None:
                        # Return a structure similar to YOLO results to avoid errors
                        return [type('obj', (object,), {'boxes': [], 'names': {}})]
                    print("Mock Model: Always returning no detections.")
                    # Include a 'names' dictionary to match the real YOLO model's output structure
                    return [type('obj', (object,), {'boxes': [], 'names': {0: "mock_sign", 1: "another_mock_sign"}})]
            return FallbackMockModel()

        try:
            model = YOLO(model_path)
            print(f"Successfully loaded YOLO model from {model_path}")
            return model
        
        except Exception as e:
            print(f"Error loading YOLO model from {model_path}: {e}")
            print("Falling back to mock detection model.")
            class FallbackMockModel:
                def __call__(self, image_input, verbose=False, imgsz=640): # Changed to match YOLO model's behavior
                    if image_input is None:
                        return [type('obj', (object,), {'boxes': [], 'names': {}})]
                    print("Mock Model (fallback): Always returning no detections.")
                    return [type('obj', (object,), {'boxes': [], 'names': {0: "mock_sign", 1: "another_mock_sign"}})]
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
                print("Could not decode image from bytes.")
            return image
        except Exception as e:
            print(f"Error decoding image from UDP packet: {e}")
            return None

    def detect_signs(self, image_data_bytes: bytes) -> Optional[str]:  # Changed return type to Optional[str]
        """
        Decodes the raw image bytes into an image, runs it through the detection model,
        and returns the class name of the detected sign if confidence is above the threshold,
        otherwise returns None.
        """
        # 1. Decode UDP packet to an image
        image = self.decode_udp_packet_to_image(image_data_bytes)
        
        if image is None:
            print("No valid image to process for detection.")
            return None

        # 2. Run the image through the model
        print(f"Running YOLO prediction on image with dimensions: {image.shape[1]}x{image.shape[0]} (width x height)")
        results = self.model(image, verbose=True, imgsz=640)

        detected_sign_type: Optional[str] = None  # Initialize as None
        highest_confidence_found = 0.0 

        print(f"YOLO raw results count: {len(results)}")
        
        # Iterate over results (Check the detection boxes)
        for r in results:
            print(f"  YOLO Result Object Info for {r.path if r.path else 'image in memory'}:")
            print(f"    Class names known by model: {r.names}")
            
            if r.boxes is not None:
                print(f"    Number of boxes detected: {len(r.boxes)}")
                for box in r.boxes:
                    conf = box.conf.item()
                    cls = box.cls.item()
                    xyxy = box.xyxy.tolist()[0]
                    
                    # Attempt to get the class name if it exists
                    class_name = r.names.get(int(cls), f"Unknown Class {int(cls)}")
                    
                    print(f"      Detected: {class_name}, Conf={conf:.4f}, Box={xyxy}")

                    # Update if the current detection has higher confidence and exceeds the threshold
                    if conf > self.confidence_threshold and conf > highest_confidence_found:
                        highest_confidence_found = conf
                        detected_sign_type = class_name  
                        break
            else:
                print("    No boxes detected in this result object.")

        # 3. Return the detected sign type or None
        if detected_sign_type:
            print(f"Detection result for image: Sign of type '{detected_sign_type}' detected (Highest confidence: {highest_confidence_found:.4f}, Threshold: {self.confidence_threshold:.4f})")
            return detected_sign_type
        else:
            print(f"Detection result for image: No sign detected above threshold (Highest confidence: {highest_confidence_found:.4f}, Threshold: {self.confidence_threshold:.4f})")
            return None

# --- TowerProtocol Class ---
class TowerProtocol:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.image_assembler = ImageAssembler()
        self.pending_requests: Dict[int, RequestForSignDetection] = {}
        self.request_timestamps: Dict[int, datetime] = {}
        self.max_response_time = 6  # 250ms timeout for faster response
        self.output_folder = "images"  # Define the output folder
        
        # Initialize SignDetectionEngine with the correct YOLOv11 model path
        self.sign_detector = SignDetectionEngine(
            model_path="/app/yolov11best.pt", # <--- Corrected model path for Docker container
            confidence_threshold=0.4
        ) 

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(self.output_folder)}")
        
    def start(self):
        """Starts the tower protocol server"""
        self.socket.bind((self.host, self.port))
        self.running = True
        print(f"Tower protocol started on {self.host}:{self.port}")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        # Start main message processing loop
        self.socket.settimeout(1.0) # Set a small timeout for recvfrom to allow graceful shutdown
        self._message_loop()
    
    def stop(self):
        """Stops the tower protocol server"""
        self.running = False
        self.socket.close()
        print("Tower protocol stopped")
    
    def _cleanup_worker(self):
        """Background worker to clean up expired image assemblies"""
        while self.running:
            time.sleep(1)
            self.image_assembler.cleanup_expired()
    
    def _message_loop(self):
        """Optimized message processing loop"""
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024) # 1MB buffer
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                self._process_message_fast(data, addr)
            except socket.timeout:
                # This is normal when a timeout is set, allows checking self.running
                pass
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
                break
    
    def _process_message_fast(self, data: bytes, addr: tuple):
        """Fast message processing without threading overhead for fragments"""
        try:
            # Try to parse as JSON first (for sign detection requests)
            try:
                message_json = data.decode('utf-8')
                message_dict = json.loads(message_json)
                
                # Check if this is a request for sign detection
                if all(key in message_dict for key in ['TotalNumberOfFragments', 'Lat', 'Lon', 'Distance', 'UniqueSequencedID']):
                    # Use threading only for requests, not fragments
                    threading.Thread(target=self._handle_sign_detection_request, args=(message_dict, addr), daemon=True).start()
                    return
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Not a JSON message, try to parse as raw bytes image fragment
                pass
            
            # Handle raw bytes image fragment directly (no threading)
            if len(data) >= 12:  # Minimum size for header (8 + 4 bytes for UniqueSequencedID and fragmentSequenceID)
                self._handle_raw_bytes_fragment(data, addr)
            
        except Exception as e:
            print(f"Error processing message from {addr}: {e}")
    
    def _handle_sign_detection_request(self, message_dict: dict, addr: tuple):
        """Handle incoming sign detection request"""
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
        
        print(f"Received sign detection request ID: {unique_id} from {addr}")
        print(f"Expecting {request.TotalNumberOfFragments} fragments")
        
        # Start a timer to ensure response within timeout
        timer_thread = threading.Thread(
            target=self._response_timer, 
            args=(unique_id, addr), 
            daemon=True
        )
        timer_thread.start()
    
    def _handle_raw_bytes_fragment(self, data: bytes, addr: tuple):
        """Optimized handler for raw bytes image fragments"""
        try:
            unique_id = struct.unpack('>Q', data[0:8])[0]
            fragment_id = struct.unpack('>I', data[8:12])[0]
            fragment_data = data[12:]
            
            if fragment_id % 10 == 0 or fragment_id == 0: # Print first and every 10th fragment
                print(f"Fragment {fragment_id} for ID: {unique_id} ({len(fragment_data)} bytes)")
            
            if unique_id not in self.pending_requests:
                print(f"Warning: Received fragment {fragment_id} for unknown/timed-out request ID: {unique_id}. Ignoring.")
                return
            
            request = self.pending_requests[unique_id]
            
            self.image_assembler.add_fragment(
                unique_id, 
                fragment_id, 
                fragment_data, 
                request.TotalNumberOfFragments
            )
            
            if self.image_assembler.is_complete(unique_id):
                self._process_complete_image(unique_id, addr)
                
        except Exception as e:
            print(f"Error processing fragment: {e}")
    
    def _process_complete_image(self, unique_id: int, addr: tuple):
        """Process complete image, save it, and send response"""
        request = self.pending_requests.get(unique_id)
        if request is None:
            print(f"Error: Request for ID {unique_id} not found when processing complete image.")
            return

        complete_image_bytes = self.image_assembler.get_complete_image(unique_id)
        
        if complete_image_bytes is None:
            print(f"Failed to get complete image for ID: {unique_id}")
            self._send_response(unique_id, False, None, addr) # Send None for DetectedSignType on error
            if unique_id in self.pending_requests:
                del self.pending_requests[unique_id]
            if unique_id in self.request_timestamps:
                del self.request_timestamps[unique_id]
            return
        
        print(f"Processing complete image for ID: {unique_id} ({len(complete_image_bytes)} bytes)")

        # Save the assembled image
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"image_{unique_id}_{timestamp}.jpeg" # Assuming JPEG for saving
            file_path = os.path.join(self.output_folder, filename)
            
            with open(file_path, "wb") as f:
                f.write(complete_image_bytes)
            print(f"Assembled image saved to: {file_path}")

            # Verification: Try to read the saved image with OpenCV
            test_read_image = cv2.imread(file_path)
            if test_read_image is None:
                print(f"CRITICAL ERROR: OpenCV could not read the saved image at {file_path}")
            else:
                print(f"SUCCESS: Saved image readable by OpenCV. Dimensions: {test_read_image.shape[1]}x{test_read_image.shape[0]}")

        except Exception as e:
            print(f"Error saving image for ID {unique_id}: {e}")

        # Perform sign detection using the updated SignDetectionEngine
        detected_sign_type = self.sign_detector.detect_signs(
            complete_image_bytes
        )
        
        # Send response
        self._send_response(unique_id, detected_sign_type is not None, detected_sign_type, addr)
        
        # Clean up
        if unique_id in self.pending_requests:
            del self.pending_requests[unique_id]
        if unique_id in self.request_timestamps:
            del self.request_timestamps[unique_id]
    
    def _response_timer(self, unique_id: int, addr: tuple):
        """Ensures response is sent within timeout"""
        start_time = self.request_timestamps.get(unique_id)
        if start_time is None:
            return # Request already processed or cleaned up

        elapsed_time = (datetime.now() - start_time).total_seconds()
        time_to_wait = self.max_response_time - elapsed_time

        if time_to_wait > 0:
            time.sleep(time_to_wait)
        
        # Check if response was already sent (i.e., if unique_id is still in pending_requests)
        if unique_id in self.pending_requests:
            print(f"Response timeout for ID: {unique_id}, sending negative response (No Detection)")
            self._send_response(unique_id, False, None, addr) # Send None for DetectedSignType on timeout
            
            # Clean up after sending timeout response
            if unique_id in self.pending_requests:
                del self.pending_requests[unique_id]
            if unique_id in self.request_timestamps:
                del self.request_timestamps[unique_id]
            # Also clean up from image_assembler if it hasn't completed
            if unique_id in self.image_assembler.fragments:
                print(f"Cleaning up incomplete image assembly for ID {unique_id} due to timeout.")
                del self.image_assembler.fragments[unique_id]
                del self.image_assembler.expected_fragments[unique_id]
                del self.image_assembler.start_times[unique_id]
    
    def _send_response(self, unique_id: int, is_detected: bool, detected_sign_type: Optional[str], addr: tuple):
        """Sends response back to vehicle"""
        response = ResponseForSignDetection(
            IsSignDetected=is_detected,
            UniqueSequencedID=unique_id,
            DetectedSignType=detected_sign_type # Include the detected sign type
        )
        
        response_dict = {
            'IsSignDetected': response.IsSignDetected,
            'UniqueSequencedID': response.UniqueSequencedID,
            'DetectedSignType': response.DetectedSignType # Add to the response dictionary
        }
        
        response_json = json.dumps(response_dict)
        response_data = response_json.encode('utf-8')
        
        self.socket.sendto(response_data, addr)
        print(f"Sent response for ID: {unique_id}, Detection: {is_detected}, Type: {detected_sign_type}")

# --- Main Function ---
def main():
    """Main function to run the tower protocol"""
    tower = TowerProtocol()
    
    try:
        tower.start()
    except KeyboardInterrupt:
        print("\nShutting down tower protocol...")
        tower.stop()

if __name__ == "__main__":
    main()