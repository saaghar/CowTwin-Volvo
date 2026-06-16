import socket
import json
import threading
import time
import struct
import os # Import the os module
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

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

# Modified to handle raw bytes instead of Base64
@dataclass
class SendImageMessage:
    UniqueSequencedID: int
    fragmentSequenceID: int
    fragment: bytes

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
        
        # Ensure fragment_id is within bounds to prevent IndexError
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
        
        # Clean up
        del self.fragments[unique_id]
        del self.expected_fragments[unique_id]
        del self.start_times[unique_id]
        
        return complete_image
    
    def cleanup_expired(self, max_age_seconds: int = 5):
        """Clean up incomplete image assemblies older than max_age_seconds"""
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

class SignDetectionEngine:
    """Optimized mock sign detection engine"""
    
    def detect_signs(self, image_data: bytes, lat: float, lon: float, distance: float) -> bool:
        # Extremely fast mock detection - no artificial delays
        import random
        
        # Quick detection based on image size
        detected = random.random() < 0.1
        print(f"Fast detection: {detected} ({len(image_data)} bytes)")
        
        return detected

class TowerProtocol:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.running = False
        self.image_assembler = ImageAssembler()
        self.sign_detector = SignDetectionEngine()
        self.pending_requests: Dict[int, RequestForSignDetection] = {}
        self.request_timestamps: Dict[int, datetime] = {}
        self.max_response_time = 0.25  # 250ms timeout for faster response
        self.output_folder = "images" # Define the output folder

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Images will be saved to: {os.path.abspath(self.output_folder)}")
        
    def start(self):
        """Start the tower protocol server"""
        self.socket.bind((self.host, self.port))
        self.running = True
        print(f"Tower protocol started on {self.host}:{self.port}")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        # Start main message processing loop
        self._message_loop()
    
    def stop(self):
        """Stop the tower protocol server"""
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
        # Increase socket buffer size for better performance
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB buffer
        
        while self.running:
            try:
                data, addr = self.socket.recvfrom(65536)
                # Process messages immediately without threading for fragments
                # to reduce latency
                self._process_message_fast(data, addr)
            except socket.error as e:
                if self.running:
                    print(f"Socket error: {e}")
                break
    
    def _process_message_fast(self, data: bytes, addr: tuple):
        """Fast message processing without threading overhead"""
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
            if len(data) >= 12:  # Minimum size for header (8 + 4 bytes)
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
            # Fast binary unpacking
            unique_id = struct.unpack('>Q', data[0:8])[0]
            fragment_id = struct.unpack('>I', data[8:12])[0]
            fragment_data = data[12:]
            
            # Reduce logging for performance
            if fragment_id % 10 == 0:  # Log every 10th fragment
                print(f"Fragment {fragment_id} for ID: {unique_id} ({len(fragment_data)} bytes)")
            
            if unique_id not in self.pending_requests:
                # If a request for this unique_id hasn't been received yet, or has timed out,
                # we can't properly track its fragments.
                # You might want to log this as a warning or just ignore.
                # For now, we'll ignore fragments without a pending request.
                return
            
            request = self.pending_requests[unique_id]
            
            # Add fragment to assembler
            self.image_assembler.add_fragment(
                unique_id, 
                fragment_id, 
                fragment_data, 
                request.TotalNumberOfFragments
            )
            
            # Check if image is complete and process immediately
            if self.image_assembler.is_complete(unique_id):
                self._process_complete_image(unique_id, addr)
                
        except Exception as e:
            print(f"Error processing fragment: {e}")
    
    def _process_complete_image(self, unique_id: int, addr: tuple):
        """Process complete image, save it, and send response"""
        request = self.pending_requests[unique_id]
        complete_image = self.image_assembler.get_complete_image(unique_id)
        
        if complete_image is None:
            print(f"Failed to get complete image for ID: {unique_id}")
            return
        
        print(f"Processing complete image for ID: {unique_id} ({len(complete_image)} bytes)")

        # --- NEW: Save the assembled image ---
        try:
            # Create a unique filename based on UniqueSequencedID and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"image_{unique_id}_{timestamp}.jpeg" # Assuming JPEG format
            file_path = os.path.join(self.output_folder, filename)
            
            with open(file_path, "wb") as f:
                f.write(complete_image)
            print(f"Assembled image saved to: {file_path}")
        except Exception as e:
            print(f"Error saving image for ID {unique_id}: {e}")
        # --- END NEW ---

        # Perform sign detection
        is_detected = self.sign_detector.detect_signs(
            complete_image, 
            request.Lat, 
            request.Lon, 
            request.Distance
        )
        
        # Send response
        self._send_response(unique_id, is_detected, addr)
        
        # Clean up
        if unique_id in self.pending_requests:
            del self.pending_requests[unique_id]
        if unique_id in self.request_timestamps:
            del self.request_timestamps[unique_id]
    
    def _response_timer(self, unique_id: int, addr: tuple):
        """Ensure response is sent within timeout"""
        time.sleep(self.max_response_time)
        
        # Check if response was already sent
        if unique_id in self.pending_requests:
            print(f"Response timeout for ID: {unique_id}, sending negative response")
            self._send_response(unique_id, False, addr)
            
            # Clean up
            if unique_id in self.pending_requests:
                del self.pending_requests[unique_id]
            if unique_id in self.request_timestamps:
                del self.request_timestamps[unique_id]
    
    def _send_response(self, unique_id: int, is_detected: bool, addr: tuple):
        """Send response back to vehicle"""
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
        
        self.socket.sendto(response_data, addr)
        print(f"Sent response for ID: {unique_id}, Detection: {is_detected}")

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