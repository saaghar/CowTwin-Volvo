# receiver.py
import socket

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 9027
BUFFER_SIZE = 1024 # A small buffer is fine for just catching any packet

def main():
    # Create a UDP socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind((HOST, PORT))
            print(f"[*] Listening for UDP packets on {HOST}:{PORT}")

            while True:
                # Receive packet from any client
                data, addr = s.recvfrom(BUFFER_SIZE)
                print(f"[*] Received {len(data)} bytes from {addr}: {data.decode(errors='ignore')}")
                # You can add a break here if you only want to receive one packet
                # break

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()