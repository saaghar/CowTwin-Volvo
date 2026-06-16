import socket
import json
import time
import struct
import os


# Serverns IP och port (din Docker-container körs på localhost:9876)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9876

# Sökväg till en testbild (ersätt med din egen)
# Se till att denna bild finns i samma mapp som test_client.py
#TEST_IMAGE_PATH = "C:/Users/A504686/Desktop/server/stop-sign-2.jpg" 
#TEST_IMAGE_PATH = "C:/Users/A504686/Desktop/server/frame_0311_jpg.rf.7140270db472e5ae14db7f7a8bdd821d.jpg"
#TEST_IMAGE_PATH = "C:/Users/A504686/Desktop/server/1277381680Image000004.jpg"
TEST_IMAGE_PATH = "C:/Users/A504686/Desktop/server/1277381674Image000011 copy.jpg"

def send_image_over_udp(image_path: str):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(5.0) # Timeout för att vänta på svar

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Generera ett unikt ID för denna bild
        unique_id = int(time.time() * 1000) # Unikt ID baserat på tid

        # Fragmentera bilden
        MAX_FRAGMENT_SIZE = 60000 # Max 60KB per fragment för att vara säkert under UDP MTU
        fragments = [image_data[i:i + MAX_FRAGMENT_SIZE] 
                     for i in range(0, len(image_data), MAX_FRAGMENT_SIZE)]
        total_fragments = len(fragments)

        print(f"Sending image '{image_path}' with {total_fragments} fragments (ID: {unique_id})")

        # 1. Skicka RequestForSignDetection (JSON)
        request_message = {
            "TotalNumberOfFragments": total_fragments,
            "Lat": 57.7089,
            "Lon": 11.9746,
            "Distance": 10.5,
            "UniqueSequencedID": unique_id
        }
        json_message = json.dumps(request_message).encode('utf-8')
        client_socket.sendto(json_message, (SERVER_HOST, SERVER_PORT))
        print(f"Sent request for ID {unique_id}")
        time.sleep(0.1) # Ge servern en kort stund att förbereda sig

        # 2. Skicka bildfragment (råa byte)
        for i, fragment in enumerate(fragments):
            # Pakethuvud: 8 bytes för UniqueSequencedID (unsigned long long), 4 bytes för fragmentSequenceID (unsigned int)
            header = struct.pack('>QI', unique_id, i) # '>Q' för 8-byte, '>I' för 4-byte
            packet = header + fragment
            client_socket.sendto(packet, (SERVER_HOST, SERVER_PORT))
            if i % 10 == 0:
                print(f"Sent fragment {i+1}/{total_fragments} for ID {unique_id}")
            time.sleep(0.001) # Liten paus för att inte överbelasta nätverket/servern för snabbt

        print(f"All fragments sent for ID {unique_id}.")

        # 3. Vänta på svar
        print(f"Waiting for response for ID {unique_id}...")
        try:
            response_data, server_address = client_socket.recvfrom(4096) # Buffertstorlek
            response_json = response_data.decode('utf-8')
            response_dict = json.loads(response_json)
            
            print(f"\nReceived response from {server_address}:")
            print(f"IsSignDetected: {response_dict.get('IsSignDetected')}")
            print(f"UniqueSequencedID: {response_dict.get('UniqueSequencedID')}")

        except socket.timeout:
            print(f"Timeout: No response received for ID {unique_id} within the specified time.")
        except json.JSONDecodeError:
            print("Error: Received malformed JSON response.")
        except Exception as e:
            print(f"An error occurred while receiving response: {e}")

    except FileNotFoundError:
        print(f"Error: Test image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    # SKAPA EN test_image.jpg I SAMMA MAPP SOM DETTA SKRIPT
    # Innan du kör detta, se till att du har en bildfil, t.ex. 'test_image.jpg',
    # i samma katalog som detta test_client.py-skript.
    # Bilden ska vara en JPG, PNG eller annan format som OpenCV kan avkoda.
    # Försök att använda en bild som antingen innehåller skyltar eller inte,
    # för att verifiera detektionslogiken.

    # Exempel: Lägg en bild som heter 'example_sign.jpg' i samma mapp
    # och ändra sedan raden nedan:
    # send_image_over_udp("example_sign.jpg")
    
    send_image_over_udp(TEST_IMAGE_PATH) # Använder den definierade TEST_IMAGE_PATH