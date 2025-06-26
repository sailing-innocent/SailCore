#!/usr/bin/env python3
"""
Demonstration script showing read-only shared memory access and server communication
"""

import socket
import time
import ctypes
import ctypes.wintypes

# Windows API constants and functions for shared memory access
FILE_MAP_ALL_ACCESS = 0x001f001f
FILE_MAP_READ = 0x0004

# Load Windows API functions
kernel32 = ctypes.windll.kernel32

OpenFileMappingA = kernel32.OpenFileMappingA
OpenFileMappingA.argtypes = [ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.c_char_p]
OpenFileMappingA.restype = ctypes.wintypes.HANDLE

MapViewOfFile = kernel32.MapViewOfFile
MapViewOfFile.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD, ctypes.wintypes.DWORD, ctypes.c_size_t]
MapViewOfFile.restype = ctypes.wintypes.LPVOID

UnmapViewOfFile = kernel32.UnmapViewOfFile
UnmapViewOfFile.argtypes = [ctypes.wintypes.LPVOID]
UnmapViewOfFile.restype = ctypes.wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
CloseHandle.restype = ctypes.wintypes.BOOL

GetLastError = kernel32.GetLastError
GetLastError.restype = ctypes.wintypes.DWORD

class Dummy(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int),
                ("name", ctypes.c_char * 32),
                ("value", ctypes.c_float)]
    
    def __str__(self):
        return f"Dummy(id={self.id}, name={self.name.decode('utf-8', errors='ignore')}, value={self.value})"

def demonstrate_ipc():
    host = 'localhost'
    port = 1234
    
    print("=== IPC Demonstration ===")
    print("This demonstrates reading shared memory and server communication")
    print()
    
    for round_num in range(3):
        print(f"--- Round {round_num + 1} ---")
        
        try:
            # Connect to server
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            print(f"Connected to server at {host}:{port}")
            
            # Send request
            request_message = f"GET_DUMMY_INFO_ROUND_{round_num + 1}"
            client_socket.send(request_message.encode('utf-8'))
            print(f"Sent request: {request_message}")
            
            # Receive response
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Server response:\n{response}")
            
            # Parse shared memory info
            memory_name_prefix = "Shared Memory Name: "
            shared_memory_name = response.split(memory_name_prefix)[-1].split("\n")[0].strip()
            
            data_size_prefix = "Shared Memory Size: "
            dummy_size = response.split(data_size_prefix)[-1].split("\n")[0].strip()
            
            print(f"Shared Memory Name: {shared_memory_name}")
            print(f"Size: {dummy_size} bytes")
            
            # Access shared memory
            if shared_memory_name and dummy_size:
                size = int(dummy_size)
                
                # Open shared memory (read-only is fine for demonstration)
                hMapFile = OpenFileMappingA(
                    FILE_MAP_READ,
                    False,
                    shared_memory_name.encode('ascii')
                )
                
                if hMapFile:
                    print("✓ Successfully opened shared memory")
                    
                    # Map view
                    pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, size)
                    
                    if pBuf:
                        print("✓ Successfully mapped view")
                        
                        # Read data
                        dummy_ptr = ctypes.cast(pBuf, ctypes.POINTER(Dummy))
                        dummy_data = dummy_ptr.contents
                        print(f"✓ Read shared data: {dummy_data}")
                        
                        # Simulate telling server we've read the data
                        read_confirmation = f"READ_COMPLETE:Round_{round_num + 1}_Data_{dummy_data.id}"
                        client_socket.send(read_confirmation.encode('utf-8'))
                        
                        # Get acknowledgment
                        ack = client_socket.recv(1024).decode('utf-8')
                        print(f"Server ack: {ack}")
                        
                        UnmapViewOfFile(pBuf)
                    else:
                        print("✗ Failed to map view")
                    
                    CloseHandle(hMapFile)
                else:
                    print("✗ Failed to open shared memory")
            
            client_socket.close()
            print()
            
        except Exception as e:
            print(f"Error in round {round_num + 1}: {e}")
        
        if round_num < 2:
            time.sleep(1)  # Wait before next round
    
    print("=== Demonstration Complete ===")

if __name__ == "__main__":
    demonstrate_ipc()
