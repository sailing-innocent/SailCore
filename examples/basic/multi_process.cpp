/**
 * @file multi_pro	// Set up security attributes to allow access from other processes
	SECURITY_ATTRIBUTES sa;
	SECURITY_DESCRIPTOR sd;
	
	// Initialize security descriptor
	if (!InitializeSecurityDescriptor(&sd, SECURITY_DESCRIPTOR_REVISION)) {
		std::cerr << "Failed to initialize security descriptor (" << GetLastError() << ")" << std::endl;
		return -1;
	}
	
	// Set DACL to NULL to allow all access
	if (!SetSecurityDescriptorDacl(&sd, TRUE, NULL, FALSE)) {
		std::cerr << "Failed to set security descriptor DACL (" << GetLastError() << ")" << std::endl;
		return -1;
	}
	
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = FALSE;
	sa.lpSecurityDescriptor = &sd;cpp
 * @brief The Multi-Process Example using proper IPC with shared memory
 * @author sailing-innocent
 * @date 2025-06-26
 */
#include <iostream>
#include <hv/TcpServer.h>
#include <windows.h>
#include <string>

// Dummy Struct in Shared Memory
struct Dummy {
	int id;
	char name[32];
	float value = 3.14f;
};

const char* SHARED_MEMORY_NAME = "SailCore_Dummy_SharedMemory";

int main() {
	int port = 1234;

	// Set up security attributes to allow access from other processes
	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.bInheritHandle = FALSE;
	sa.lpSecurityDescriptor = NULL;// Use default security descriptor

	// Create shared memory mapping with proper security attributes
	HANDLE hMapFile = CreateFileMappingA(
		INVALID_HANDLE_VALUE,// use paging file
		&sa,				 // security attributes for cross-process access
		PAGE_READWRITE,		 // read/write access
		0,					 // maximum object size (high-order DWORD)
		sizeof(Dummy),		 // maximum object size (low-order DWORD)
		SHARED_MEMORY_NAME	 // name of mapping object
	);

	if (hMapFile == NULL) {
		std::cerr << "Could not create file mapping object (" << GetLastError() << ")" << std::endl;
		return -1;
	}

	// Check if the mapping already exists
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		std::cout << "Shared memory mapping already exists, using existing one." << std::endl;
	} else {
		std::cout << "Created new shared memory mapping." << std::endl;
	}

	// Map a view of the file mapping into the address space
	Dummy* dummy = (Dummy*)MapViewOfFile(
		hMapFile,			// handle to map object
		FILE_MAP_ALL_ACCESS,// read/write permission
		0,
		0,
		sizeof(Dummy));

	if (dummy == NULL) {
		std::cerr << "Could not map view of file (" << GetLastError() << ")" << std::endl;
		CloseHandle(hMapFile);
		return -1;
	}

	// Initialize dummy data in shared memory
	dummy->id = 1;
	snprintf(dummy->name, sizeof(dummy->name), "DummyName");
	dummy->value = 3.14f;
	std::cout << "Dummy allocated in shared memory: " << dummy->id << ", " << dummy->name << ", " << dummy->value << std::endl;
	std::cout << "Shared memory name: " << SHARED_MEMORY_NAME << std::endl;

	hv::TcpServer server;
	int listenfd = server.createsocket(port);
	if (listenfd < 0) {
		std::cerr << "Failed to create socket on port " << port << std::endl;
		return -1;
	}

	std::cout << "Server is listening on port " << port << " with fd: " << listenfd << std::endl;

	// register hooks
	server.onConnection = [](const hv::SocketChannelPtr& channel) {
		std::string peeraddr = channel->peeraddr();
		if (channel->isConnected()) {
			std::cout << "New connection from " << peeraddr << std::endl;
		} else {
			std::cout << "Connection closed from " << peeraddr << std::endl;
		}
		std::cout << "Channel fd: " << channel->fd() << std::endl;
	};
	server.onMessage = [dummy](const hv::SocketChannelPtr& channel, hv::Buffer* buf) {
		if (buf->size() > 0) {
			std::string message(reinterpret_cast<const char*>(buf->data()), buf->size());
			std::cout << "Received message from channel fd " << channel->fd() << ": " << message << std::endl;

			if (message.find("DONE_MODIFYING:") == 0) {
				// Client has finished modifying shared memory
				std::cout << "=== CLIENT MODIFICATION COMPLETE ===" << std::endl;
				std::cout << "Client reported: " << message.substr(15) << std::endl;

				// Read and print the current state of shared memory
				std::cout << "Current shared memory state:" << std::endl;
				std::cout << "  ID: " << dummy->id << std::endl;
				std::cout << "  Name: " << dummy->name << std::endl;
				std::cout << "  Value: " << dummy->value << std::endl;
				std::cout << "======================================" << std::endl;

				// Send acknowledgment
				std::string ack_message = "ACK: Server read modified data successfully";
				channel->write(ack_message.c_str(), ack_message.length());

			} else if (message.find("READ_COMPLETE:") == 0) {
				// Client has read the shared memory data
				std::cout << "=== CLIENT READ COMPLETE ===" << std::endl;
				std::cout << "Client reported: " << message.substr(14) << std::endl;

				// Update shared memory for next read to show it's working
				dummy->value += 1.0f;
				snprintf(dummy->name, sizeof(dummy->name), "PostRead_%d", dummy->id);

				std::cout << "Updated shared memory after client read:" << std::endl;
				std::cout << "  ID: " << dummy->id << std::endl;
				std::cout << "  Name: " << dummy->name << std::endl;
				std::cout << "  Value: " << dummy->value << std::endl;
				std::cout << "==============================" << std::endl;

				// Send acknowledgment
				std::string ack_message = "ACK: Server updated data after client read";
				channel->write(ack_message.c_str(), ack_message.length());

			} else {
				// Regular request for shared memory info
				static int counter = 0;
				dummy->id = ++counter;
				dummy->value = 3.14f + counter * 0.1f;
				snprintf(dummy->name, sizeof(dummy->name), "ServerInit_%d", counter);

				std::cout << "Initialized shared memory with counter: " << counter << std::endl;
				std::cout << "Sharing memory via name: " << SHARED_MEMORY_NAME << std::endl;
				std::cout << "Dummy Memory Size: " << sizeof(Dummy) << std::endl;

				// Send a response back to the client with shared memory info
				char buffer[256];
				snprintf(buffer, sizeof(buffer),
						 "Shared Memory Name: %s\nShared Memory Size: %zu\nData: id=%d, name=%s, value=%.2f\n",
						 SHARED_MEMORY_NAME, sizeof(Dummy), dummy->id, dummy->name, dummy->value);
				channel->write(buffer);
			}
		}
	};
	server.setThreadNum(4);// Set the number of threads to handle connections
	server.start();

	// press Enter to stop
	while (getchar() != '\n') {
		// wait for user input
	}

	// Cleanup
	UnmapViewOfFile(dummy);
	CloseHandle(hMapFile);

	return 0;
}