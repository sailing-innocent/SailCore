#pragma once

/**
  * @file alloc.h
  * @brief Cutlass Allocator
  * @author sailing-innocent
  * @date 2025-01-20
  */
#include <cuda_runtime_api.h>
#include <sstream>
#include "cutlass/array.h"

namespace sail::cut {

namespace device_memory {
/// Allocate a buffer of \p count elements of type \p T on the current CUDA device
template<typename T>
T* allocate(size_t count = 1) {

	T* ptr = 0;
	size_t bytes = 0;

	bytes = count * sizeof(T);

	cudaError_t cuda_error = cudaMalloc((void**)&ptr, bytes);
	if (cuda_error != cudaSuccess) {
		throw std::exception("Failed to allocate memory", cuda_error);
	}

	return ptr;
}

/// Free the buffer pointed to by \p ptr
template<typename T>
void free(T* ptr) {
	if (ptr) {
		cudaError_t cuda_error = (cudaFree(ptr));
		if (cuda_error != cudaSuccess) {
			throw cuda_exception("Failed to free device memory", cuda_error);
		}
	}
}

/******************************************************************************
 * Data movement
 ******************************************************************************/

template<typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
	size_t bytes = count * ::cutlass::sizeof_bits<T>::value / 8;
	if (bytes == 0 && count > 0) {
		bytes = 1;
	}
	cudaError_t cuda_error = (cudaMemcpy(dst, src, bytes, kind));
	if (cuda_error != cudaSuccess) {
		std::ostringstream os;
		os << "cutlass::device_memory::copy: cudaMemcpy() failed: "
		   << "dst=" << dst << ", src=" << src
		   << ", bytes=" << bytes << ", count=" << count;
		if (kind == cudaMemcpyHostToDevice) {
			os << ", kind=cudaMemcpyHostToDevice";
		} else if (kind == cudaMemcpyDeviceToHost) {
			os << ", kind=cudaMemcpyDeviceToHost";
		} else if (kind == cudaMemcpyDeviceToDevice) {
			os << ", kind=cudaMemcpyDeviceToDevice";
		} else if (kind == cudaMemcpyHostToHost) {
			os << ", kind=cudaMemcpyHostToHost";
		} else if (kind == cudaMemcpyDefault) {
			os << ", kind=cudaMemcpyDefault";
		} else {
			os << ", kind=Unknown";
		}
		os << ", error: " << cudaGetErrorString(cuda_error);

		throw cuda_exception(os.str().c_str(), cuda_error);
	}
}

template<typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
	copy(dst, src, count, cudaMemcpyHostToDevice);
}

template<typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
	copy(dst, src, count, cudaMemcpyDeviceToHost);
}

template<typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
	copy(dst, src, count, cudaMemcpyDeviceToDevice);
}

template<typename T>
void copy_host_to_host(T* dst, T const* src, size_t count = 1) {
	copy(dst, src, count, cudaMemcpyHostToHost);
}

/// Copies elements from device memory to host-side range
template<typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
	size_t elements = end - begin;
	copy_to_host(&*begin, device_begin, elements);
}

/// Copies elements to device memory from host-side range
template<typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
	size_t elements = end - begin;
	copy_to_device(device_begin, &*begin, elements);
}

}// namespace device_memory

using ::cutlass::sizeof_bits;

template<typename T>
class DeviceAllocation {
public:
	/// Delete functor for CUDA device memory
	struct deleter {
		void operator()(T* ptr) {
			cudaError_t cuda_error = (cudaFree(ptr));
			if (cuda_error != cudaSuccess) {
				// noexcept
				//                throw cuda_exception("cudaFree() failed", cuda_error);
				return;
			}
		}
	};

	//
	// Data members
	//

	/// Number of elements of T allocated on the current CUDA device
	size_t capacity;

	/// Smart pointer
	::cutlass::platform::unique_ptr<T, deleter> smart_ptr;

	//
	// Static methods
	//

	/// Static member to compute the number of bytes needed for a given number of elements
	static size_t bytes(size_t elements) {
		if (::cutlass::sizeof_bits<T>::value < 8) {
			size_t const kElementsPerByte = 8 / ::cutlass::sizeof_bits<T>::value;
			return elements / kElementsPerByte;
		}
		size_t const kBytesPerElement = sizeof_bits<T>::value / 8;
		return elements * kBytesPerElement;
	}
	//
	// Methods
	//

	/// Constructor: allocates no memory
	DeviceAllocation() : capacity(0) {}

	/// Constructor: allocates \p capacity elements on the current CUDA device
	DeviceAllocation(size_t _capacity) : smart_ptr(device_memory::allocate<T>(_capacity)), capacity(_capacity) {}

	/// Constructor: allocates \p capacity elements on the current CUDA device taking ownership of the allocation
	DeviceAllocation(T* ptr, size_t _capacity) : smart_ptr(ptr), capacity(_capacity) {}

	/// Copy constructor
	DeviceAllocation(DeviceAllocation const& p) : smart_ptr(device_memory::allocate<T>(p.capacity)), capacity(p.capacity) {

		device_memory::copy_device_to_device(smart_ptr.get(), p.get(), capacity);
	}

	/// Move constructor
	DeviceAllocation(DeviceAllocation&& p) : capacity(0) {
		std::swap(smart_ptr, p.smart_ptr);
		std::swap(capacity, p.capacity);
	}

	/// Destructor
	~DeviceAllocation() { reset(); }

	/// Returns a pointer to the managed object
	T* get() const { return smart_ptr.get(); }

	/// Releases the ownership of the managed object (without deleting) and resets capacity to zero
	T* release() {
		capacity = 0;
		return smart_ptr.release();
	}

	/// Deletes the managed object and resets capacity to zero
	void reset() {
		capacity = 0;
		smart_ptr.reset();
	}

	/// Deletes managed object, if owned, and allocates a new object
	void reset(size_t _capacity) {
		reset(device_memory::allocate<T>(_capacity), _capacity);
	}

	/// Deletes managed object, if owned, and replaces its reference with a given pointer and capacity
	void reset(T* _ptr, size_t _capacity) {
		smart_ptr.reset(_ptr);
		capacity = _capacity;
	}

	/// Allocates a new buffer and copies the old buffer into it. The old buffer is then released.
	void reallocate(size_t new_capacity) {

		::cutlass::platform::unique_ptr<T, deleter> new_allocation(device_memory::allocate<T>(new_capacity));

		device_memory::copy_device_to_device(
			new_allocation.get(),
			smart_ptr.get(),
			std::min(new_capacity, capacity));

		std::swap(smart_ptr, new_allocation);
		std::swap(new_capacity, capacity);
	}

	/// Returns the number of elements
	size_t size() const {
		return capacity;
	}

	/// Returns the number of bytes needed to store the allocation
	size_t bytes() const {
		return bytes(capacity);
	}

	/// Returns a pointer to the object owned by *this
	T* operator->() const { return smart_ptr.get(); }

	/// Returns the deleter object which would be used for destruction of the managed object.
	deleter& get_deleter() { return smart_ptr.get_deleter(); }

	/// Returns the deleter object which would be used for destruction of the managed object (const)
	const deleter& get_deleter() const { return smart_ptr.get_deleter(); }

	/// Copies a device-side memory allocation
	DeviceAllocation& operator=(DeviceAllocation const& p) {
		if (capacity != p.capacity) {
			smart_ptr.reset(device_memory::allocate<T>(p.capacity));
			capacity = p.capacity;
		}
		device_memory::copy_device_to_device(smart_ptr.get(), p.get(), capacity);
		return *this;
	}

	/// Move assignment
	DeviceAllocation& operator=(DeviceAllocation&& p) {
		std::swap(smart_ptr, p.smart_ptr);
		std::swap(capacity, p.capacity);
		return *this;
	}

	/// Copies the entire allocation from another location in device memory.
	void copy_from_device(T const* ptr) const {
		copy_from_device(ptr, capacity);
	}

	/// Copies a given number of elements from device memory
	void copy_from_device(T const* ptr, size_t elements) const {
		device_memory::copy_device_to_device(get(), ptr, elements);
	}

	void copy_to_device(T* ptr) const {
		copy_to_device(ptr, capacity);
	}

	void copy_to_device(T* ptr, size_t elements) const {
		device_memory::copy_device_to_device(ptr, get(), elements);
	}

	void copy_from_host(T const* ptr) const {
		copy_from_host(ptr, capacity);
	}

	void copy_from_host(T const* ptr, size_t elements) const {
		device_memory::copy_to_device(get(), ptr, elements);
	}

	void copy_to_host(T* ptr) const {
		copy_to_host(ptr, capacity);
	}

	void copy_to_host(T* ptr, size_t elements) const {
		device_memory::copy_to_host(ptr, get(), elements);
	}
};

}// namespace sail::cut
