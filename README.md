# The SailCore Module

## All Start

xmake f `
-m release `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=true `
--llvm_path="D:/env/LLVM" `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=true `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `


xmake f `
-m releasedbg `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=true `
--llvm_path="D:/env/llvm" `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=true `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `


CUDNN has no debug info 

xmake f `
-m debug `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=true `
--llvm_path="D:/env/llvm" `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=false `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `

## NO TENSOR and LLVM


xmake f `
-m release `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=false `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=false `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `

xmake f `
-m releasedbg `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=false `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=false `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `

## NO CUDA

xmake f `
-m debug `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=false `
--llvm_path="D:/env/llvm" `
--sail_enable_cuda=false `
--sail_enable_cuda_tensor=false `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `


## NO CUDA Tensor

xmake f `
-m debug `
--sail_core_standalone=true `
--sail_enable_gl=true `
--sail_enable_llvm=false `
--llvm_path="D:/env/llvm" `
--sail_enable_cuda=true `
--sail_enable_cuda_tensor=false `
--sail_enable_vk=true `
--sail_enable_dx=true `
--sail_enable_test=true `
--sail_enable_pybind=true `

# Structure

- external: the external dependencies
- internal: the standalone modules
  - cu: CUDA implementation
  - cut: CUDA Tensor implementation
  - dx: DirectX implementation
  - gl: OpenGL implementation
  - llvm: llvm-relevant utilities
  - vk: Vulkan implementation

