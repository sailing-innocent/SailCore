includes("basic")
includes("cpu")
includes("ogl") -- opengl examples

includes("llvm")

if has_config("sail_enable_cuda") then 
    includes("cuda")
    if has_config("sail_enable_cuda_tensor") then 
        includes("cut")
    end
end