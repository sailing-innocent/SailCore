if get_config("sail_enable_cuda") then
    includes("cuda")
    if get_config("sail_enable_cuda_tensor") then
        includes("cut")
    end
end

includes("cpu")