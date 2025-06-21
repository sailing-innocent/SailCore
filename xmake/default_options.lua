-- llvm path
option("llvm_path")
    set_default("llvm")
    set_showmenu(true)
option_end()

-- gl_support
option("sail_enable_gl")
    set_default(false)
    set_showmenu(true)
option_end()

-- vk_support
option("sail_enable_vk")
    set_default(false)
    set_showmenu(true)
option_end()

-- cuda_support
option("sail_enable_cuda")
    set_default(false)
    set_showmenu(true)
option_end()

option("sail_enable_cuda_tensor")
    set_default(false)
    set_showmenu(true)
option_end()

-- dx_support
option("sail_enable_dx")
    set_default(false)
    set_showmenu(true)
option_end()

-- llvm
option("sail_enable_llvm")
    set_default(false)
    set_showmenu(true)
option_end()

option("sail_enable_test")
    set_default(false)
    set_showmenu(true)
option_end()

option("sail_core_standalone")
    set_default(false)
    set_showmenu(true)
option_end()