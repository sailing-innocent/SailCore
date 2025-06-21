SHARED_MODULE("SailCPU", "SAIL_CPU", engine_version)
    -- CPU HPC and Graphics lib based on TBB and Embree
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_deps("SailBase")
    add_packages("tbb", "embree", { public = true })  

