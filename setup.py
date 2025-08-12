from juliacall import Main as jl
print("Installing CUDA for juliacall...")
jl.seval("using Pkg") 
jl.seval('Pkg.add("CUDA")')
print("Done!")