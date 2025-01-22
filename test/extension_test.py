import torch
import torch_pace
import pandas as pd

# PwPA Parameters definitions
N = 2000000    # Number of input points
D = 3       # Polynomial Degree
P = 256      # Number of partitions
x_min = -5  # Minumum of inputs points
x_max = 5   # Maximum of input points
c_min = -10 # Minumum of Coeffient range
c_max = 10  # Maximum of Coeffient range

# PwPA Data definitons
X = torch.linspace(x_min, x_max, N)
partition_points = torch.linspace(x_min-1, x_max+1, P+1) # NOTE: first and last bound must be respectively lt and gt any number in X
coeffs = torch.randn((P,D+1))

# C++ versions
base_cpu = torch_pace.ops._pwpa(X.cpu(), coeffs.cpu(), partition_points.cpu())
base_cpu_half = torch_pace.ops._pwpa(X.half().cpu(), coeffs.half().cpu(), partition_points.half().cpu())
opt_cpu_aos = torch_pace.ops.pwpa(X.cpu(), coeffs.cpu(), partition_points.cpu())
opt_cpu_soa = torch_pace.ops.pwpa(X.cpu(), torch_pace.ops.aos2soa(coeffs.cpu(), D), partition_points.cpu(), AoS=False)
opt_cpu_aos_half = torch_pace.ops.pwpa(X.half().cpu(), coeffs.half().cpu(), partition_points.half().cpu())
opt_cpu_soa_half = torch_pace.ops.pwpa(X.half().cpu(), torch_pace.ops.aos2soa(coeffs.half().cpu(), D), partition_points.half().cpu(), AoS=False)

# CUDA versions
base_cuda = torch_pace.ops._pwpa(X.cuda(), coeffs.cuda(), partition_points.cuda())
base_cuda_half = torch_pace.ops._pwpa(X.half().cuda(), coeffs.half().cuda(), partition_points.half().cuda())
opt_cuda_aos = torch_pace.ops.pwpa(X.cuda(), coeffs.cuda(), partition_points.cuda())
opt_cuda_soa = torch_pace.ops.pwpa(X.cuda(), torch_pace.ops.aos2soa(coeffs.cuda(), D), partition_points.cuda(), AoS=False)
opt_cuda_aos_half = torch_pace.ops.pwpa(X.half().cuda(), coeffs.half().cuda(), partition_points.half().cuda())
opt_cuda_soa_half = torch_pace.ops.pwpa(X.half().cuda(), torch_pace.ops.aos2soa(coeffs.half().cuda(), D), partition_points.half().cuda(), AoS=False)

# Results
print(f"(base_cpu\t\t==\t\topt_cpu_AoS)\t\t: {(base_cpu == opt_cpu_aos).all()}") # True
print(f"(base_cpu\t\t==\t\topt_cpu_SoA)\t\t: {(base_cpu == opt_cpu_soa).all()}") # True
print(f"(base_cpu_half\t\t==\t\topt_cpu_AoS_half)\t: {(base_cpu_half == opt_cpu_aos_half).all()}") # True
print(f"(base_cpu_half\t\t==\t\topt_cpu_SoA_half)\t: {(base_cpu_half == opt_cpu_soa_half).all()}") # True
print(f"(base_cuda\t\t==\t\topt_cuda_AoS)\t\t: {(base_cuda == opt_cuda_aos).all()}") # True
print(f"(base_cuda\t\t==\t\topt_cuda_SoA)\t\t: {(base_cuda == opt_cuda_soa).all()}") # True
print(f"(base_cuda_half\t\t==\t\topt_cuda_AoS_half)\t: {(base_cuda_half == opt_cuda_aos_half).all()}") # True
print(f"(base_cuda_half\t\t==\t\topt_cuda_SoA_half)\t: {(base_cuda_half == opt_cuda_soa_half).all()}") # True
print(f"(base_cuda.cpu()\t==\t\tbase_cpu)\t\t: {(base_cuda.cpu() == base_cpu).all()}") # False
print(f"(base_cuda_half.cpu()\t==\t\tbase_cpu_half)\t\t: {(base_cuda_half.cpu() == base_cpu_half).all()}") # False
# CUDA and C++ distribution of differences distributions
print("\nDistribution of abs(base_cpu - base_cuda):")
print(pd.DataFrame(torch.abs(base_cpu-base_cuda.cpu())).describe()) 
print("\nDistribution of abs(base_cpu_half - base_cuda_half):")
print(pd.DataFrame(torch.abs(base_cpu_half.float()-base_cuda_half.float().cpu())).describe())