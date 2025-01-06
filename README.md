# torchPACE
PyTorch CUDA/C++ extension of PACE. A Transformer non-linearlity accelerator engine.

## Installation
Built with [PyPA/Build](https://github.com/pypa/build), but you can use Pip or similar.

To build: </br>
```text
python -m build -n
```
    
To install:  </br>
```text
pip install dist\<builded_extension_file.whl>
```

> [!Important]
> Requirements: 
>    - torch>=2.4 with CUDA enabled (here is 2.5.1+cu118)
>    - CUDA toolkit (here is 11.7)
>    - Python>=3.8 (here is 3.12.8)

## Example
```python
import torch
import torch_pace

### Parameters definitions
N = 10    # Number of input points
D = 3       # Polynomial Degree
P = 8      # Number of partitions
x_min = -5  # Minumum of inputs points
x_max = 5   # Maximum of input points
c_min = -10 # Minumum of Coeffient range
c_max = 10  # Maximum of Coeffient range

### Data definitons
X = torch.linspace(x_min, x_max, N)
partition_points = torch.linspace(x_min-1, x_max+1, P+1) # NOTE: first and last bound must be respectively lt and gt of any number in X
coeffs = torch.randn((P,D+1))

# C++ base-version
ext_cpu = torch_pace.ops._pwpa(X.cpu(), coeffs.cpu(), partition_points.cpu())
# C++ optimizaed-version
ext_soa_cpu = torch_pace.ops.pwpa(X.cpu(), coeffs.cpu(), partition_points.cpu())
# CUDA base-version
ext_cuda = torch_pace.ops._pwpa(X.cuda(), coeffs.cuda(), partition_points.cuda()).cpu() if torch.cuda.is_available() else None
# CUDA optimized-version w/out Shared Memory
ext_soa_cuda = torch_pace.ops.pwpa(X.cuda(), coeffs.cuda(), partition_points.cuda()).cpu() if torch.cuda.is_available() else None
# CUDA optimized-version w/ Shared Memory
ext_shmem_cuda = torch_pace.ops.pwpa_shmem(X.cuda(), coeffs.cuda(), partition_points.cuda()).cpu() if torch.cuda.is_available() else None

# Print results
print(f"CPU:\n{ext_cpu.numpy()}")
print(f"CPU optimized:\n{ext_soa_cpu.numpy()}")
print(f"CUDA:\n{ext_cuda.numpy()}")
print(f"CUDA optimized w/out SharedMemory:\n{ext_soa_cuda.numpy()}")
print(f"CUDA optimized w/ SharedMemory:\n{ext_shmem_cuda.numpy()}")
```


## Results

## Credits

Extension structure inspired from [this tutorial](https://github.com/pytorch/extension-cpp).
