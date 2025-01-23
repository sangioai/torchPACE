# torchPACE
PyTorch C++ and CUDA extension for PACE's Piecewise Polynomial Approximation(PwPA), a Transformer non-linerarities accelaration engine.

## Introduction
This extension integrates PwPA CUDA kernels for both AoS and SoA coefficients' data structure using a simple unrolling technic.</br>
More details [here](extra/README.md).

## Setup
Built with [PyPA/Build](https://github.com/pypa/build), but you can use Pip or similar.

To build: </br>
```text
python -m build -n
```
    
To install:  </br>
```text
pip install dist\<builded_extension_file.whl>
```

To test:  </br>
```text
python test\extension_test.py
```

```text
python test\extension_test.py
```


To use:  </br>
```python
import torch_pace
...
# base kernel
y = torch_pace.ops._pwpa(x, coeffs, partition_points, AoS=true)
# optimized kernel
y = torch_pace.ops.pwpa(x, coeffs, partition_points, AoS=true)
# AoS to SoA coefficients rearrangement
coeffs_soa = torch_pace.ops.aos2soa(coeffs, degree)
# optimized kernel with SoA coefficients' data structure
y = torch_pace.ops.pwpa(x, coeffs_soa, partition_points, AoS=false)
```

> [!Important]
> Requirements: 
>    - torch>=2.4 with CUDA enabled (mine is 2.5.1+cu118)
>    - CUDA toolkit (mine is 11.7)
>    - Python>=3.8 (mine is 3.12.8)

## Examples

This is the ouput of running [approximation_test.py](test/approximation_test.py):
![immagine](https://github.com/user-attachments/assets/01ecdbec-d232-4e9e-99f5-f5d38cadfeb3)


## ToDo
A brief list of things to do or fix in this extension:
- [x] PyTorch Half type support
- [ ] Extension Benchmark on non-linearities in plain CUDA code
- [ ] Extension Benchmark on PyTorch non-linearities
- [x] aos2soa function
- [ ] soa2aos function
- [ ] Neural Net example

## Credits

Extension backbone inspired by [this tutorial](https://github.com/pytorch/extension-cpp).

## Authors

[Marco Sangiorgi](https://github.com/SangioAI)
