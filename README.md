# torchPACE
PyTorch C++ and CUDA extension for PACE's Piecewise Polynomial Approximation(PwPA), a Transformer non-linerarities accelaration engine.

This library integrates the best performing CUDA kernels in [here](extra\test_optimization.cu).
w.r.t. a simple speedup metric shown in this [README](extra\README.md).

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

To test:  </br>
```text
python test\extension_test.py
```
</br>
```text
python test\approximation_test.py
```

> [!Important]
> Requirements: 
>    - torch>=2.4 with CUDA enabled (mine is 2.5.1+cu118)
>    - CUDA toolkit (mine is 11.7)
>    - Python>=3.8 (mine is 3.12.8)

> [!Note]
> To test you also need: 
>    - Pandas
>    - NumPy
>    - Matplotlib

## Results

This is the ouput of running [approximation_test.py](test\approximation_test.py):

## ToDo
A brief list of things to do or fix in this extension:
- [x] PyTorch Half type support
- [ ] Extension Benchmark on non-linearities in plain CUDA code
- [ ] Extension Benchmark on PyTorch non-linearities

## Credits

Extension backbone inspired by [this tutorial](https://github.com/pytorch/extension-cpp).
