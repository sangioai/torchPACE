# torchPACE
PyTorch CUDA/C++ extension of PACE. A Transformer non-linearlity accelerator engine.

Inspired from here: </br>
    https://github.com/pytorch/extension-cpp

Builded with PyPA/Build: </br>
    https://github.com/pypa/build

To build: </br>
    ```python -m build -n```
    
To install:  </br>
    ```pip install dist\<builded_extension_file.whl>```

> [!NOTE]
> Requirements: 
>    - torch>=2.4 with CUDA enabled (here is 2.5.1+cu118)
>    - CUDA toolkit (here is 11.7)
>    - Python>=3.8 (here is 3.12.8)

## Results
