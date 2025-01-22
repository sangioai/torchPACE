import numpy as np
import matplotlib.pyplot as plt
import torch
import torch_pace

###########################################################################################
################################### UTILITY FUNCTIONs ##################################### 
###########################################################################################
def get_coeffs(x, y, P, deg=3):
    """
    Computes the polynomial coefficients for different segments of data points.

    This function divides the data into `P` partitions and fits a polynomial of degree `deg`
    to each partition. The coefficients for each polynomial fit are returned as a list of 
    arrays, where each array corresponds to the coefficients of a polynomial for a specific
    segment of the data.

    Parameters:
    - x (array-like): The independent variable values (x-coordinates).
    - y (array-like): The dependent variable values (y-coordinates).
    - P (int): The number of partitions to divide the data into.
    - deg (int, optional): The degree of the polynomial fit for each partition (default is 3).

    Returns:
    - list: A list containing arrays of polynomial coefficients for each partition.
    """
    coeffs = []
    N = len(x)
    for i in range(1,P):
        prev_bound = (i-1)*N//P
        next_bound = i*N//P
        # https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html#numpy.polynomial.polynomial.Polynomial.fit
        _coeffs = np.polynomial.Polynomial.fit(x[prev_bound:next_bound], y[prev_bound:next_bound], deg).convert().coef
        _coeffs_pad = np.zeros((deg+1))
        _coeffs_pad[:_coeffs.shape[0]] = _coeffs
        coeffs += [_coeffs_pad]
    return coeffs

def get_partition_points(x, N, P):
    """
    Determines the partition points for dividing the data into `P` segments.

    The function calculates the indices in `x` that will divide the data into `P` partitions,
    returning a list of pairs of partition boundaries.

    Parameters:
    - x (array-like): The independent variable values (x-coordinates).
    - N (int): The number of data points.
    - P (int): The number of partitions to divide the data into.

    Returns:
    - list: A list of lists, each containing two partition points (x-values) for each partition.
    """
    partition_points = []
    for i in range(1,P-1):
        prev_point = i*N//P
        next_point = (i+1)*N//P
        partition_points += [[x[prev_point],x[next_point]]]
    return partition_points

def prepare_coeffs(poly_coeffs):
    """
    Prepares the polynomial coefficients for use in a PwPA extension.

    The function add coefficients of initial and leadining paritions and then
    it flips each partition's coefficients to be applied in Horner method.

    Parameters:
    - poly_coeffs (array-like): The list of polynomial coefficients.

    Returns:
    - torch.Tensor: A tensor of polynomial coefficients, reversed to be used in PwPA extension.
    """
    coeffs = torch.concat([
        torch.zeros_like(torch.tensor(poly_coeffs[0:1])),
        torch.tensor(poly_coeffs),
        torch.where(torch.arange(len(poly_coeffs[0])) == 1, 1., 0.).view(1,-1)
    ])
    coeffs = torch.flip(coeffs, dims=[1])
    return coeffs

def prepare_partition_points(bounds, neg_inf=-10000., pos_inf=1000.):
    """
    Prepares partition points by adding bounds for the outer segments.

    The function adds negative infinity and positive infinity bounds to the first and last
    partition points, respectively, ensuring the entire data range is covered.

    Parameters:
    - bounds (list): A list of partition points, where each entry is a list with two values.
    - neg_inf (float, optional): The value representing negative infinity for the first partition (default is -10000).
    - pos_inf (float, optional): The value representing positive infinity for the last partition (default is 1000).

    Returns:
    - torch.Tensor: A tensor containing the partition points, with unique values and outer bounds.
    """
    partition_points = [[neg_inf, bounds[0][0]]] + bounds + [[bounds[-1][-1], pos_inf]]
    return torch.tensor(partition_points).unique()


def get_poly_approx(func=torch.nn.Hardswish(), n_points=1000, x_min=-5, x_max=5, n_partitions=50, deg=3):
    """
    Generates a polynomial approximation for a given function.

    This function evaluates a given function over a range of points, determines partition
    points, computes polynomial coefficients for each partition, and returns the polynomial
    approximation for the function in terms of the original function's values, partition points,
    and polynomial coefficients.

    Parameters:
    - func (callable, optional): The function to approximate (default is `torch.nn.Hardswish()`).
    - n_points (int, optional): The number of points to sample for the approximation (default is 1000).
    - x_min (float, optional): The minimum x-value for the range (default is -5).
    - x_max (float, optional): The maximum x-value for the range (default is 5).
    - n_partitions (int, optional): The number of partitions to divide the data into (default is 50).
    - deg (int, optional): The degree of the polynomial for each partition (default is 3).

    Returns:
    - tuple: A tuple containing the following:
        - x (torch.Tensor): The sampled x-values for the approximation.
        - y (numpy.ndarray): The corresponding y-values from the function evaluation.
        - partition_points (torch.Tensor): The points where the data is partitioned.
        - coeffs (torch.Tensor): The polynomial coefficients for each partition.
    """
    # prepare function
    x = torch.linspace(x_min, x_max, n_points)
    y = func(torch.tensor(x)).numpy()
    # define paritions points
    partition_points = get_partition_points(x, n_points, n_partitions)
    partition_points = prepare_partition_points(partition_points).to(torch.float)
    # find polynomial coeffs
    coeffs = get_coeffs(x, y, n_partitions, deg=deg)
    coeffs = prepare_coeffs(coeffs).to(torch.float)
    return x, y, partition_points, coeffs


###########################################################################################
################################## APPROXIMATIONS TESTs ################################### 
###########################################################################################

N = 10000
x_min = -5
x_max = 5
n_partitions = 1024
degree = 1 

funcs = [
    torch.nn.Hardswish(),
    torch.nn.Softmax(),
    torch.nn.ReLU(),
    torch.nn.LeakyReLU(),
    torch.nn.GELU(),
    torch.nn.Sigmoid(),
]

i = 0
plt.figure(figsize=(10,10))
plt.suptitle(f"Non-linearities Approximations tests\nN={N} partitions={n_partitions} degree={degree}")
for f in funcs:
    i+=1
    plt.subplot((len(funcs)+1)//2, 2, i)
    x, y, partition_points, coeffs = get_poly_approx(func=f, n_points=N, x_min=x_min, x_max=x_max, n_partitions=n_partitions, deg=degree)
    y2 = torch_pace.ops.pwpa(x, coeffs, partition_points)
    plt.title(f"{f} Approx")
    plt.plot(x, y2, ".", color="green")
    plt.plot(x, y, "-", color="orange")
    plt.legend(["PwPA approx", "Target"])
plt.show()