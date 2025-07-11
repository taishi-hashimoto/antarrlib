# antarrlib

Basic operations on antenna array, including:

- Frequency, wave length, and wave number conversion
- Decibel and its inverse function
- Steering vector of the antenna array
- Spherical integral

## Notes

Generally, in antenna array theories, signal time series from multiple channels (i.e., antenna receivers or transmit frequencies) are arranged as a set of column vectors:
```
[ x_{11}, x_{12}, ..., x_{1T} ]
[ x_{21}, x_{22}, ..., x_{2T} ]
[                 ...         ]
[ x_{M1}, x_{M2}, ..., x_{MT} ],           (1)
```
where `M` is the number of channels and `T` is the number of time samples.

In this library, however, this kind of **list-of-vectors**-structure is defined as a set of row vectors as follows:
```
X = [ x_{11}, x_{12}, ..., x_{1M} ]
    [ x_{21}, x_{22}, ..., x_{2M} ]
    [                 ...         ]
    [ x_{T1}, x_{T2}, ..., x_{TM} ],           (2)
```
which is the transpose of those in general theories in `(1)`.  
This applies to **signal time series**, **weight vectors**, and so on.

For example, if you have signals `X` in the form of `(2)`, signal correlation matrix will be computed as:

```Python
# In thextbook, this is usually written as `R = (1 / T) * X X^H`.
R = (1 / T) * X.T @ X.conj()
```

and if you have an weight vector `W = [w_1, w_2, ..., w_M]` in row vector form, the synthesized output power of the array `y` is:

```Python
# Again, in textbook, this is usually `y = W^H X
y = W.conj().dot(X.T)
```
