# frft2-python
discrete 1d and 2d fractional fourier transfrom in python

## usage:

### frft2d(mat,ax,ay)
  
- mat: the numberic matrix to be transformed
  
- ax,ay: the order of transform along x and y axis
  
- returns the frft spectrum of mat
  
### disfrft(f,a;p)
  
- f: the discrete signal to be trasformed

- a: the order of transform

- p(optional): the order of approximation, equals to half of the length of f by default

- returns the frft spectrum of f
  
### this code is translated from its matlab version, and i actually dont know how the algorithm works xd. a senpai gave it to me and i dont know its origin.
