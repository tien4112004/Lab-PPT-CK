# Lab-PPT-CK
Lab PPT CK 

## Find root of a linear algebra systems
```matlab
% Define the coefficient matrix A and the right-hand side vector b
A = [3, 2; 1, 4];
b = [7; 3];

% Solve the system of equations Ax = b
x = A \ b;
```

## Find root of a polynomial
```matlab
% Define the coefficients of the polynomial
% For example, for the polynomial x^3 - 6x^2 + 11x - 6, the coefficients are [1, -6, 11, -6]
coefficients = [1, -6, 11, -6];

% Find the roots of the polynomial
roots = roots(coefficients);
```
