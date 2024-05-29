# Lab-PPT-CK
Lab PPT CK 

## Find root of a linear algebra systems
```matlab
% Define the coefficient matrix A and the right-hand side vector b
A = [3, 2; 1, 4];
b = [7; 3];

% Solve the system of equations Ax = b
x = A \ b; or x = x = linsolve(A,b)
```
## Get coefficients of the polynomial
```matlab
% For example, the polynomial is f = @(x) x^3 - 6x^2 + 11x - 6
coef = coeffs(f, All)
```
## Find root of a polynomial
```matlab
% Define the coefficients of the polynomial
% For example, for the polynomial x^3 - 6x^2 + 11x - 6, the coefficients are [1, -6, 11, -6]
coefficients = [1, -6, 11, -6];
% Find the roots of the polynomial
roots = roots(coefficients);

% If there's a complex one like: f = @(x)exp(m*x/10) - m*x^2 - x*sin(m*x/10)
% Use fsolve function with x0 is the initial value: 
roots = fsolve(f, x0) 
```
