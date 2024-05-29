% 22120368 - Phan Thanh Tien
% Clear workspaces
clear all;
clc;

%% Beginning of Chapter 1

%% 13
A = 2 ^ 3 - (1 + 2) * (2 + 3) / (3 + 4) + sqrt(2) / sqrt(sqrt(3));
B = sin(pi / 3) - 2 * cos(pi / 4) + 3 * tan(pi / 6) / (2 - cot(5 * pi / 6));
C = exp(-sqrt(2)) - log(2/3) + log(exp(1) + 2);
D = (2 * A + 3 * B) / (C ^ 2 - 2 * C)

%% 14
a = 2; b = 3; c = 1;
A = (b + sqrt(b ^ 2 - 4 * a * c)) / (2 * a);
B = [a * sin(b) * cos(c), a * sin(b) * sin(c), a * cos(b)];
C = [(a + b) / (a + b + c), (a - b + c) / (a + b + c), (c ^ 2 - a * b) / (a + b + c), 1 / (a + b + c)];
% D = A * B + C % Error

%% 15
f = @(x) x .* sin(x);

fplot(f, [-2, 4]);
title('Plot of f(x) = x*sin(x)');
xlabel('x');
ylabel('f(x)');
grid on;

syms x;
f_sym = x * sin(x);
fplot(f_sym, [-2, 4]);
title('Plot of f(x) = x*sin(x)');
xlabel('x');
ylabel('f(x)');
grid on;

%% 16
% Bai 16.b
syms x;
syms y;
fxy = abs(x) + 2 * abs(y);
f1_2 = subs(subs(fxy, x, 1), y, 2);
ezsurf(f, [-2, 4, -3, 3]);

% Bai 16.a
f = @(x, y) abs(x) + 2 * abs(y);
f1 = f(1, 2);
[x, y] = meshgrid(-2:0.1:4, -3:0.1:3);
fxy = f(x, y);
surf(x, y, fxy);

%% 17 Find the derivative and integral of a function
% 17.a
syms x;
f = x ^ 2 + 2 * x + 4;

df = diff(f, x);
df2 = subs(df, x, 2);

f_i = int(f, x, 0, 1);
f_a = int(f);

% 17.b
g = (x .^ 2 + 1) / (x + 1);
dg = diff(g, x);
dg2 = diff(dg, x);
dg2_1 = sub(dg, x, 1);

G = int(g);
G_1 = int(g, -1, 1);

% % 17.c
% h = @x sin(2 * x);
% dh = diff(h, x);
% dh_0 = dh(0);
% H = int(h);
% H_1 = int(h, 0, inf);

%% 18 Solve quadratic equation
a = 2;
b = -3;
c = 1;

[x1, x2] = solve_quadratic(a, b, c);

fprintf('x1 = %f\n', x1);
fprintf('x2 = %f\n', x2);

function [x1, x2] = solve_quadratic(a, b, c)
    delta = b ^ 2 - 4 * a * c;

    if delta < 0
        x1 = NaN;
        x2 = NaN;
    else
        x1 = (-b + sqrt(delta)) / (2 * a);
        x2 = (-b - sqrt(delta)) / (2 * a);
    end

end

%% 19 Find the extreme values of a function
clear all;
clc;
syms x;
f = x ^ 3 - 6 * x;
[x, y] = find_extreme(f, -2, 2);

fprintf('Cuc tri cua f(x) = x^3 - 6x la:\n');
fprintf('x = %f\n', x);
fprintf('y = %f\n', y);

function [x, y] = find_extreme(f, a, b)
    syms x;
    df = diff(f, x);
    x = solve(df, x);
    y = subs(f, x);
end

%% 20 Find the 2nd derivative of 2 variables function, check if f_xy = f_yx
f = (x / y) * sin(y / x);

find_l2_derivative(f);

function find_l2_derivative(f)
    syms x y;
    f_xx = diff(f, x, 2);
    f_yy = diff(f, y, 2);
    f_xy = diff(diff(f, x), y);
    f_yx = diff(diff(f, y), x);

    if f_xy == f_yx
        fprintf('f_xy = f_yx\n');
    else
        fprintf('f_xy != f_yx\n');
    end

end

%% End of Chapter 1

%% Beginning of Chapter 2: Error and Approximation

clear all;
format long;
clc;

%% 1, 2.
% a.
p_e = 0.9857; % p*
p_a = 0.9768; % \bar{p}
% b.
p_e = 421;
p_a = 397;
% c.
p_e = 1102;
p_a = 1113;
% d.
p_e = 2.5743;
p_a = 2.6381;

% Code for a, b, c, d
aEp = abs(p_e - p_a);
rEp = aEp / abs(p_e);

%% 3. Find the error between true value with approximation using round() and floor()
% a.
p_e = pi;
% b.
p_e = exp(1);
% c.
p_e = ln(2);
% d.
p_e = sqrt(2);
% e.
p_e = sin(1);

% Code for a, b, c, d, e
p_1 = round(p_e, 3);
aEp1 = abs(p_e - p_1);
rEp1 = aEp1 / abs(p_e);

p_2 = floor(p_e * 10 ^ 3) / (10 ^ 3);
aEp2 = abs(p_e - p_2);
rEp2 = aEp2 / abs(p_e);

fprintf('[round]\n');
fprintf('p_1 = %f\n', p_1);
fprintf('Absolute error: %f\n', aEp1);
fprintf('Relative error: %f\n', rEp1);

fprintf('[floor]\n');
fprintf('p_2 = %f\n', p_2);
fprintf('Absolute error: %f\n', aEp2);
fprintf('Relative error: %f\n', rEp2);

%% 4. Check if the p_L <= p* <= p_R
% a.
p_t = 15.932; %\bar{p}
aEp = 1.247; % Absolute error
p_e = 17.351; % p*
% b.
p_t = 11115;
aEp = 120;
p_e = 11205;
% c.
p_t = 36.215;
aEp = 1.327;
p_e = 38.735;
% d.
p_t = 297;
aEp = 15;
p_e = 319;

% Code for a, b, c, d
p_L = p_t - aEp;
p_R = p_t + aEp;

if (p_L <= p_e) && (p_e <= p_R)
    fprintf('p_L <= p* <= p_R\n');
else
    fprintf('p_L > p* or p* > p_R\n');
end

%% 8. Given the following function
function [aEp, rEp] = saiso(p_e, p_a)
    aEp = abs(p_e - p_a);
    rEp = aEp / abs(p_e);
end

% a. What is the input and output?
% Input: p_e, p_a
% Output: aEp, rEp: The absolute and relative error

% b. Where the function has been put? Name of the function?
% saiso.m

% c. How to use this function?
% [aEp, rEp] = saiso(p_e, p_a)

%% 9. Function to find approximation of a Non-repeating decimal value
% Test the function
value = pi;
[round_value, floor_value] = approx_non_repeating_decimal(value);
fprintf('round_value = %f\n', round_value);
fprintf('floor_value = %f\n', floor_value);

function [round_value, floor_value] = approx_non_repeating_decimal(value)
    round_value = round(value, 3);
    floor_value = floor(value * 10 ^ 3) / (10 ^ 3);
end

%% 10. Function to check if the error is within the delta value (true/false)
% p_a is the approximation value
% p_e is the true value
% Test the function
p_e = pi;
p_a = 3.14;
delta_value = 0.01;
is_valid_approximation = is_valid_approximation(p_e, p_a, delta_value);
fprintf('is_valid_approximation = %d\n', is_valid_approximation);

function [is_valid_approximation] = is_valid_approximation(p_e, p_a, delta_value)
    p_L = p_e - aEp;
    p_R = p_e + aEp;
    is_valid_approximation = (p_L <= p_a) && (p_a <= p_R);
end

%% 11. Function to check if the error is within the tolerance
% Test the function
p_e = pi;
p_a = 3.14;
tolerance = 0.01;
valid = is_within_tolerance(p_e, p_a, tolerance);
fprintf('is_within_tolerance = %d\n', valid);

function [valid] = is_within_tolerance(p_e, p_a, tolerance)
    aEp = abs(p_e - p_a);
    rEp = aEp / abs(p_e);
    valid = rEp <= tolerance;
end

%% 12. Function to find the absolute and relative error of a function that has n variables
% Test the function
f = @(x) x ^ 2 + 2 * x + 4;
p_e = 10;
p_a = 11;
[aEp, rEp] = find_error(f, p_e, p_a);
fprintf('aEp = %f\n', aEp);
fprintf('rEp = %f\n', rEp);

function [aEp, rEp] = find_error(f, p_e, p_a)
    aEp = abs(p_e - p_a);
    rEp = aEp / abs(p_e);
end

%% 13. Function to find the value of function u (get 3 decimal places) and the absolute and relative error
% a.
syms x y;
u = log(2 * y + x ^ 2);
x_0 = 1.796;
y_0 = 0.532;  % Use a different variable name for the numeric value
p_e = double(subs(subs(u, x, x_0), y, y_0));  % Substitute y with y_0
p_a = round(p_e * 1000) / 1000;
[aEp, rEp] = find_error_v2(u, p_e, p_a);
fprintf('p_a = %f\n', p_a);
fprintf('aEp = %f\n', aEp);
fprintf('rEp = %f\n', rEp);

function [aEp, rEp] = find_error_v2(u, p_e, p_a)
    aEp = abs(p_e - p_a);
    rEp = aEp / abs(p_e);
end

%% End of chapter 2

%% Beginning of Chapter 3: Solve the equation
%% 1. Bisection method
% Function to solve the equation using Bisection method
% Test the function
f = @(x) x ^ 3 - 6 * x ^ 2 + 11 * x - 6.1;
a = 0;
b = 1;
tol = 1e-5;
[x, n] = bisection_method(f, a, b, tol);
fprintf('x = %f\n', x);
fprintf('n = %d\n', n);

% Input:
%   f: A function handle for the function you want to find a root of.
%   a, b: The interval that contains the root.
%   tol: The tolerance for the stopping criterion.
% Output:
%   x: The approximate root of the function f.
%   n: The number of iterations needed to reach the desired tolerance.
function [x, n] = bisection_method(f, a, b, tol)
    n = 0;
    % Print the table header
    fprintf('%5s %10s %10s %10s %10s %10s %10s\n', 'k', 'a', 'b', 'c', 'f(c)', 'f(c) < tol', 'f(c) * f(a)');

    figure; hold on; % Create a new figure and hold it for multiple plots
    fplot(f, [a b]); % Plot the function f over the interval [a, b]
    title('Bisection Method');
    xlabel('x');
    ylabel('f(x)');

    while (b - a) / 2 > tol
        n = n + 1;
        c = (a + b) / 2;
        fc = f(c);
        plot(c, f(c), 'ro'); % Plot the midpoint at each iteration
        % Print the iteration results
        fprintf('%5d %10.5f %10.5f %10.5f %10.5f %10d %10.5f\n', n, a, b, c, fc, abs(fc) < tol, fc * f(a));

        if fc == 0
            break;
        elseif f(a) * fc < 0
            b = c;
        else
            a = c;
        end
    end

    x = (a + b) / 2;

end

%% Given function:
chiadoi(@(x) exp(x) - x - 3, 0, 3, 10 ^ (-3));

function [c, fc] = chiadoi(f, a, b, Df)
    k = 1; hold on

    while 1
        c = (a + b) / 2;
        fc = f(c);
        rEc = abs((a - c) / a);
        disp([k c fc]);
        plot(k, fc, 'ro');
        if abs(fc) < Df, break, end

        if f(a) * f(c) > 0, a = c;
        else b = c;
        end

        k = k + 1;
    end
end

%% 2. Fixed point iteration method
% Function to solve the equation using Fixed point iteration method
% Test the function
g = @(x) 1 + 1 / x;
x0 = 1;
tol = 1e-5;
[x, n, x_vals] = fixed_point_iteration_method(g, x0, tol);
fprintf('x = %f\n', x);
fprintf('n = %d\n', n);

% The Fixed Point Iteration method for root finding.
% Input:
%   g: A function handle for the function you want to find a fixed point of.
%   x0: The initial guess for the fixed point.
%   tol: The tolerance for the stopping criterion.
% Output:
%   x: The approximate fixed point of the function g.
%   n: The number of iterations needed to reach the desired tolerance.
%   x_vals: The estimates at each iteration.
function [x, n, x_vals] = fixed_point_iteration_method(g, x0, tol)
    n = 0;
    x = x0;
    x_vals = x; % Initialize x_vals with the initial guess

    % Print the table header
    fprintf('%5s %10s %10s %10s\n', 'k', 'xk', 'f(xk)', 'f(xk) < tol');

    while true
        n = n + 1;
        x1 = g(x);

        % Print the iteration results
        fprintf('%5d %10.5f %10.5f %10d\n', n, x, g(x), abs(x1 - x) < tol);

        if abs(x1 - x) < tol
            break;
        end

        x = x1;
        x_vals = [x_vals, x]; % Store the estimate at each iteration
    end

    % Plot the estimates at each iteration
    figure;
    plot(1:n, x_vals, 'o-');
    xlabel('Iteration');
    ylabel('Estimate');
    title('Convergence of Fixed Point Iteration');
end


%% 3. Newton-Raphson method
% Function to solve the equation using Newton-Raphson method
% Test the function
f = @(x) x ^ 3 - 6 * x ^ 2 + 11 * x - 6.1;
df = @(x) 3 * x ^ 2 - 12 * x + 11;
x0 = 0;
tol = 1e-5;
[x, n] = newton_raphson_method(f, df, x0, tol);
fprintf('x = %f\n', x);
fprintf('n = %d\n', n);

% The Newton-Raphson method for root finding.
% Input:
%   f: A function handle for the function you want to find a root of.
%   df: A function handle for the derivative of f.
%   x0: The initial guess for the root.
%   tol: The tolerance for the stopping criterion.
% Output:
%   x: The approximate root of the function f.
%   n: The number of iterations needed to reach the desired tolerance.
function [x, n, x_vals] = newton_raphson_method(f, df, x0, tol)
    n = 0;
    x = x0;
    x_vals = x; % Initialize x_vals with the initial guess

    % Print the table header
    fprintf('%5s %10s %10s %10s\n', 'k', 'xk', 'f(xk)', 'f(xk) < tol');

    while true
        n = n + 1;
        x1 = x - f(x) / df(x);

        % Print the iteration results
        fprintf('%5d %10.5f %10.5f %10d\n', n, x, f(x), abs(x1 - x) < tol);

        if abs(x1 - x) < tol
            break;
        end

        x = x1;
        x_vals = [x_vals, x]; % Store the estimate at each iteration
    end

    % Plot the estimates at each iteration
    figure;
    plot(1:n, x_vals, 'o-');
    xlabel('Iteration');
    ylabel('Estimate');
    title('Convergence of Newton-Raphson Method');
end


%% 4. Secant method
% Test the function
f = @(x) x ^ 3 - 6 * x ^ 2 + 11 * x - 6.1;
x0 = 0;
x1 = 1;
tol = 1e-5;
[x, n] = secant_method(f, x0, x1, tol);
fprintf('x = %f\n', x);
fprintf('n = %d\n', n);

% The Secant method for root finding.
% Input:
%   f: A function handle for the function you want to find a root of.
%   x0, x1: The initial guesses for the root.
%   tol: The tolerance for the stopping criterion.
% Output:
%   x: The approximate root of the function f.
%   n: The number of iterations needed to reach the desired tolerance.
function [x, n, x_vals] = secant_method(f, x0, x1, tol)
    n = 0;
    x_vals = x1; % Initialize x_vals with the initial guess

    % Print the table header
    fprintf('%5s %10s %10s %10s %10s %10s %10s\n', 'k', 'a', 'b', 'x', 'f(x)', 'f(x) < tol', 'f(x) * f(a)');

    while true
        n = n + 1;
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0));

        % Print the iteration results
        fprintf('%5d %10.5f %10.5f %10.5f %10.5f %10d %10.5f\n', n, x0, x1, x2, f(x2), abs(x2 - x1) < tol, f(x2) * f(x0));

        if abs(x2 - x1) < tol
            break;
        end

        x0 = x1;
        x1 = x2;
        x_vals = [x_vals, x1]; % Store the estimate at each iteration
    end

    x = x2;

    % Plot the estimates at each iteration
    figure;
    plot(1:n, x_vals, 'o-');
    xlabel('Iteration');
    ylabel('Estimate');
    title('Convergence of Secant Method');
end


%% End of chapter 3


%% Chapter 4. Linear Algebra Functions


% Test the function
% Define the coefficient matrix A and the right-hand side vector b
A = [5 1 1; 1 10 1; 1 1 20];
b = [7; 12; 22];

% Define the initial guess x0, the tolerance tol, and the maximum number of iterations max_iter
x0 = [0; 0; 0];
tol = 1e-6;
max_iter = 100;

% Call the iterative function
x_iterative = iterative(A, b, x0, tol, max_iter);

% Call the gauss_seidel function
x_gauss_seidel = gauss_seidel(A, b, x0, tol, max_iter);

% Print the results
fprintf('Solution from iterative method: \n');
disp(x_iterative);
fprintf('Solution from Gauss-Seidel method: \n');
disp(x_gauss_seidel);


% The Iterative method for solving linear systems.
% Input:
%   A: The coefficient matrix.
%   b: The right-hand side vector.
%   x0: The initial guess for the solution.
%   tol: The tolerance for the stopping criterion.
%   max_iter: The maximum number of iterations.
% Output:
%   x: The approximate solution of the system Ax = b.
function x = iterative(A, b, x0, tol, max_iter)
    D = diag(diag(A)); % Diagonal matrix
    L = tril(A, -1); % Lower triangular matrix
    U = triu(A, 1); % Upper triangular matrix
    T = -inv(D) * (L + U); % Iterative matrix
    c = inv(D) * b; % Iterative vector
    x = x0; % Initial guess
    x_vals = x; % Initialize x_vals with the initial guess

    % Print the table header
    fprintf('%5s %15s\n', 'Iter', 'Root');

    for i = 1:max_iter
        x_new = T * x + c;

        % Print the root after each iteration
        for j = 1:length(x_new)
            fprintf('%5d %15.6f\n', i, x_new(j));
        end

        if norm(x_new - x, inf) < tol
            x = x_new;
            return;
        end

        x = x_new;
        x_vals = [x_vals, x]; % Store the root after each iteration
    end

    % Plot the root after each iteration
    figure;
    plot(1:max_iter, x_vals, 'o-');
    xlabel('Iteration');
    ylabel('Root');
    title('Convergence of Iterative Method');

    error('Solution did not converge');
end


% The Gauss-Seidel method for solving linear systems.
% Input:
%   A: The coefficient matrix.
%   b: The right-hand side vector.
%   x0: The initial guess for the solution.
%   tol: The tolerance for the stopping criterion.
%   max_iter: The maximum number of iterations.
% Output:
%   x: The approximate solution of the system Ax = b.
function x = gauss_seidel(A, b, x0, tol, max_iter)
    D = diag(diag(A)); % Diagonal matrix
    L = tril(A, -1); % Lower triangular matrix
    U = triu(A, 1); % Upper triangular matrix
    T = -inv(D + L) * U; % Iterative matrix
    c = inv(D + L) * b; % Iterative vector
    x = x0; % Initial guess
    x_vals = x; % Initialize x_vals with the initial guess

    % Print the table header
    fprintf('%5s %15s\n', 'Iter', 'Root');

    for i = 1:max_iter
        x_new = T * x + c;

        % Print the root after each iteration
        for j = 1:length(x_new)
            fprintf('%5d %15.6f\n', i, x_new(j));
        end

        if norm(x_new - x, inf) < tol
            x = x_new;
            return;
        end

        x = x_new;
        x_vals = [x_vals, x]; % Store the root after each iteration
    end

    % Plot the root after each iteration
    figure;
    plot(1:max_iter, x_vals, 'o-');
    xlabel('Iteration');
    ylabel('Root');
    title('Convergence of Gauss-Seidel Method');

    error('Solution did not converge');
end


% Iterative method (other way) - [DEPRECATED]
function x = iterative2(A, C, tol, max_iter)
    % Compute B and G
    D = diag(A); % Diagonal elements of A
    B = -A ./ D + eye(size(A)); % B matrix
    G = C ./ D; % G vector

    % Initialize x
    x = G;

    % Iteration
    for k = 1:max_iter
        x_new = B * x + G;

        if norm(A * x_new - C) < tol
            x = x_new;
            return;
        end

        x = x_new;
    end

    error('Solution did not converge');
end

%% End of Chapter 4

%% Chap 5.1. Interpolation
%% 5.1.1. Polynomial interpolation
% Test the function
x = [1, 2.2, 3.1, 4];
y = [1.678, 3.267, 2.198, 3.787];
x_new = 2.5;
[y_new, a] = polynomial_interpolation(x, y, x_new);

% Print the interpolated y value
fprintf('y_new = %f\n', y_new);

% Print the regression function coefficients
fprintf('The regression function coefficients are: \n');
disp(a);

% Polynomial Interpolation Function
% This function performs polynomial interpolation for a given set of x and y values.
% It finds the coefficients of the polynomial that fits the given points, evaluates this polynomial at a new point or a set of new points, and plots the original points and the interpolated points.
% Additionally, it finds the regression function, prints it, and returns it.
%
% Inputs:
%   x: A vector of x values of the points.
%   y: A vector of y values of the points.
%   x_new: A new x value or a vector of new x values at which to evaluate the polynomial.
%
% Output:
%   y_new: The value or values of the polynomial at the new point or points.
%   polynomial_str: The string of polynomial 
%   coef: The coefficients of the polynomial (the regression function).
% Usage:
%   [y_new, f] = polynomial_interpolation(x, y, x_new);
%
% Example:
%   x = [1, 2.2, 3.1, 4];
%   y = [1.678, 3.267, 2.198, 3.787];
%   x_new = 2.5;
%   [y_new, polynomial_str, coef] = polynomial_interpolation(x, y, x_new);
%   fprintf('y_new = %f\n', y_new);
%   fprintf('The regression function coefficients are: \n');
%   disp(polynomial_str);
%
%   % Draw the function of polynomial interpolation
%   x_new = 1:0.1:4;
%   [y_new, polynomial_str, coef] = polynomial_interpolation(x, y, x_new);
%   plot(x, y, 'o', x_new, y_new, '-');
%
%   % Print the regression function coefficients
%   fprintf('The regression function coefficients are: \n');
%   disp(coef);
function [y_new, polynomial_str, coef] = polynomial_interpolation(x, y, x_new)
    n = length(x);
    A = zeros(n, n);

    for i = 1:n
        A(:, i) = x .^ (i - 1);
    end

    coef = A \ y';
    y_new = zeros(size(x_new));

    for i = 1:n
        y_new = y_new + coef(i) * x_new .^ (i - 1);
    end
    
    % Build the polynomial string
    n = length(x);
    polynomial_str = 'Polynomial interpolation: f(x) = ';
    for i = n:-1:2
        polynomial_str = [polynomial_str, num2str(coef(i)), ' * x^', num2str(i - 1), ' + '];
    end
    polynomial_str = [polynomial_str, num2str(coef(1))];
    
    % Return the coefficients of the polynomial
    return;
end
% Print the polynomial
disp( polynomial_str);

% Draw the function of polynomial interpolation
x = [1, 2.2, 3.1, 4];
y = [1.678, 3.267, 2.198, 3.787];
x_new = 1:0.1:4;
y_new = polynomial_interpolation(x, y, x_new);
plot(x, y, 'o', x_new, y_new, '-');
title('Polynomial Interpolation');

%% 5.1.2. Lagrange Interpolation Function
% This function performs Lagrange interpolation for a given set of x and y values.
% It constructs the Lagrange polynomial that fits the given points, and evaluates this polynomial at a new point or a set of new points.
%
% Inputs:
%   x: A vector of x values of the points. Should be the same length as y.
%   y: A vector of y values of the points. Should be the same length as x.
%   x_new: A new x value or a vector of new x values at which to evaluate the polynomial.
%
% Outputs:
%   y_new: The value or values of the polynomial at the new point or points.
%   polynimial_str: The string form of polynomial
%   coef: The coefficients of the polynomial (the regression function).
% Usage:
%   [y_new, polynimial_str, coef] = lagrange_interpolation(x, y, x_new);
%
% Example:
%   x = [1, 2.2, 3.1, 4];
%   y = [1.678, 3.267, 2.198, 3.787];
%   x_new = 2.5;
%   [y_new, polynimial_str, coef] = lagrange_interpolation(x, y, x_new);
%   fprintf('y_new = %f\n', y_new);
%   disp(coef)
%   disp(polynomial_str)
% Note: If x_new is a vector, the function will plot the original points (x, y) and the interpolated points (x_new, y_new).

function y_new = lagrange_interpolation(x, y, x_new)
    % Initialize y_new
    y_new = zeros(size(x_new));

    % Number of data points
    n = length(x);

    % Initialize symbolic variable
    syms x_sym;

    % Initialize Lagrange polynomial
    L = 0;
    
    % Construct the Lagrange polynomial
    for i = 1:n
        % Initialize Li,n(x) as 1
        Li = 1;
        for j = 1:n
            if i ~= j
                Li = Li * (x_sym - x(j)) / (x(i) - x(j));
            end
        end
        % Add the term Li,n(x) * y(i) to the Lagrange polynomial
        L = L + Li * y(i);
    end

    % Print the Lagrange polynomial
    fprintf('Lagrange interpolation: L(x) = %s\n', char(L));

    % Evaluate the polynomial at the new points
    for k = 1:length(x_new)
        y_new(k) = double(subs(L, x_sym, x_new(k)));
    end

    % Return the interpolated values
    return;
end

% Draw the function of Lagrange interpolation
x_new = 1:0.1:4;
y_new = lagrange_interpolation(x, y, x_new);
plot(x, y, 'o', x_new, y_new, '-');
title('Lagrange Interpolation');
xlabel('x');
ylabel('y');
legend('Original points', 'Interpolated points');
grid on;

%% 5.1.3. Newton Interpolation Function
% This function performs Newton interpolation for a given set of x and y values.
% It finds the coefficients of the Newton polynomial that fits the given points, evaluates this polynomial at a new point or a set of new points.
% Additionally, it prints the Newton polynomial in a symbolic format and returns the interpolated y values and the polynomial string.
%
% Inputs:
%   x: A vector of x values of the points. Should be the same length as y.
%   y: A vector of y values of the points. Should be the same length as x.
%   x_new: A new x value or a vector of new x values at which to evaluate the polynomial.
%
% Outputs:
%   y_new: The value or values of the polynomial at the new point or points.
%   polynomial_str: The string form of polynomial
%   coef: The coefficients of the polynomial (the regression function).
% Usage:
%   [y_new, polynomial_str, coef] = newton_interpolation(x, y, x_new);
%
% Example:
% x = [1, 2.2, 3.1, 4];
% y = [1.678, 3.267, 2.198, 3.787];
% x_new = 2.5;
% [y_new, polynomial_str, coef] = newton_interpolation(x, y, x_new);
% fprintf('y_new = %f\n', y_new);
% disp(coef);
% disp(polynomial_str);
%
% Note: If x_new is a vector, the function will plot the original points (x, y) and the interpolated points (x_new, y_new).

function [y_new, polynimial_str] = newton_interpolation(x, y, x_new)
    n = length(x);
    f = zeros(n, n);
    f(:, 1) = y(:); % the first column is y

    for j = 2:n
        for i = j:n
            f(i, j) = (f(i, j - 1) - f(i - 1, j - 1)) / (x(i) - x(i - j + 1));
        end
    end

    % Compute the interpolated values
    y_new = f(n, n);

    for i = (n - 1):-1:1
        y_new = y_new .* (x_new - x(i)) + f(i, i);
    end

    % Print the Newton polynomial
    syms x_sym; % Create a symbolic variable
    polynimial_str = ['Newton interpolation: f(x) = ', char(poly2sym(f(n, :), x_sym))];

    % Return the interpolated values
    return;
end

% Draw the function of Newton interpolation
x_new = 1:0.1:4;
y_new = newton_interpolation(x, y, x_new);
plot(x, y, 'o', x_new, y_new, '-');
title('Newton Interpolation');
xlabel('x');
ylabel('y');
legend('Original points', 'Interpolated points');
grid on;

%% 5.1.4. Cubic Spline Interpolation Function
% This function performs cubic spline interpolation for a given set of x and y values.
% It constructs the cubic spline that fits the given points, evaluates this spline at a new point or a set of new points.
% Additionally, it prints the cubic spline in a symbolic format and returns the interpolated y values and the spline function.
%
% Inputs:
%   x: A vector of x values of the points. Should be the same length as y.
%   y: A vector of y values of the points. Should be the same length as x.
%   x_new: A new x value or a vector of new x values at which to evaluate the spline.
%
% Outputs:
%   y_new: The value or values of the spline at the new point or points.
%   pp: The piecewise polynomial representing the spline.
%   spline_str: String form of polynomial
% Usage:
%   [y_new, pp, spline_str] = spline_interpolation(x, y, x_new);
%
% Example:
%   x = [1, 2.2, 3.1, 4];
%   y = [1.678, 3.267, 2.198, 3.787];
%   x_new = 2.5;
%   [y_new, pp, spline_str] = spline_interpolation(x, y, x_new);
%   fprintf('y_new = %f\n', y_new);
%   disp(spline_str);
%
% Note: If x_new is a vector, the function will plot the original points (x, y) and the interpolated points (x_new, y_new).

function [y_new, pp, spline_str] = spline_interpolation(x, y, x_new)
    % Construct the cubic spline
    pp = spline(x, y);

    % Evaluate the spline at the new points
    y_new = ppval(pp, x_new);

    % Initialize the spline string
    spline_str = ['Spilne interpolation: ' newline];

    % Iterate over the pieces
    for i = 1:pp.pieces
        % Get the coefficients of the i-th piece
        coefs = pp.coefs(i, :);

        % Construct a string for the i-th piece
        piece_str = sprintf('(%g <= x < %g) : f%i(x) = f%g*(x - %g)^3 + %g*(x - %g)^2 + %g*(x - %g) + %g\n', ...
            x(i), x(i+1), i, coefs(1), x(i), coefs(2), x(i), coefs(3), x(i), coefs(4));

        % Add the string of the i-th piece to the spline string
        spline_str = [spline_str, piece_str];
    end

    % If x_new is a vector, plot the original points and the interpolated points
    if length(x_new) > 1
        plot(x, y, 'o', x_new, y_new, '-');
        title('Cubic Spline Interpolation');
        xlabel('x');
        ylabel('y');
        legend('Original points', 'Interpolated points');
        grid on;
    end

    % Return the interpolated values, the spline, and the spline string
    return;
end
% Define the data points
x = [1, 2.2, 3.1, 4];
y = [1.678, 3.267, 2.198, 3.787];

% Define the new x value
x_new = 2.5;

% Call the spline_interpolation function
[y_new, pp, spline_str] = spline_interpolation(x, y, x_new);

% Print the interpolated y value
fprintf('The interpolated y value at x = %f is y = %f\n', x_new, y_new);
% Print the polynimial 
disp(spline_str)

% Plot the original points and the interpolated point
figure;
plot(x, y, 'o', x_new, y_new, 'x');
title('Cubic Spline Interpolation');
xlabel('x');
ylabel('y');
legend('Original points', 'Interpolated point');
grid on;

%% Chap 5.2. Regression
%% 5.2.1. Least Squares R(x) = ax + b
function [a, b] = least_squares(x, y)
    % LEAST_SQUARES Fit a line to the data using least squares.
    %
    % Syntax:
    %   [a, b] = LEAST_SQUARES(x, y)
    %
    % Description:
    %   LEAST_SQUARES(x, y) fits a line to the data in vectors x and y
    %   using the method of least squares. The line is of the form R(x) = ax + b.
    %   The function returns the slope a and y-intercept b of the line, and also
    %   plots the original data and the fitted line.
    %
    % Inputs:
    %   x - a vector of x coordinates
    %   y - a vector of y coordinates
    %
    % Outputs:
    %   a - the slope of the fitted line
    %   b - the y-intercept of the fitted line
    %
    % Example:
    %   x = [1, 2.2, 3.1, 4];
    %   y = [1.678, 3.267, 2.198, 3.787];
    %   [a, b] = least_squares(x, y);

    % Calculate the means of x and y
    x_mean = mean(x);
    y_mean = mean(y);

    % Calculate the slope (a) and y-intercept (b) using the formula for least squares fitting
    a = sum((x - x_mean) .* (y - y_mean)) / sum((x - x_mean) .^ 2);
    b = y_mean - a * x_mean;

    % Plot the original data and the fitted line
    plot(x, y, 'o');
    hold on;
    plot(x, a * x + b, '-');
    hold off;
    title('Least Squares Line Fit');
    xlabel('x');
    ylabel('y');
    legend('Original data', 'Fitted line');
    grid on;

    % Print the equation of the line
    fprintf('R(x) = %f*x + %f\n', a, b);

    % Return the slope and y-intercept
    return;
end

x = [1, 2.2, 3.1, 4];
y = [1.678, 3.267, 2.198, 3.787];
[a, b] = least_squares(x, y);

%% 5.2.2. Least squares exp
function [a, b] = least_squares_exp_linear(x, y)
    % Take the logarithm of y
    logy = log(y);

    % Calculate the means of x and logy
    x_mean = mean(x);
    logy_mean = mean(logy);

    % Calculate the slope (b) and intercept (A) using the formula for least squares fitting
    b = sum((x - x_mean) .* (logy - logy_mean)) / sum((x - x_mean) .^ 2);
    A = logy_mean - b * x_mean;

    % Calculate a from A
    a = exp(A);

    % Plot the original data and the fitted function
    plot(x, y, 'o');
    hold on;
    plot(x, a * exp(b * x), '-');
    hold off;
    title('Least Squares Exponential Fit');
    xlabel('x');
    ylabel('y');
    legend('Original data', 'Fitted function');
    grid on;

    % Print the equation of the function
    fprintf('R(x) = %f * e^(%f*x)\n', a, b);

    % Return the parameters a and b
    return;
end

x = [1, 2.2, 3.1, 4];
y = [2.718, 7.389, 20.085, 54.598];
[a, b] = least_squares_exp_linear(x, y);

%% 5.2.3.a. Least squares quadratic
function [a, b] = least_squares_quad(x, y)
    % LEAST_SQUARES_QUAD Fit a quadratic function to the data using least squares.
    %
    % Syntax:
    %   [a, b] = LEAST_SQUARES_QUAD(x, y)
    %
    % Description:
    %   LEAST_SQUARES_QUAD(x, y) fits a quadratic function to the data in vectors x and y
    %   using the method of least squares. The function is of the form R(x) = ax^2 + b.
    %   The function returns the parameters a and b of the function, and also
    %   plots the original data and the fitted function.
    %
    % Inputs:
    %   x - a vector of x coordinates
    %   y - a vector of y coordinates
    %
    % Outputs:
    %   a - the parameter a of the fitted function
    %   b - the parameter b of the fitted function
    %
    % Example:
    %   x = [1, 2.2, 3.1, 4];
    %   y = [2, 5.84, 10.61, 17];
    %   [a, b] = least_squares_quad(x, y);

    % Fit a quadratic function to the data using least squares
    p = polyfit(x, y, 2);

    % Extract the parameters a and b
    a = p(1);
    b = p(3);

    % Plot the original data and the fitted function
    plot(x, y, 'o');
    hold on;
    plot(x, a * x .^ 2 + b, '-');
    hold off;
    title('Least Squares Quadratic Fit');
    xlabel('x');
    ylabel('y');
    legend('Original data', 'Fitted function');
    grid on;

    % Print the equation of the function
    fprintf('R(x) = %f*x^2 + %f\n', a, b);

    % Return the parameters a and b
    return;
end

x = [1, 2.2, 3.1, 4];
y = 0.5 * x .^ 2 + 1.5;
[a, b] = least_squares_quad(x, y);

%% 5.2.3.b. Least squares power
function [a, b] = least_squares_power(x, y)
    % LEAST_SQUARES_POWER Fit a power function to the data using least squares.
    %
    % Syntax:
    %   [a, b] = LEAST_SQUARES_POWER(x, y)
    %
    % Description:
    %   LEAST_SQUARES_POWER(x, y) fits a power function to the data in vectors x and y
    %   using the method of least squares. The function is of the form R(x) = a * x^b.
    %   The function returns the parameters a and b of the function, and also
    %   plots the original data and the fitted function.
    %
    % Inputs:
    %   x - a vector of x coordinates
    %   y - a vector of y coordinates
    %
    % Outputs:
    %   a - the parameter a of the fitted function
    %   b - the parameter b of the fitted function
    %
    % Example:
    %   x = [1, 2.2, 3.1, 4];
    %   y = [2, 5.84, 10.61, 17];
    %   [a, b] = least_squares_power(x, y);

    % Take the logarithm of x and y
    logx = log(x);
    logy = log(y);

    % Fit a linear function to the transformed data using least squares
    p = polyfit(logx, logy, 1);

    % The slope of the fitted line is the parameter b, and the exponent of the
    % intercept is the parameter a
    b = p(1);
    a = exp(p(2));

    % Plot the original data and the fitted function
    plot(x, y, 'o');
    hold on;
    plot(x, a * x .^ b, '-');
    hold off;
    title('Least Squares Power Fit');
    xlabel('x');
    ylabel('y');
    legend('Original data', 'Fitted function');
    grid on;

    % Print the equation of the function
    fprintf('R(x) = %f * x^%f\n', a, b);

    % Return the parameters a and b
    return;
end

x = [1, 2.2, 3.1, 4];
y = 2 * x .^ 1.3;
[a, b] = least_squares_power(x, y);

%% Chap 5 end

%% Chap 6. Numerical Differentiation
%% 6.1.1. Forward differences
x = [1, 2, 3, 4, 5];
y = [1, 4, 9, 16, 25]; % y = x^2
dy = calculate_derivative_forward(x, y);
disp(dy); % Display the result

function dy = calculate_derivative_forward(x, y)
    % Calculate the derivative of a function given tables of x and y values.
    %
    % Inputs:
    %   x - a vector of x values
    %   y - a vector of y values corresponding to f(x)
    %
    % Outputs:
    %   dy - a vector of the derivative of f at each x

    % Initialize the output vector
    dy = zeros(size(y));

    % Calculate the derivative at the interior points
    dy(2:end - 1) = (y(3:end) - y(1:end - 2)) ./ (x(3:end) - x(1:end - 2));

    % Use forward difference for the first point
    dy(1) = (y(2) - y(1)) / (x(2) - x(1));

    % Use backward difference for the last point
    dy(end) = (y(end) - y(end - 1)) / (x(end) - x(end - 1));
end

%% 6.2. Backward differences
x = [1, 2, 3, 4, 5];
y = [1, 4, 9, 16, 25]; % y = x^2
dy = calculate_derivative_backward(x, y);
disp(dy); % Display the result

function dy = calculate_derivative_backward(x, y)
    % Calculate the derivative of a function given tables of x and y values using backward differences.
    %
    % Inputs:
    %   x - a vector of x values
    %   y - a vector of y values corresponding to f(x)
    %
    % Outputs:
    %   dy - a vector of the derivative of f at each x

    % Initialize the output vector
    dy = zeros(size(y));

    % Use backward difference for all points, except the first one
    dy(2:end) = (y(2:end) - y(1:end - 1)) ./ (x(2:end) - x(1:end - 1));

    % Use forward difference for the first point
    dy(1) = (y(2) - y(1)) / (x(2) - x(1));
end

%% 6.2. Numerical Integration

%% 6.2.1. Trapezoid method
function I = trapezoid_method_data(x, y)
    % Check that the data is evenly spaced
    h = diff(x);

    if any(abs(h - h(1)) > 1e-6)
        error('Data must be evenly spaced to use the trapezoidal rule');
    end

    % Initialize the integral
    I = 0;

    % Add the contributions from the endpoints
    I = I + y(1) / 2 + y(end) / 2;

    % Add the contributions from the interior points
    for i = 2:length(y) - 1
        I = I + y(i);
    end

    % Multiply by h
    I = I * h(1);

    % Print the result
    fprintf('The integral of the function is %f.\n', I);

    % Return the integral
    return;
end

% Call the functions with data points
x = linspace(0, 1, 101);
y = x .^ 2;
I_trapezoid = trapezoid_method_data(x, y);

%% 6.2.2.1. Simpson 1/3 (known function)
function I = simpsons_one_third(f, a, b, n)
    % SIMPSONS_ONE_THIRD Calculate the integral of a function using the Simpson's 1/3 rule.
    %
    % Syntax:
    %   I = SIMPSONS_ONE_THIRD(f, a, b, n)
    %
    % Description:
    %   SIMPSONS_ONE_THIRD(f, a, b, n) calculates the integral of the function f from a to b using the Simpson's 1/3 rule with n intervals.
    %
    % Inputs:
    %   f - a function handle
    %   a - the lower limit of integration
    %   b - the upper limit of integration
    %   n - the number of intervals
    %
    % Outputs:
    %   I - the integral of f from a to b

    % Calculate the step size
    h = (b - a) / n;

    % Initialize the integral
    I = 0;

    % Add the contributions from the endpoints
    I = I + f(a) + f(b);

    % Add the contributions from the interior points
    for i = 1:n - 1

        if mod(i, 2) == 0
            I = I + 2 * f(a + i * h);
        else
            I = I + 4 * f(a + i * h);
        end

    end

    % Multiply by h/3
    I = I * h / 3;

    % Print the result
    fprintf('The integral of the function from %f to %f is %f.\n', a, b, I);

    % Return the integral
    return;
end

f = @(x) x .^ 2;
a = 0;
b = 1;
n = 100;
I = simpsons_one_third(f, a, b, n);

%% 6.2.2.2. Simpson 1/3 (unknown function)
function I = simpsons_one_third_data(x, y)
    % SIMPSONS_ONE_THIRD_DATA Calculate the integral of a function using the Simpson's 1/3 rule.
    %
    % Syntax:
    %   I = SIMPSONS_ONE_THIRD_DATA(x, y)
    %
    % Description:
    %   SIMPSONS_ONE_THIRD_DATA(x, y) calculates the integral of the function represented by the data points (x, y) using the Simpson's 1/3 rule.
    %
    % Inputs:
    %   x - the x values of the data points
    %   y - the y values of the data points
    %
    % Outputs:
    %   I - the integral of the function represented by the data points

    % Check that the data is evenly spaced
    h = diff(x);

    if any(abs(h - h(1)) > 1e-6)
        error('Data must be evenly spaced to use Simpsons 1/3 rule');
    end

    % Initialize the integral
    I = 0;

    % Add the contributions from the endpoints
    I = I + y(1) + y(end);

    % Add the contributions from the interior points
    for i = 2:length(y) - 1

        if mod(i, 2) == 0
            I = I + 4 * y(i);
        else
            I = I + 2 * y(i);
        end

    end

    % Multiply by h/3
    I = I * h(1) / 3;

    % Print the result
    fprintf('The integral of the function is %f.\n', I);

    % Return the integral
    return;
end

% Call the function with data points
x = linspace(0, 1, 101);
y = x .^ 2;
I = simpsons_one_third_data(x, y);

%% 6.2.3.1. Simpson 3/8 (known function)
function I = simpsons_three_eighths(f, a, b, n)
    % SIMPSONS_THREE_EIGHTHS Calculate the integral of a function using the Simpson's 3/8 rule.
    %
    % Syntax:
    %   I = SIMPSONS_THREE_EIGHTHS(f, a, b, n)
    %
    % Description:
    %   SIMPSONS_THREE_EIGHTHS(f, a, b, n) calculates the integral of the function f from a to b using the Simpson's 3/8 rule with n intervals.
    %
    % Inputs:
    %   f - a function handle
    %   a - the lower limit of integration
    %   b - the upper limit of integration
    %   n - the number of intervals
    %
    % Outputs:
    %   I - the integral of f from a to b

    % Calculate the step size
    h = (b - a) / n;

    % Initialize the integral
    I = 0;

    % Add the contributions from the endpoints
    I = I + f(a) + f(b);

    % Add the contributions from the interior points
    for i = 1:n - 1

        if mod(i, 3) == 0
            I = I + 2 * f(a + i * h);
        else
            I = I + 3 * f(a + i * h);
        end

    end

    % Multiply by 3h/8
    I = I * 3 * h / 8;

    % Print the result
    fprintf('The integral of the function from %f to %f is %f.\n', a, b, I);

    % Return the integral
    return;
end

f = @(x) x .^ 2;
a = 0;
b = 1;
n = 100;
I = simpsons_three_eighths(f, a, b, n);

%% 6.2.3.2. Simpson 3/8 (unknown function)
function I = simpsons_three_eighths_data(x, y)
    % SIMPSONS_THERE_EIGHTHS_DATA Calculate the integral of a function using the Simpson's 3/8 rule.
    %
    % Syntax:
    %   I = SIMPSONS_THERE_EIGHTHS_DATA(x, y)
    %
    % Description:
    %   SIMPSONS_THERE_EIGHTHS_DATA(x, y) calculates the integral of the function represented by the data points (x, y) using the Simpson's 3/8 rule.
    %
    % Inputs:
    %   x - the x values of the data points
    %   y - the y values of the data points
    %
    % Outputs:
    %   I - the integral of the function represented by the data points

    % Check that the data is evenly spaced
    h = diff(x);

    if any(abs(h - h(1)) > 1e-6)
        error('Data must be evenly spaced to use Simpson''s 3/8 rule');
    end

    % Initialize the integral
    I = 0;

    % Add the contributions from the endpoints
    I = I + y(1) + y(end);

    % Add the contributions from the interior points
    for i = 2:length(y) - 1

        if mod(i, 3) == 0
            I = I + 2 * y(i);
        else
            I = I + 3 * y(i);
        end

    end

    % Multiply by 3h/8
    I = I * 3 * h(1) / 8;

    % Print the result
    fprintf('The integral of the function is %f.\n', I);

    % Return the integral
    return;
end

x = linspace(0, 1, 101);
y = x .^ 2;
I_simpsons38 = simpsons_three_eighths_data(x, y);

%% 6.2.4. Newton-Cotes formulas
% Define a function handle for the function to integrate
f = @(x) x .^ 2;

% Define the x and y values of the data points for Newton-Cotes method
x = linspace(a, b, 101);
y = f(x);

% Call the Newton-Cotes function
[I_newton, H] = newton_cotes(x, y, 'trapezoid');

% Print the result
fprintf('The integral of the function using Newton-Cotes method is %f.\n', I_newton);

% % Print the H matrix
%fprintf('The H matrix is:\n');
%disp(H);

function [I, H] = newton_cotes(x, y, rule)
    h = diff(x);

    if any(

    h = h(1);
    n = length(y);
    H = h * ones(n, n);

    for i = 1:n

        for j = i + 1:n
            H(i, j) = H(i, j - 1) + h;
            H(j, i) = H(i, j);
        end

    end

    switch rule
        case 'trapezoid'
            I = h / 2 * (y(1) + 2 * sum(y(2:end - 1)) + y(end));
        case 'simpson1/3'

            if mod(n, 2) == 0
                error('Simpson''s 1/3 rule requires an odd number of intervals');
            end

            I = h / 3 * (y(1) + 4 * sum(y(2:2:end - 1)) + 2 * sum(y(3:2:end - 2)) + y(end));
        case 'simpson3/8'

            if mod(n, 3) ~= 1
                error('Simpson''s 3/8 rule requires a number of intervals that is one more than a multiple of three');
            end

            I = 3 * h / 8 * (y(1) + 3 * sum(y(2:3:end - 2)) + 3 * sum(y(3:3:end - 1)) + 2 * sum(y(4:3:end - 3)) + y(end));
        otherwise
            error('Invalid rule. Valid rules are ''trapezoid'', ''simpson1/3'', and ''simpson3/8''.');
    end

end

%% 6.2.5. Gauss 2 - 3 - 4 points
function I = gauss_legendre_2(f, a, b)
    x = [-1 / sqrt(3), 1 / sqrt(3)];
    w = [1, 1];
    I = 0.5 * (b - a) * sum(w .* arrayfun(f, 0.5 * ((b - a) * x + a + b)));
end

function I = gauss_legendre_3(f, a, b)
    x = [-sqrt(3/5), 0, sqrt(3/5)];
    w = [5/9, 8/9, 5/9];
    I = 0.5 * (b - a) * sum(w .* arrayfun(f, 0.5 * ((b - a) * x + a + b)));
end

function I = gauss_legendre_4(f, a, b)
    x = [-sqrt((3/7) - (2/7) * sqrt(6/5)), sqrt((3/7) - (2/7) * sqrt(6/5)), ...
             -sqrt((3/7) + (2/7) * sqrt(6/5)), sqrt((3/7) + (2/7) * sqrt(6/5))];
    w = [(18 + sqrt(30)) / 36, (18 + sqrt(30)) / 36, (18 - sqrt(30)) / 36, (18 - sqrt(30)) / 36];
    I = 0.5 * (b - a) * sum(w .* arrayfun(f, 0.5 * ((b - a) * x + a + b)));
end

% Define a function handle for the function to integrate
f = @(x) x .^ 2;

% Define the lower and upper limits of integration
a = 0;
b = 1;

% Call the Gauss-Legendre quadrature functions
I_gauss2 = gauss_legendre_2(f, a, b);
I_gauss3 = gauss_legendre_3(f, a, b);
I_gauss4 = gauss_legendre_4(f, a, b);

% Print the results
fprintf('The integral of the function using Gauss-Legendre 2-point quadrature is %f.\n', I_gauss2);
fprintf('The integral of the function using Gauss-Legendre 3-point quadrature is %f.\n', I_gauss3);
fprintf('The integral of the function using Gauss-Legendre 4-point quadrature is %f.\n', I_gauss4);

%% End of chapter 6



%% Chapter 7. Ordinary Differential Equations
% 7.1. Iteration method
function [x, y] = solve_ode_analytical(f, x0, x1, h)
    % Solve an ODE using analytical integration.
    %
    % Inputs:
    %   f - function handle for the ODE (dy/dx = f(x))
    %   x0 - initial x value
    %   x1 - final x value
    %   h - step size
    %
    % Outputs:
    %   x - vector of x values
    %   y - solution at each x value

    % Initialize the x values
    x = x0:h:x1;

    % Initialize the symbolic variable
    syms s

    % Define the symbolic function
    g = matlabFunction(int(f(s), s));

    % Calculate the solution
    y = g(x);
end

% Define the ODE (dy/dt = y)
f = @(t) t;

% Solve the ODE from t = 0 to t = 2 with step size 0.01
[t, y] = solve_ode_analytical(f, 0, 2, 0.01);

% Print out the solution
disp('The solution y at each time t is:');
disp(y);

% Plot the solution
plot(t, y);

% Add a title and labels to the axes
title('Solution of dy/dt = y using Analytical method');
xlabel('Time (t)');
ylabel('Solution (y)');

% 7.2. Euler's method
function [t, y] = solve_ode_euler(f, tspan, y0, h)
    % Solve an ODE using the Euler method.
    %
    % Inputs:
    %   f - function handle for the ODE (dy/dt = f(t, y))
    %   tspan - 2-element vector specifying the time range [t0 tf]
    %   y0 - initial condition
    %   h - step size
    %
    % Outputs:
    %   t - vector of time points
    %   y - solution at each time point

    % Initialize the time points
    t = tspan(1):h:tspan(2);

    % Initialize the solution vector
    y = zeros(size(t));

    % Set the initial condition
    y(1) = y0;

    % Iterate over each time point
    for i = 1:(length(t) - 1)
        % Update the solution using the Euler method
        y(i + 1) = y(i) + h * f(t(i), y(i));
    end
end

% Define the ODE (dy/dt = y)
f = @(t, y) y;

% Solve the ODE from t = 0 to t = 2 with initial condition y(0) = 1 and step size 0.01
[t, y] = solve_ode_euler(f, [0 2], 1, 0.01);

% Print the solution
% disp(y);

% Plot the solution
plot(t, y);

% Add a title and labels to the axes
title('Solution of dy/dt = y using Euler method');
xlabel('Time (t)');
ylabel('Solution (y)');


% 7.3. Euler's improved method
function [t, y] = solve_ode_euler_improved(f, tspan, y0, h)
    % Solve an ODE using the Euler's improved method.
    %
    % Inputs:
    %   f - function handle for the ODE (dy/dt = f(t, y))
    %   tspan - 2-element vector specifying the time range [t0 tf]
    %   y0 - initial condition
    %   h - step size
    %
    % Outputs:
    %   t - vector of time points
    %   y - solution at each time point

    % Initialize the time points
    t = tspan(1):h:tspan(2);

    % Initialize the solution vector
    y = zeros(size(t));

    % Set the initial condition
    y(1) = y0;

    % Iterate over each time point
    for i = 1:(length(t) - 1)
        % Update the solution using the Euler's improved method
        y(i + 1) = y(i) + h * f(t(i), y(i));
        y(i + 1) = y(i) + (h / 2) * (f(t(i), y(i)) + f(t(i+1), y(i+1)));

    end
end

% Define the ODE (dy/dt = y)
f = @(t, y) y + t; 

% Solve the ODE from t = 0 to t = 2 with initial condition y(0) = 1 and step size 0.01
[t, y] = solve_ode_euler_improved(f, [0 0.4], 1, 0.1);

% Print the solution
% disp(y);

% Plot the solution
plot(t, y);

% Add a title and labels to the axes
title('Solution of dy/dt = y using Euler method');
xlabel('Time (t)');
ylabel('Solution (y)');


% 7.4. Runge-Kutta method
function [t, y] = solve_ode_rk4(f, tspan, y0, h)
    % Solve an ODE using the 4th order Runge-Kutta method.
    %
    % Inputs:
    %   f - function handle for the ODE (dy/dt = f(t, y))
    %   tspan - 2-element vector specifying the time range [t0 tf]
    %   y0 - initial condition
    %   h - step size
    %
    % Outputs:
    %   t - vector of time points
    %   y - solution at each time point

    % Initialize the time points
    t = tspan(1):h:tspan(2);

    % Initialize the solution vector
    y = zeros(size(t));

    % Set the initial condition
    y(1) = y0;

    % Iterate over each time point
    for i = 1:(length(t) - 1)
        % Calculate the Runge-Kutta coefficients
        k1 = h * f(t(i), y(i));
        k2 = h * f(t(i) + h/2, y(i) + k1/2);
        k3 = h * f(t(i) + h/2, y(i) + k2/2);
        k4 = h * f(t(i) + h, y(i) + k3);

        % Update the solution
        y(i + 1) = y(i) + (k1 + 2*k2 + 2*k3 + k4) / 6;
    end
end

% Define the ODE (dy/dt = y)
f = @(t, y) y;

% Solve the ODE from t = 0 to t = 2 with initial condition y(0) = 1 and step size 0.01
[t, y] = solve_ode_rk4(f, [0 2], 1, 0.01);

% Print the solution
disp(y);

% Plot the solution
plot(t, y);

% Add a title and labels to the axes
title('Solution of dy/dt = y using Runge-Kutta method');
xlabel('Time (t)');
ylabel('Solution (y)');

%% End of chapter 7 
