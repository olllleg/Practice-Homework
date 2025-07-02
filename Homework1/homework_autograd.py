import torch
import numpy
import math
# Задание 2: Автоматическое дифференцирование

# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor([5.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)
z = torch.tensor([8.0], requires_grad=True)

# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2*x*y*z
print(f'f = {f}')

# Найдите градиенты по всем переменным
f.backward()
print(f'x.grad = {x.grad}')
print(f'y.grad = {y.grad}')
print(f'z.grad = {z.grad}')

# Проверьте результат аналитически
'''Градиент переменной y оказался наибольшим, x и z равным,
   я понимаю это так, что переменная y сильнее всего влияет на функцию потерь'''


# Реализуйте функцию MSE (Mean Squared Error):
x = torch.tensor(4.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# MSE = (1/n) * Σ(y_pred - y_true)^2
y_pred = w * x + b
mse = (y_pred - 9)**2
# где y_pred = w * x + b (линейная функция)
mse.backward()

# Найдите градиенты по w и b
print(f'w.grad = {w.grad}')
print(f'b.grad = {b.grad}')
'''Градиент w является квадратом градиента b'''


# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
x = torch.tensor(2.0, requires_grad=True)

y = torch.sin(x**2 + 1)


# Найдите градиент df/dx

print(math.cos(2**2 + 1) * 4)
# Проверьте результат с помощью torch.autograd.grad
print(torch.autograd.grad(y, x)[0])
