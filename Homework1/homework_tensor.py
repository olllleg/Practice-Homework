import torch
import numpy as np


#Задание 1: Создание и манипуляции с тензорами


# Создайте следующие тензоры:
# - Тензор размером 2x3x4, заполненный нулями
zeros_tensor = torch.zeros((2, 3, 4))
print('zeros:', zeros_tensor)
# - Тензор размером 5x5, заполненный единицами
ones_tensor = torch.ones((5, 5))
print('ones:', ones_tensor)
# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
arrange_tensor = torch.arange(0, 16).reshape(4, 4)
print('arrange:', arrange_tensor)
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
random_tensor = torch.randint(0, 2, (3, 4))
print('rand:', random_tensor)


# Дано: тензор A размером 3x4 и тензор B размером 4x3
#A = torch.rand((3, 4))
A = torch.tensor([[10, 2, 5, 5], [3, 4, 6, 3], [2, 5, 7, 9]], dtype=torch.float32)
B = torch.tensor([[10, 2, 5], [3, 4, 6], [2, 7, 9], [1, 1, 1]], dtype=torch.float32)
# Выполните:
# - Транспонирование тензора A
transposed_A = torch.transpose(A, 0, 1)
print('transposed_A:', transposed_A)
# - Матричное умножение A и B
print(f'torch.matmul(A, B) = {torch.matmul(A, B)}')
# - Поэлементное умножение A и транспонированного B
transposed_B = torch.transpose(B, 0, 1)
print(f'A * transposed_B = {A * transposed_B}')
# - Вычислите сумму всех элементов тензора A
print('sum_A:', A.sum())

# Создайте тензор размером 5x5x5
tensor_for_slicing = torch.arange(0, 125).reshape(5, 5, 5)
print(tensor_for_slicing)
# Извлеките:
# - Первую строку
# - я сперва взял только одну первую строку [0][0], но потом решил, что имеется ввиду другое
print('first_row:', tensor_for_slicing[:, 0, :])
# - Последний столбец
print('last_column:', tensor_for_slicing[:, :, -1])
# - Подматрицу размером 2x2 из центра тензора
print('submatrix:', tensor_for_slicing[2, 1:3, 1:3])
# - Все элементы с четными индексами
# - только те у которых все индексы чётные
print('even:', tensor_for_slicing[::2, ::2, ::2])

# Создайте тензор размером 24 элемента
changing_tensor= torch.arange(0, 24)
# Преобразуйте его в формы:
# - 2x12
print('2x12:',changing_tensor.reshape(2, 12))
# - 3x8
print('3x8:',changing_tensor.reshape(3, 8))
# - 4x6
print('4x6:',changing_tensor.reshape(4, 6))
# - 2x3x4
print('2x3x4:',changing_tensor.reshape(2, 3, 4))
# - 2x2x2x3
print('2x2x2x3:',changing_tensor.reshape(2, 2, 2, 3))