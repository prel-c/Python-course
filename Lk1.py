# Обухов Михаил

# Библиотеки Python для научных расчётов и машинного обученичя
import sys
import array
import numpy as np
print(np.__version__)

# Машинноое обучение расботает тголько с чилами

# в питоне динамическая типизация - поэтому мы не моежм заранее знать тип перпеменной
x = 1
print(type(x))
x = "hello"
print(type(x))
# поэтмоу чтобы поддерджать возможность ДТ нужно хранить много доп инфы
# хранит само значение, счётчик ссылок - это влечёт лищние расоды

l = [True, "2", 3.0, 1]
print([type(i) for i in l])

print(sys.getsizeof(l))

l1 = []
print(sys.getsizeof(l1))

al = array.array('i', [])
print(type(al))
print(sys.getsizeof(al))

al = array.array('i', [1])
print(sys.getsizeof(al))

al = array.array('i', [1,2])
print(sys.getsizeof(al))

# NumPy & Python array - массивы хранят элементы одного типа
# способы создания списка NumPy
# 1) создание из списка
l = []
a = np.array(l)
print(a, type(a))

print('list(python)', sys.getsizeof(l))
ap = array.array('i', l)
print('array(python)', sys.getsizeof(ap))
print('array(NumPy)', sys.getsizeof(a))

# "повышающее" приведение типов
a = np.array([1.01,2,3,4,5, 'a'])
print(type(a), a)

# явно задать тип
a = np.array([1.99,2,3,4,5], dtype=int)
print(type(a), a)

# одномерные массивы
a = np.array(range(2,5))
print(a)

# многмерыне массивы
a = np.array([range(i, i+5) for i in [1,2,3]])
print(a)
# 1 2 3 4 5
# 2 3 4 5 6
# 3 4 5 6 7

# с нуля
print(np.zeros(10, dtype=int))

# многомерный из 1
print(np.ones((3,5), dtype=float))

# можно предопределить значение
print(np.full((3,3), 3.1415, dtype=float))

# линейная послежовательность чисел
print(np.arange(0,20,2))

# генерация знчений из интервала с шагом
print(np.linspace(0, 1, 11))

# заполнить массив случайцными величинами по распеределению
# равномерное распределение от 0 до 1
print(np.random.random((2,4)))

# нормальное распределение
print(np.random.normal(0,1,(2,4)))

# равномерное распределение от x до y
print(np.random.randint(0,5,(2,2)))

# единичная матрица
print(np.eye(5, dtype=int))

# числовые типы данных NumPy

a1 = np.zeros(10, dtype=int)
a2 = np.zeros(10, dtype='int16')
a3 = np.zeros(10, dtype=np.int16)
print(a1, type(a1), a1.dtype) # тип Python
print(a2, type(a2), a2.dtype) # тип NumPy
print(a3, type(a3), a3.dtype) # тип NumPy

# a1 = np.zeros(10, dtype=int16)
# NameError

# Numerical Python = NumPy
# - работа с атрибутами массивов
# - индексация
# - срезы
# - изменение формы массивов
# - разбиение и объединение

# атрибуты массива: ndim - число размерностей, shape - величина каждой размерности, size - общий размер массива
np.random.seed(1)
x1 = np.random.randint(10, size = 3)
print(x1)
print(x1.ndim, x1.shape, x1.size)

x2 = np.random.randint(10, size = (3,2))
print(x2)
print(x2.ndim, x2.shape, x2.size)

x3= np.random.randint(10, size = (3,2,2))
print(x3)
print(x3.ndim, x3.shape, x3.size)

# - индексация
# одномерный
a = np.array([1,2,3,4,5])
print(a[0])
print(a[-2])

a[1] = 20
print(a[1])

# многомерный
a = np.array([[1,2], [3,4]])
print(a)

print(a[0,0])
print(a[-1,-2]) # строка -1, столбец -2

# вставки
a = np.array([1,2,3,4,5])
print(a.dtype)

a[0] = 3.14
print(a)
print(a.dtype)

a.dtype = float

a[0] = 3.14
print(a)
print(a.dtype)
# типы заданные вначале лучше не менять и массивы лениво меняют типы, только явно через dtype

# - срезы - подмассив массива [начало:конец:шаг] - [::] = [0,конец,1]

a = np.array([1,2,3,4,5])
print(a[:3])
print(a[3:])
print(a[1:4])
print(a[::2])
print(a[1::2])
# если шаг < 0, то конец и начало меняются местами
print(a[::-1])

# срезы в многмоерных массивах
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[:2,:3])
print(a[:,::2])
print(a[::-1, ::-1])
print(a[:,0])
print(a[0,:])
print(a[0])

# Срезы в Python - копии подмассивов, а в NumPy - представление (view)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)
a_2x2 = a[:2, :2]
print(a_2x2)

a_2x2[0,0] = 999
print(a)

a_2x2 = a[:2, :2].copy()
a_2x2[0,0] = 1001
print(a)
print(a_2x2)

# изменение формы массива
a = np.arange(1,13)
print(a, a.shape, a.ndim)
print(a[3], a[11])

a1 = a.reshape(1,12)
print(a1, a1.shape, a1.ndim)
print(a1[0,3], a1[0,11])

a2 = a.reshape(2,6)
print(a2, a2.shape, a2.ndim)

a3 = a.reshape(2,2,3)
print(a3, a3.shape, a3.ndim)
print(a3[0,1,2])

# 12 = 2*2*3 = 1*12*1*1

a4 = a.reshape(1,12,1,1)
print(a4, a4.shape, a4.ndim)
print(a4[0,2,0,0])

a5 = a.reshape((2,6)) # - лучще прям явно кортежем писать размерность
print(a5, a5.shape, a5.ndim)
print(a5[1,4])

a6 = a.reshape((2,6), order="F")
print(a6, a6.shape, a6.ndim)
print(a6[1,4])
# порядок F меняет порядок заполнениея элемпентов (либо сначала строка потом столбец +, либо наоброт)

a = np.arange(1,13)
print(a, a.shape, a.ndim)
print(a[3], a[11])

a1 = a.reshape(1,12)
print(a1, a1.shape, a1.ndim)
a2 = a[np.newaxis, :]
print(a2, a2.shape, a2.ndim)
a3 = a[:, np.newaxis]
print(a3, a3.shape, a3.ndim)
