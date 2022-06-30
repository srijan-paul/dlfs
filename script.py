# y = 5 * x1 + 3 * x2 + x3
from random import randint, random


n = int(input("Enter the size of the dataset: "))

out_str = 'x1, x2, x3, y\n'
for i in range(n):
  # 1. assume x1, x2, and x3 to be an arbitrary value.
  x1 = randint(0, 10)
  x2 = randint(0, 20)
  x3 = randint(0, 5)

  # 2. Calculate y accordingly
  y = 5*x1 + 3*x2 + x3
  out_str += f'{x1}, {x2}, {x3}, {y}\n'
  print(f'5*{x1} 3*{x2} {x3} = {y}')

with open("data.csv", "w") as outfile:
  outfile.write(out_str)
