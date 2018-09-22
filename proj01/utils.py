import math
import matplotlib.pyplot as plt

def print_cells(contents, size):
  print('|'.join([str(c).center(size) for c in contents]))


def plot(epoch, trn, dev, title):
  plt.style.use('ggplot')
  plt.plot(epoch, trn, label='trn')
  plt.plot(epoch, dev, label='dev')
  plt.title(title)
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()
