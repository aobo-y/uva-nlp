import os

dname = os.path.dirname(__file__)

def load_data(filename):
  lines= open(dname + '/' + filename).read().strip().split('\n')

  data = [[t.split('/') for t in l.strip().split(' ')] for l in lines]
  return data


def write_data(filename, data):
  open(dname + '/' + filename, 'w+').write(data)
