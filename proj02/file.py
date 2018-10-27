import os

dname = os.path.dirname(__file__)

def load_data(filename):
  lines= open(dname + '/' + filename).read().strip().split('\n')

  data = [[t.split('/') for t in l.strip().split(' ')] for l in lines]

  # Should not convert all to lower case, coz it matters
  # ex. New in New York, New Zealand are N, new is A
  # if lowercase:
  #   for line in data:
  #     for tokens in line:
  #       tokens[0] = tokens[0].lower()

  return data


def write_data(filename, data):
  open(dname + '/' + filename, 'w+').write(data)
