import numpy as np

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

if __name__ == '__main__':
    print(entropy(2/3))
