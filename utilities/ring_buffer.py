import numpy as np

class ring_buffer:
    def __init__(self, size):
        self.buffer = np.zeros((size,), dtype=float)
        self.in_ptr = 0
        self.out_ptr = 0
        self.size = size

    def put(self, data):
        for i in range(data.shape[0]):
            self.buffer[self.in_ptr] = data[i]
            self.in_ptr = (self.in_ptr + 1)%self.size

    def get(self, data, length, step):
        # print(f"Ring_buffer length {length}, step {step}, out_ptr  {self.out_ptr}")
        ptr = self.out_ptr
        for i in range(length):
            data[i] = self.buffer[ptr]
            ptr = (ptr + 1)%self.size
        self.out_ptr = (self.out_ptr + step)%self.size

    def get_length(self):
        if self.in_ptr - self.out_ptr >= 0:
            length = self.in_ptr - self.out_ptr
        else:
            length = self.size + self.in_ptr - self.out_ptr
        return length


