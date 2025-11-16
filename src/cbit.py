import numpy as np

"""

"""
class CBit():

    def __init__(self, num_bits: int):
        self.c_bits = np.array([0] * num_bits)

    def set_bit(self, index: int, value: int):
        if index < 0 or index >= self.c_bits.size:
            raise IndexError(f"Classical register index {index} is out of bounds!")
        if value not in (0, 1):
            raise ValueError("Classical bits only store values of 0 or 1!")
        self.c_bits[index] = value
    
    def get_bit(self, index: int):
        if index < 0 or index >= self.c_bits.size:
            raise IndexError(f"Classical register index {index} is out of bounds!")
        return self.c_bits[index]
    
    def get_bits(self):
        return self.c_bits
    
    def reset(self):
        self.c_bits.fill(0)
    
    def print_bit(self, bit):
        if self.c_bits.size == 0:
            raise ValueError("Classical register set is empty!")
        elif bit < 0 or bit >= self.c_bits.size:
            raise ValueError(f"Classical register index {bit} is out of bounds!")
        print(f"Classical register {bit} contains value: {self.c_bits[bit]}")
    
    def print_bits(self):
        if self.c_bits.size == 0:
            raise ValueError("Classical register set is empty!")
        for index, bit in enumerate(self.c_bits):
            print(f"Classical register {index} contains value: {bit}")
    
    def __len__(self):
        return self.c_bits.size

def main():
    bits = CBit(4)
    bits.print_bit(1)
    bits.print_bits()
    bits.set_bit(1, 1)
    bits.print_bit(1)

if __name__ == "__main__":
    main()