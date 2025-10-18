import numpy as np

class custom_channel:

    def __init__(self, kraus_ops):
        self.kraus_ops = kraus_ops
        self._check_completeness()
    
    def _check_completeness(self):
        I = np.eye(self.kraus_ops[0].shape[0])
        total = sum(K.conj().T @ K for K in self.kraus_ops)
        if not np.allclose(total, I):
            raise ValueError("Kraus operators are not complete!")
    
    def apply_density_matrix(self, rho):
        return sum(K @ rho @ K.conj().T for K in self.kraus_ops)
    

def main():
    p = 0.1
    K0 = np.sqrt(1 - p) * np.eye(2)
    K1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])

    kraus_ops = [K0, K1]
    ops = custom_channel(kraus_ops=kraus_ops)
    print(ops.kraus_ops)

    rho0 = np.array([[1, 0], [0, 0]])
    rho_final = ops.apply_density_matrix(rho0)
    print(rho_final)

if __name__ == "__main__":
    main()