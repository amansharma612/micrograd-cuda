import ctypes

lib = ctypes.CDLL('./libops.so')

# Parameter Types of Shared Library Functions
lib.zeros_gpu.argtypes = (ctypes.c_int, ctypes.c_int)
lib.ones_gpu.argtypes = (ctypes.c_int, ctypes.c_int)
lib.move_to_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int)
lib.move_to_cpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int)
lib.matmul_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int)
lib.add_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)
lib.mul_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)
lib.scalar_mul_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int)
lib.pow_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int)
lib.tanh_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)
lib.relu_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)
lib.relugrad_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)
lib.transpose_gpu.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int)

# Return type of shared library functions
lib.move_to_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.matmul_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.add_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.mul_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.scalar_mul_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.pow_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.tanh_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.relu_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.relugrad_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.transpose_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.zeros_gpu.restype = ctypes.POINTER(ctypes.c_float)
lib.ones_gpu.restype = ctypes.POINTER(ctypes.c_float)




def reshape(mat_a, a_1, a_2):
    if len(mat_a) != a_1 * a_2:
        raise ValueError("Number of Elements do not match")

    x = [[mat_a[a_2 * i + j] for j in range(a_2)] for i in range(a_1)]
    return x

def flatten_matrix(mat_a , a_1, a_2) -> list:
    
    final = [mat_a[i][j] for i in range(a_1) for j in range(a_2)]
    return final

    


class OperationsGpu:
    @staticmethod
    def to(mat_a, a_1, a_2):
        new_mat_a = flatten_matrix(mat_a, a_1, a_2)
        new_mat_a_type = ctypes.c_float * len(new_mat_a)
        result = lib.move_to_gpu(new_mat_a_type(*new_mat_a), a_1 * a_2 * ctypes.sizeof(ctypes.c_float))
        return result

    @staticmethod
    def matrix_zeros(a_1, a_2):
        result = lib.zeros_gpu(a_1, a_2)
        return result
    
    @staticmethod
    def matrix_ones(a_1, a_2):
        result = lib.ones_gpu(a_1, a_2)
        return result

    @staticmethod
    def matmul(mat_a, mat_b, a_1, a_2, b_2):
        res = lib.matmul_gpu(mat_a, mat_b, a_1, a_2, b_2)
        return res

    @staticmethod
    def mul(mat_a, mat_b, a_1, a_2):
        res = lib.mul_gpu(mat_a, mat_b, a_1, a_2)
        return res

    @staticmethod
    def add(mat_a, mat_b, a_1, a_2):
        res = lib.add_gpu(mat_a, mat_b, a_1, a_2)
        return res

    @staticmethod
    def scalar_mul(mat_a, val, a_1, a_2):
        res = lib.scalar_mul_gpu(mat_a, val, a_1, a_2)
        return res

    @staticmethod
    def pow(mat_a, val, a_1, a_2):
        res = lib.pow_gpu(mat_a, val, a_1, a_2)
        return res

    @staticmethod
    def tanh(mat_a, a_1, a_2):
        res = lib.tanh_gpu(mat_a, a_1, a_2)
        return res
    
    @staticmethod
    def transpose(mat_a, a_1, a_2):
        res = lib.transpose_gpu(mat_a, a_1, a_2)
        return res

    @staticmethod
    def relu(mat_a, a_1, a_2):
        res = lib.relu_gpu(mat_a, a_1, a_2)
        return res
    
    @staticmethod
    def relugrad(mat_a, a_1, a_2):
        res = lib.relugrad_gpu(mat_a, a_1, a_2)
        return res

class OperationsCpu:
    
    @staticmethod
    def to(mat_a, a_1, a_2):
        zero_mat = [0 for i in range(a_1 * a_2)]
        zero_mat_type = ctypes.c_float * len(zero_mat)
        result = zero_mat_type(*zero_mat)
        lib.move_to_cpu(result, mat_a, a_1 * a_2 * ctypes.sizeof(ctypes.c_float))
        result = result[: a_1 * a_2]
        return reshape(result, a_1, a_2)
    
    @staticmethod
    def matrix_ones(a_1, a_2):
        return [[1 for j in range(a_2)] for i in range(a_1)]
    
    @staticmethod
    def matrix_zeros(a_1, a_2):
        return [[1 for j in range(a_2)] for i in range(a_1)]

class Operations:
    @staticmethod
    def __new__(self, device):
        if device == "cpu":
            return OperationsCpu
        if device == "gpu":
            return OperationsGpu

    
