from ops import Operations
import time



def test_gpu_to_cpu():
    a = [[1, 1], [1, 1]]
    y = ops.to(a, 2, 2)
    ops.to_cpu(y, 2, 2)

def test_matmul_gpu(sz):
    a = [[1 for i in range(sz)] for j in range(sz)]
    b = [[2 for i in range(1)] for j in range(sz)]
    x = ops.to(a, sz, sz)
    y = ops.to(b, sz, 1)

    res  = ops.matmul_on_gpu(x, y, sz, sz, 1)
    # ops.to_cpu(res, 10, sz)

def test_add_gpu(sz):
    a = [[1 for i in range(sz)] for j in range(sz)]
    b = [[2 for i in range(sz)] for j in range(sz)]
    x = Operations("gpu").to(a, sz, sz)
    y = Operations("gpu").to(b, sz, sz)

    res  = Operations("gpu").add(x, y, sz, sz)
    Operations("cpu").to(res, sz, sz)
    # ops.to_cpu(res, sz, sz)

def test_scalar_mul_gpu(sz):
    a = [[1 for i in range(sz)] for j in range(sz)]
    x = ops.to(a, sz, sz)
    res = ops.scalar_mul_on_gpu(x, 3, sz, sz)
    ops.to_cpu(res, sz, sz)

def test_tanh_gpu(sz):
    a = [[0.54124 for i in range(sz)] for j in range(sz)]
    x = ops.to(a, sz, sz)
    res = ops.tanh_on_gpu(x, sz, sz)
    ops.to_cpu(res, sz, sz)


def test_relu_gpu(sz):
    a = [[(-1) ** (2 * i + j) for i in range(sz)] for j in range(sz)]
    x = ops.to(a, sz, sz)
    res = ops.relu_on_gpu(x, sz, sz)
    ops.to_cpu(res, sz, sz)


# def test_mul(sz):


def test_ones(sz):  
    y = Operations("gpu").matrix_ones(sz, sz)
    Operations("cpu").to(y, sz, sz)




if __name__ == "__main__":
    # Run your tests here
    start = time.time()
    test_mul(3)
    end = time.time()
    print(end - start)
