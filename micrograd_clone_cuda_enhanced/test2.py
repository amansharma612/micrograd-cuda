from engine import Tensor

a = Tensor([[1], [3], [9]], shape = (3, 1))
b = Tensor([[1, 2]], shape = (1, 2))
print(a)
a.to("gpu")
print(b)
b.to("gpu")
c = a @ b
c.to("cpu")
print(c)
print(c.shape)


from ops import Operations


# def test(sz1, sz2, sz3):
#     a = [[2 for i in range(sz2)] for j in range(sz1)]
#     x = Operations("gpu").to(a, sz1, sz2)
#     b = [[2 for i in range(sz3)] for j in range(sz2)]
#     y = Operations("gpu").to(b, sz2, sz3)
#     res = Operations("gpu").matmul(x, y, sz1, sz2, sz3)
#     res = Operations("cpu").to(res, sz1, sz3)
#     print(res)


# test(1, 4, 1)
