import torch
import random

'''
Raw examples of nondeterminism in GPU compute, from Yu Guo.
'''

# from yu guo: on deterministic operators, you can try https://fburl.com/code/bhciu9nh without DeterministicGuard(True)
# index_add as well: https://fburl.com/code/4slorwuj

class DeterministicGuard:
    def __init__(self, deterministic, *, warn_only=False):
        self.deterministic = deterministic
        self.warn_only = warn_only

    def __enter__(self):
        self.deterministic_restore = torch.are_deterministic_algorithms_enabled()
        self.warn_only_restore = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(
            self.deterministic,
            warn_only=self.warn_only)

    def __exit__(self, exception_type, exception_value, traceback):
        torch.use_deterministic_algorithms(
            self.deterministic_restore,
            warn_only=self.warn_only_restore)

device = 'cuda'

with DeterministicGuard(True):
    m = random.randint(20, 30)
    elems = random.randint(2000 * m, 3000 * m)
    dim = 0
    src = torch.randn(elems, device=device)
    idx = torch.randint(m, (elems,), device=device)

    x = torch.zeros(m, device=device)
    res = x.scatter_add(dim, idx, src)

    expected = torch.zeros(m, device=device)
    for i in range(elems):
        expected[idx[i]] += src[i]

    #assertEqual(res, expected, atol=0, rtol=0)
    diff = abs(res - expected)
    for i in range(len(diff)):
        if diff[i] != 0.0:
            print(i, res[i], expected[i], diff[i])

def _prepare_data_for_index_copy_and_add_deterministic(dim, device):
    assert (dim >= 0 and dim < 3)
    a = [5, 4, 3]
    a[dim] = 2000
    x = torch.zeros(a, device=device)
    b = a.copy()
    elems = a[dim] * 20
    b[dim] = elems
    src = torch.rand(b, device=device)
    index = torch.randint(a[dim], (elems,), device=device)
    return (x, index, src)


def assertEqual(A, B, atol, rtol):
    print(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            for k in range(len(A[0][0])):
                if abs(A[i][j][k] - B[i][j][k]) > atol or abs(A[i][j][k] - B[i][j][k])/(1e-12 + max(abs(A[i][j][k]), abs(B[i][j][k]))) > rtol:
                    return False
    return True

for dim in range(3):
    x, index, src = _prepare_data_for_index_copy_and_add_deterministic(dim, device)
    alpha = random.random() + 1
    # on CPU it should be deterministic regardless of the deterministic mode
    with DeterministicGuard(True):
        y0 = torch.index_add(x, dim, index, src, alpha=alpha)
        for _ in range(3):
            y = torch.index_add(x, dim, index, src, alpha=alpha)
            #self.assertEqual(y, y0, atol=0, rtol=0)
            assertEqual(y, y0, atol=0, rtol=0)

    with DeterministicGuard(False):
        y0 = torch.index_add(x, dim, index, src, alpha=alpha)
        for _ in range(3):
            y_nd = torch.index_add(x, dim, index, src, alpha=alpha)
            #self.assertEqual(y_nd, y0, atol=1e-3, rtol=1e-5)
            assertEqual(y_nd, y0, atol=1e-3, rtol=1e-5)

        diff = abs(res - expected)
        for i in range(len(diff)):
            if diff[i] != 0.0:
                print(i, res[i], expected[i], diff[i])

'''
Yu: both these two operators have cuda atomic_add. if you try index_copy_, you should see another kind of race
index_copy: https://fburl.com/code/2wa2ig57
'''
