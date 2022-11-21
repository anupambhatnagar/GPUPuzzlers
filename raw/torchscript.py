import torch

@torch.jit.script
def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r


def bar(x, y, z):
    return foo(x, y) + z

traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3)))
print(traced_bar.code)
print(traced_bar.graph)

@torch.jit.script
def foo(len):
    # type: (int) -> torch.Tensor
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv

print(foo.code)
print(foo.graph)
