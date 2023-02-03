import torch
import os
from timeit import default_timer as timer

'''
MWE for static runtime
'''
 
class DeepAndWide(torch.nn.Module):
    def __init__(self, num_features=50):
        super(DeepAndWide, self).__init__()
        self.mu = torch.rand(1, num_features)
        self.sigma = torch.rand(1, num_features)
        self.fc_w = torch.rand(1, num_features + 1)
        self.fc_b = torch.rand(1)
 
    def forward(self, ad_emb_packed, user_emb, wide):
        wide_offset = wide + self.mu
        wide_normalized = wide_offset * self.sigma
        wide_preproc = torch.clamp(wide_normalized, 0., 10.)
        user_emb_t = torch.transpose(user_emb, 1, 2)
        dp_unflatten = torch.bmm(ad_emb_packed, user_emb_t)
        dp = torch.flatten(dp_unflatten, 1, -1)
        inp = torch.cat([dp, wide_preproc], 1)
        fc1 = torch.addmm(self.fc_b, inp, torch.t(self.fc_w), beta=1, alpha=1)
        return torch.sigmoid(fc1)
 
 
if __name__ == "__main__":
    # Use JIT's simple executor
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)
    torch.manual_seed(1337)
 
    # Construct the model
    num_features = 50
    m = torch.jit.script(DeepAndWide(num_features))
    m.eval()
 
    # Phabricate sample inputs
    batch_size = 1
    embedding_size = 32
    ad_emb_packed = torch.rand(batch_size, 1, embedding_size)
    user_emb = torch.rand(batch_size, 1, embedding_size)
    wide = torch.rand(batch_size, num_features)
    inps = (ad_emb_packed, user_emb, wide)
 
 
    warmup = 10
    iters = 20000
 
    if os.environ.get('TEST_USE_STATIC_RUNTIME'):
        # Construct Static Runtime version
        static_runtime = torch._C._jit_to_static_runtime(m._c)
 
        # Warmup
        for _ in range(warmup):
            y = static_runtime.run(inps)
 
        # Run the model
        start = timer()
        for _ in range(iters):
            y = static_runtime.run(inps)
        end = timer()
    else:
        # Warmup
        for _ in range(warmup):
            y = m(*inps)
 
        # Run the model
        start = timer()
        for _ in range(iters):
            y = m(*inps)
        end = timer()
 
    print(end - start)
