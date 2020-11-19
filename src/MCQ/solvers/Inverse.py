import torch
import numpy as np

from .SolverBase import SolverBase
class PInverse(SolverBase):
    def __init__(self, M, K, backend="scipy"):
        super().__init__(backend)
        self.M = M
        self.K = K
        self._shift = torch.arange(0, self.M * self.K, self.K).cuda()
        self._previous = 1e6

    def solve(self, X: torch.Tensor, B: torch.Tensor, alternateWhenOutlier=False):
        """ Solve BC = X by calculating C = B_inv @ X """
        if B.shape[-1] == self.M:
            # [N, M]
            iy = B + self._shift
            # [N, 1] -> [N, M]
            ix = torch.arange(X.shape[0])[:, None].expand_as(iy).cuda()
            b = torch.zeros(B.shape[0], self.M * self.K, device="cuda")
            b[[ix, iy]] = 1
        elif B.shape[-1] == self.M * self.K:
            b = B.cuda()
        else:
            raise RuntimeError(f"B not match any of [{self.M}, {self.M * self.K}], got {B.shape[-1]}")
        if self._backend == "scipy":
            # [M, N]
            b_inv = np.linalg.pinv(b.detach().cpu().numpy())
            result = b_inv @ X.detach().cpu().numpy()
            return torch.from_numpy(result.reshape(self.M, self.K, -1)).cuda()
        elif self._backend == "cupy":
            raise NotImplementedError("Since CUDA > 9.1 can't solve svd in tall matrices, the cupy version still can't use.")
            # b_inv = cp.linalg.pinv(cp.fromDlpack(to_dlpack(b)))
            # result = from_dlpack(b_inv.toDlpack())
            # return result.reshape(self.M, self.K, -1).cuda()
        else:
            b_inv = torch.pinverse(b)
            result = b_inv @ X.cuda()
            if alternateWhenOutlier:
                mse = ((b @ result - X) ** 2).sum(-1).mean()
                if mse > 2.0 * self._previous or torch.isnan(mse):
                    self._backend = "scipy"
                    r = self.solve(X, b)
                    self._backend = "torch"
                    self._previous = ((b @ r.reshape(self.M * self.K, -1) - X) ** 2).sum(-1).mean()
                    return r
                else:
                    self._previous = mse
            return result.reshape(self.M, self.K, -1).cuda()
