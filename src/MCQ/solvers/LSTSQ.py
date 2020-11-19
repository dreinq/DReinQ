from logging import Logger

import torch
import scipy.linalg

from .SolverBase import SolverBase

class LeastSquare(SolverBase):
    def __init__(self, M, K, backend="scipy", logger: Logger = None):
        super().__init__(backend, logger)
        self.M = M
        self.K = K
        self._shift = torch.arange(0, self.M * self.K, self.K).to("cuda" if backend == "torch" else "cpu")

    def solve(self, X: torch.Tensor, B: torch.Tensor, alternateWhenOutlier=False):
        if self._backend != "torch":
            B = B.detach().cpu()
            X = X.detach().cpu()
        if B.shape[-1] == self.M:
            # [N, M]
            iy = B + self._shift
            # [N, 1] -> [N, M]
            ix = torch.arange(X.shape[0], device=B.device)[:, None].expand_as(iy)
            b = torch.zeros((B.shape[0], self.M * self.K), device=B.device)
            b[[ix, iy]] = 1
        elif B.shape[-1] == self.M * self.K:
            b = B
        else:
            raise RuntimeError(f"B not match any of [{self.M}, {self.M * self.K}], got {B.shape[-1]}")
        if self._backend == "scipy":
            result, _, _, _ = scipy.linalg.lstsq(a=b.numpy(), b=X.numpy(), overwrite_a=True, overwrite_b=True, check_finite=False, lapack_driver="gelsy")
            return torch.from_numpy(result.reshape(self.M, self.K, -1)).cuda()
        elif self._backend == "cupy":
            raise NotImplementedError("Since CUDA > 9.1 can't solve svd in tall matrices, the cupy version still can't use.")
            # b_gpu = cp.asarray(b.cpu().numpy())
            # X_gpu = cp.asarray(X.cpu().numpy())
            # result, _, _, _ = cp.linalg.lstsq(a=b_gpu, b=X_gpu)
            # return cp.asnumpy(result).reshape(self.M, self.K, -1)
        else:
            result, _ = torch.lstsq(X.cuda(), b.cuda())
            if alternateWhenOutlier:
                mse = ((b @ result[:self.M * self.K] - X) ** 2).sum(-1).mean()
                if mse > 1e6 or torch.isnan(mse):
                    self._logger.debug("Meet irregular value, use scipy alternate")
                    result, _, _, _ = scipy.linalg.lstsq(a=b.detach().cpu().numpy(), b=X.detach().cpu().numpy(), check_finite=False, lapack_driver="gelsy")
                    return torch.from_numpy(result.reshape(self.M, self.K, -1)).cuda()
            return result[:self.M * self.K].reshape(self.M, self.K, -1).cuda()
