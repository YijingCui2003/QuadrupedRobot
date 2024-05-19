import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import solve, inv, pinv
import numpy as np


def bmv(A, b):
    """Compute matrix multiply vector in batch mode."""
    bs = b.shape[0]
    if A.shape[0] == 1:
        # The same A for different b's; use matrix multiplication instead of broadcasting
        return (A.squeeze(0) @ b.t()).t()
    else:
        return (A @ b.unsqueeze(-1)).squeeze(-1)


def bma(A, B):
    """Batch-matrix-times-any, where any can be matrix or vector."""
    return (A @ B) if A.dim() == B.dim() else bmv(A, B)


def bsolve(A, B):
    """Compute solve(A, B) in batch mode, where the first dimension of A can be singleton."""
    if A.dim() == 3 and B.dim() == 2 and A.shape[0] == 1:
        return torch.linalg.solve(A.squeeze(0), B.t()).t()
    else:
        return torch.linalg.solve(A, B)


class QPSolver(nn.Module):
    """
    Solve QP problem:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b >= 0,
    where x in R^n, b in R^m.
    """

    def __init__(
        self, device, n, m, alpha=1, beta=1, symmetric_constraint=False, buffered=False
    ):
        """
        Initialize the QP solver.

        device: PyTorch device

        n, m: dimensions of decision variable x and constraint vector b

        P, Pinv, H: Optional matrices that define the QP. If not provided, must be supplied during forward pass. At most one of P and Pinv can be specified.

        alpha, beta: Parameters of the PDHG algorithm

        preconditioner: Optional preconditioner module

        symmetric_constraint: Flag for making the inequality constraint symmetric; when True, the constraint is assumed to be -1 <= Hx + b <= 1, instead of Hx + b >= 0.

        buffered: Flag for indicating whether the problem is modeled with the buffer variable \epsilon. When True, it is assumed that the first (n-1) decision variables are the original x, and the last decision variable is \epsilon; in this case, if symmetric constraint is enabled, then the projection is done as follows:
        1. Project epsilon to [0, +\infty)
        2. Project H_x x + b_x to [-1 - eps, 1 + eps]

        Note: Assumes that H is full column rank when m >= n, and full row rank otherwise.
        """
        super().__init__()
        self.device = device
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.symmetric_constraint = symmetric_constraint
        self.buffered = buffered

        self.bIm = torch.eye(m, device=device).unsqueeze(0)
        self.X0 = torch.zeros((1, 2 * self.m), device=self.device)

    def get_sol_transform(self, H, bP):
        """
        Computes the transformation from dual variable z to primal variable x.

        H: Constraint matrix
        bP, bPinv: Either the matrix P or its inverse. Exactly one must be specified. Specifying Pinv can reduce number of linear solves.

        Returns: Function that performs the transformation
        """
        bH = H
        if self.m >= self.n:
            return lambda z, q, b: bmv(pinv(bH), z - b)
        else:

            def get_sol(z, q, b):
                t = lambda bM: bM.transpose(-1, -2)
                bPinvHt = solve(bP, t(bH))
                Mt = solve(t(bH @ bPinvHt), t(bPinvHt))
                M = t(Mt)
                bPinvq = solve(bP, q)
                return bmv(M @ bH, bPinvq) - bPinvq + bmv(M, z - b)

            return get_sol

    def get_AB(self, q, b, H, P):
        """
        Computes matrices A and B used in the PDHG iterations.

        q, b: Coefficients in the objective and constraint
        H, P, Pinv: Matrix H, and (either the matrix P or its inverse). Must be specified if not initialized. Specifying Pinv can reduce number of linear solves.

        Returns: Matrices A and B
        """
        # preconditioner: (bs, m, m)
        D = torch.eye(self.m, device=self.device)
        D /= self.beta
        # bHPinvHt = H @ solve(P, H.transpose(-1, -2))
        # tD_inv = D + bHPinvHt
        # tD = inv(tD_inv)  # (*, m, m)

        tD = D - H @ inv(P + H.transpose(-1, -2) @ H) @ H.transpose(-1, -2)  # (*, m, m

        mu = bmv(tD, bmv(H, bsolve(P, q)) - b)  # (bs, m)
        tDD = tD @ D

        A = torch.cat(
            [
                torch.cat([tDD, tD], dim=2),
                torch.cat(
                    [
                        -2 * self.alpha * tDD + self.bIm,
                        self.bIm - 2 * self.alpha * tD,
                    ],
                    dim=2,
                ),
            ],
            dim=1,
        )  # (bs, 2m, 2m)
        B = torch.cat([mu, -2 * self.alpha * mu], 1)  # (bs, 2m)
        return A, B

    def compute_residuals(self, x, z, u, q, b, P, H):
        """
        Computes the primal and dual residuals.

        x, z: Primal variables
        u: Dual variable
        q, b: Coefficients in the objective and constraint
        P, H, Pinv: Optional matrices defining the QP. Must be provided if not initialized.

        Returns: Primal and dual residuals
        """
        # Compute primal residual: Hx + b - z
        primal_residual = bmv(H, x) + b - z

        # Compute dual residual: Px + q - H'u
        dual_residual = bmv(P, x) + q - bmv(H.transpose(-1, -2), u)

        return primal_residual, dual_residual

    def forward(self, q, b, P, H, X0=None, iters=1000, return_residuals=False):
        """
        Solves the QP problem using PDHG.

        q, b: Coefficients in the objective and constraint
        P, H, Pinv: Optional matrices defining the QP, i.e., matrix H, and (either the matrix P or its inverse). Must be provided if not initialized. Using Pinv is more efficient in learned setting.
        iters: Number of PDHG iterations

        Returns: History of primal-dual variables, primal solutions, and optionally residuals of the last iteration
        """
        # q: (bs, n), b: (bs, m)
        bs = q.shape[0]
        if X0 is not None:
            X = X0
        else:
            X = torch.zeros((1, 2 * self.m), device=self.device)
        A, B = self.get_AB(q, b, H, P)
        for k in range(1, iters + 1):
            # PDHG update
            X = bmv(A, X) + B  # (bs, 2m)
            if not self.symmetric_constraint:
                # Project to [0, +\infty)
                F.relu(X[:, self.m :], inplace=True)
            else:
                if not self.buffered:
                    # Project to [-1, 1]
                    projected = torch.clamp(X[:, self.m :], -1, 1)
                    X = torch.cat((X[:, : self.m], projected), dim=1)
                else:
                    # Hybrid projection: epsilon to [0, +\infty), the rest decision variables to [-1 - eps, 1 + eps]
                    # Project epsilon
                    F.relu(X[:, -1:], inplace=True)
                    # Project the rest variables
                    projected = torch.clamp(
                        X[:, self.m : -1], -1 - X[:, -1:], 1 + X[:, -1:]
                    )
                    # Concatenate
                    X = torch.cat((X[:, : self.m], projected, X[:, -1:]), dim=1)
        # Compute residuals for the last step if the flag is set
        if return_residuals:
            get_sol = self.get_sol_transform(H, P)
            x_last = get_sol(X[:, self.m :], q, b)
            z_last = X[:, self.m :]
            u_last = X[:, : self.m]
            primal_residual, dual_residual = self.compute_residuals(
                x_last, z_last, u_last, q, b, P, H
            )
            return X, x_last, (primal_residual, dual_residual)
        else:
            get_sol = self.get_sol_transform(H, P)
            primal_sols = get_sol(X[:, self.m :], q, b)
            return X, primal_sols
