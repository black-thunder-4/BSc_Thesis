import numpy as np
import torch


def kabsch_align(P, Q):
    """
    Align Q (pred) onto P (true) with a proper rotation (no reflection) + translation.
    P, Q: (L, 3) NumPy arrays or Torch tensors (row-vector points)
    Returns: Q_aligned (L,3), R (3,3), t (3,)
    Uses row-vector convention: X' = X @ R + t
    """
    is_torch = torch.is_tensor(P)
    xp = torch if is_torch else np

    # means
    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)

    # center
    Pc = P - muP
    Qc = Q - muQ

    # covariance (pred -> true)
    H = Qc.T @ Pc if not is_torch else Qc.transpose(0,1) @ Pc

    # SVD
    if is_torch:
        U, S, Vt = torch.linalg.svd(H)
        R = U @ Vt
        # reflection fix
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = muP - (muQ @ R)
        Q_aligned = (Q @ R) + t
    else:
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        # reflection fix
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = muP - (muQ @ R)
        Q_aligned = (Q @ R) + t

    return Q_aligned


def kabsch_align_batch(P, Q, mask):
    """
    P, Q: (B, L, 3)
    mask: (B, L)
    Returns:
        Q_aligned: (B, L, 3)
        R: (B, 3, 3)
        t: (B, 1, 3)
    """
    eps = 1e-8
    mask = mask.unsqueeze(-1)  # (B, L, 1)

    # 1) masked centroids
    w = mask.sum(dim=1, keepdim=True).clamp(min=eps)
    muP = (P * mask).sum(dim=1, keepdim=True) / w
    muQ = (Q * mask).sum(dim=1, keepdim=True) / w

    Pc = P - muP
    Qc = Q - muQ

    # 2) masked covariance
    Pc_masked = Pc * mask
    Qc_masked = Qc * mask
    H = torch.matmul(Qc_masked.transpose(1, 2), Pc_masked)  # (B, 3, 3)

    # 3) SVD
    U, S, Vt = torch.linalg.svd(H)

    # 4) rotation
    R = torch.matmul(U, Vt)

    # 5) fix improper rotation (det<0) without breaking gradients
    det = torch.linalg.det(R).unsqueeze(-1).unsqueeze(-1)
    S_fix = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(R.size(0),1,1)
    S_fix[:, 2, 2] = torch.sign(det).squeeze()
    R = torch.matmul(U, torch.matmul(S_fix, Vt))

    # 6) translation
    t = muP - torch.matmul(muQ, R)

    # 7) apply alignment
    Q_aligned = torch.matmul(Q, R) + t

    return Q_aligned