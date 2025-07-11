import numpy as np
from antarrlib.imaging import reconstruct_x


def test_reconstruct_x():
    N, T = 5, 10
    rng = np.random.default_rng(0)
    X_true = (rng.standard_normal((N, T)) + 1j*rng.standard_normal((N, T))) / np.sqrt(2)
    R = X_true @ X_true.conj().T / T

    X_hat = reconstruct_x(R, T, valid=True)
    print("X_hat.shape =", X_hat.shape)
    R_hat = X_hat @ X_hat.conj().T / T
    print("‖R - R_hat‖_F =", np.linalg.norm(R - R_hat))
    assert np.allclose(R, R_hat), (
        "Reconstructed covariance matrix does not match the original: "
        f"‖R - R_hat‖_F = {np.linalg.norm(R - R_hat)}"
    )
