import os
import math
import pickle
import argparse
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.special import roots_legendre, eval_jacobi

# ═════════════════════════════════════════════════════════════════════════
# Parser
# ═════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--N",     type=int,   default=384,   help="number of basis functions")
parser.add_argument("--Q",     type=int,   default=16384, help="number of Gauss–Legendre points")
parser.add_argument("--tol",   type=float, default=2e-13, help="angular quadrature tolerance (abs = rel)")
parser.add_argument("--limit", type=int,   default=2000,  help="max subintervals of the angular quadrature")
args = parser.parse_args()

N     = args.N
Q     = args.Q
tol   = args.tol
limit = args.limit

os.makedirs("result/", exist_ok=True)
suffix = f"N={N}_Q={Q}_tol={tol:g}"

# ═════════════════════════════════════════════════════════════════════════
# Numerical parameters
# ═════════════════════════════════════════════════════════════════════════
π      = math.pi
s_tay  = 1e-2
xtol   = 5e-15
rtol   = 4 * np.finfo(float).eps
FACT   = [math.factorial(n) for n in range(9)]

# ═════════════════════════════════════════════════════════════════════════
# Functions
# ═════════════════════════════════════════════════════════════════════════
def G(s):
    """s + e^{-s} - 1, Taylor series up to s^8 for small s."""
    if s < s_tay:
        return sum((-1)**n * s**n / FACT[n] for n in range(2, 9))
    return s + math.exp(-s) - 1.0

def S_root(h, θ):
    """s = -log r on the closed streamline H = h:  G(s) - 2 sin²(θ/2) e^{-s} - (h-1) = 0."""
    a = 2.0 * math.sin(0.5 * θ)**2
    f = lambda s: G(s) - a * math.exp(-s) - (h - 1.0)
    return brentq(f, 0.0, h + 1.0, xtol=xtol, rtol=rtol, maxiter=500)

def σ(z):
    """σ(z) = S(h)/z = ∫_0^π r²/z dθ,  θ = π t²,  r²/z = exp[-2 + 2(h - s)]."""
    h = 1.0 - 0.5 * math.log(z)
    def integrand(t):
        θ = π * t * t
        return math.exp(-2.0 + 2.0 * (h - S_root(h, θ))) * 2.0 * π * t
    val, _ = quad(integrand, 0.0, 1.0, epsabs=tol, epsrel=tol, limit=limit)
    return val

def Basis(z, N):
    """b_j(z) = (1-z) P_j^{(1,0)}(2z-1) / sqrt(4π(j+1)),  j = 0..N-1.  Shape (Q, N)."""
    j = np.arange(N)
    P = eval_jacobi(j[None, :], 1, 0, (2 * z - 1)[:, None])
    return (1 - z)[:, None] * P / np.sqrt(4 * π * (j + 1))[None, :]

# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════
xg, wg = roots_legendre(Q)
z, w = 0.5 * (xg + 1), 0.5 * wg

wσ = w * np.array([σ(zq) for zq in z])
B  = Basis(z, N)
M  = B.T @ (wσ[:, None] * B)                     # stiffness matrix is the identity
μ  = eigh(M, subset_by_index=[N - 1, N - 1], eigvals_only=True)[0]
κ  = 1.0 / μ

with open(f"result/RED_KAPPA_{suffix}.pkl", "wb") as f: pickle.dump(κ, f)