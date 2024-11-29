import numpy as np
import math
import jax.numpy as jnp


def _alpha_recursion(j, k, n):
    if k == 0:
        if j < n - 1:
            return 0.0
        else:
            return 1.0

    return _alpha_recursion(j, k - 1, n) + (
        (-1) ** (n + k - j - 1)
        * math.comb(n, k)
        * math.comb(n - 1, j)
        * k ** (n - j - 1)
    )


class CardinalSplineKernel:
    """
    Creates Irwin-Hall function for cardinal basis.

    Parameters
    ----------
    n : `int`
        order of the basis function.

    Atttributes
    -----------
    alphas : `np.ndarray`
        piecewise polynomial coefficients matrix (n+1,n+1)
    """

    def __init__(self, n):
        self.n = n
        self.alphas = np.zeros((n + 1, n + 1))
        for j in range(n + 1):
            for k in range(n + 1):
                self.alphas[j, k] = _alpha_recursion(j, k, n + 1)
        self.alphas = jnp.array(self.alphas)

    def __call__(self, x, *args):
        ks = jnp.floor(x + ((self.n + 1) / 2)).astype(int)
        cond1 = ks >= 0
        cond2 = ks <= (self.n)

        f = jnp.where(
            (cond1 * cond2),
            jnp.polyval(self.alphas[::-1, ks], x + ((self.n + 1) / 2)),
            0.0,
        )

        return f
