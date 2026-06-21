"""Generator for Chapter 05 — Linear Algebra III: Eigenvalues, SVD & PCA.

Run from anywhere:  python tools/generators/ch05_eigen_svd_pca.py
Produces two notebooks in 05-linear-algebra-3-eigen-svd-pca/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "05-linear-algebra-3-eigen-svd-pca")


# ---------------------------------------------------------------------------
# Notebook 05a — Eigenvalues & eigenvectors
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 05a — Eigenvalues & Eigenvectors

A square matrix $A$ acts on a vector by **transforming** it — rotating,
stretching, shearing. For *most* directions the output points somewhere new.
But some special directions are only **stretched** (or shrunk, or flipped),
never rotated. Those directions are the **eigenvectors**, and the stretch
factor is the **eigenvalue**.

$$A\,v = \lambda\, v, \qquad v \neq 0.$$

This single equation runs through the rest of machine learning: it is the heart
of diagonalization, the SVD, and PCA. Run each cell with **Shift + Enter**.
"""))

a.append(md(r"""
## 1. Setup

We use NumPy for the linear algebra and Matplotlib for pictures. A fixed random
generator (`default_rng(0)`) keeps results reproducible.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible randomness
np.set_printoptions(precision=4, suppress=True)  # tidy number printing
"""))

a.append(md(r"""
## 2. The defining equation $A v = \lambda v$

Let's take a concrete matrix and a vector that happens to be an eigenvector,
and check the equation by hand.

$$A = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}.$$

A diagonal matrix simply scales the $x$-axis by 2 and the $y$-axis by 3, so the
axis directions are obvious eigenvectors.
"""))
a.append(code("""
A = np.array([[2.0, 0.0],
              [0.0, 3.0]])

v = np.array([1.0, 0.0])   # the x-axis direction
print("A @ v =", A @ v)    # -> [2, 0] = 2 * v, so lambda = 2
print("2 * v =", 2 * v)
"""))

a.append(md(r"""
The output of `A @ v` is exactly `2 * v`: the vector kept its direction and was
scaled by $\lambda = 2$. That is what "eigenvector" means.
"""))

a.append(md(r"""
## 3. Computing eigenvalues with `np.linalg.eig`

For anything bigger than a diagonal matrix we let NumPy do the work.
`np.linalg.eig(A)` returns

- `vals` — a 1-D array of eigenvalues $\lambda_i$,
- `vecs` — a matrix whose **columns** are the matching eigenvectors (each
  normalized to length 1).
"""))
a.append(code("""
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

vals, vecs = np.linalg.eig(A)
print("eigenvalues :", vals)
print("eigenvectors (columns):")
print(vecs)
"""))

a.append(md(r"""
**Reading the output:** eigenvalue `vals[i]` goes with the eigenvector in
column `vecs[:, i]`. Let's verify $A v = \lambda v$ for the first pair.
"""))
a.append(code("""
lam = vals[0]
v = vecs[:, 0]            # the i-th eigenvector is a COLUMN

print("A @ v     =", A @ v)
print("lam * v   =", lam * v)
print("close?    ", np.allclose(A @ v, lam * v))   # True
"""))

a.append(md(r"""
## 4. Seeing it: eigen vs non-eigen directions

A picture makes the idea click. We draw a vector (blue) and its image $A v$
(red). For an **eigenvector** the red arrow lies on the same line as the blue
one — only the length changes. For a generic direction the red arrow swings off
to the side.
"""))
a.append(code("""
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eig(A)

# Pick an eigenvector and a deliberately non-eigen direction.
eig_dir = vecs[:, 0]                    # an eigenvector
non_dir = np.array([1.0, 0.0])          # the x-axis: NOT an eigenvector here

def draw(ax, v, title):
    Av = A @ v
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
              color='tab:blue', label='v')
    ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
              color='tab:red', label='A v')
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4)
    ax.set_aspect('equal'); ax.grid(True)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_title(title); ax.legend(loc='upper left')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
draw(ax1, eig_dir, "Eigenvector: A v stays on the line")
draw(ax2, non_dir, "Non-eigenvector: A v rotates away")
plt.tight_layout()
plt.show()
"""))

a.append(md(r"""
On the left the red arrow is a stretched copy of the blue one — same line. On
the right the red arrow points in a new direction: the $x$-axis is not an
eigenvector of this $A$.
"""))

a.append(md(r"""
## 5. Diagonalization $A = P D P^{-1}$

Collect the eigenvectors as the columns of $P$ and the eigenvalues on the
diagonal of $D$. Then

$$A = P\,D\,P^{-1}.$$

This says: to apply $A$, first change coordinates into the eigenvector basis
($P^{-1}$), scale each axis by its eigenvalue ($D$), then change back ($P$).
Powers become easy too: $A^k = P\,D^k\,P^{-1}$.
"""))
a.append(code("""
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eig(A)

P = vecs                 # columns are eigenvectors
D = np.diag(vals)        # eigenvalues on the diagonal
Pinv = np.linalg.inv(P)

reconstructed = P @ D @ Pinv
print("P D P^-1 =")
print(reconstructed)
print("matches A?", np.allclose(reconstructed, A))   # True
"""))

a.append(code("""
# Bonus: A^3 the easy way, via D^3
A3_fast = P @ np.diag(vals**3) @ Pinv
A3_slow = A @ A @ A
print("same A^3?", np.allclose(A3_fast, A3_slow))     # True
"""))

a.append(md(r"""
## 6. Symmetric matrices are special: use `np.linalg.eigh`

A **symmetric** matrix ($A = A^{\top}$) always has

- **real** eigenvalues, and
- eigenvectors that are mutually **orthogonal** (perpendicular).

For these, prefer `np.linalg.eigh` ("h" for Hermitian/symmetric): it is faster,
more accurate, and returns the eigenvalues in ascending order.
"""))
a.append(code("""
# Build a guaranteed-symmetric matrix: M + M^T is always symmetric.
M = rng.standard_normal((3, 3))
A = M + M.T
print("symmetric?", np.allclose(A, A.T))   # True

vals, vecs = np.linalg.eigh(A)
print("eigenvalues (real, ascending):", vals)
"""))
a.append(code("""
# Orthogonal eigenvectors: V^T V should be the identity matrix.
print("V^T V =")
print(vecs.T @ vecs)        # ~ identity -> columns are orthonormal
"""))

a.append(md(r"""
Because the eigenvectors are orthonormal, $P^{-1} = P^{\top}$, and
diagonalization simplifies to $A = V\,D\,V^{\top}$.
"""))
a.append(code("""
V, D = vecs, np.diag(vals)
print("V D V^T matches A?", np.allclose(V @ D @ V.T, A))   # True
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

For the matrix
$$A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix},$$
compute its eigenvalues and eigenvectors with `np.linalg.eig`, then verify
numerically that $A v = \lambda v$ for **each** eigenpair.
"""))
a.append(md("**Solution:**"))
a.append(code("""
A = np.array([[4.0, 1.0],
              [2.0, 3.0]])
vals, vecs = np.linalg.eig(A)

for i in range(len(vals)):
    lam = vals[i]
    v = vecs[:, i]
    print(f"lambda = {lam:.4f},  A v = {A @ v},  lam v = {lam * v}")
    print("   match?", np.allclose(A @ v, lam * v))
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

The **trace** (sum of the diagonal) of a matrix equals the **sum** of its
eigenvalues, and the **determinant** equals their **product**. Verify both
facts for a random $4\times 4$ matrix.
"""))
a.append(md("**Solution:**"))
a.append(code("""
A = rng.standard_normal((4, 4))
vals, _ = np.linalg.eig(A)

print("trace        :", np.trace(A))
print("sum of eigs  :", np.sum(vals).real)     # .real drops tiny imaginary dust
print()
print("determinant  :", np.linalg.det(A))
print("prod of eigs :", np.prod(vals).real)
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Take $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ (a 90° rotation).
   Compute its eigenvalues with `np.linalg.eig`. Why are they **complex**, and
   what does that say about real eigenvector directions for a pure rotation?
2. Build a random symmetric $5\times 5$ matrix. Use `np.linalg.eigh` and confirm
   that every eigenvalue is real and that the eigenvectors are orthonormal
   (check `V.T @ V`).
3. For the symmetric matrix in (2), reconstruct it as $V D V^{\top}$ and report
   the maximum absolute reconstruction error.
4. Write a function `power_iteration(A, steps=100)` that starts from a random
   vector, repeatedly applies `A` and renormalizes, and returns the dominant
   eigenvector. Compare its result to the top eigenvector from `np.linalg.eig`.
"""))

save(os.path.join(CH, "05a_eigenvalues.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 05b — SVD, low-rank approximation & PCA
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 05b — SVD, Low-Rank Approximation & PCA

Eigenvalues only exist for square matrices. The **Singular Value Decomposition
(SVD)** generalizes the idea to *any* matrix and is arguably the most useful
factorization in all of applied mathematics:

$$A = U\,\Sigma\,V^{\top}.$$

- $U$ and $V$ have **orthonormal columns** (rotations/reflections),
- $\Sigma$ is diagonal with non-negative **singular values**
  $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ (pure stretches).

Every linear map is therefore *rotate → stretch → rotate*. In this notebook we
use the SVD for **low-rank approximation** (compression) and then build
**PCA** from scratch.
"""))

b.append(md(r"""
## 1. Setup
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)
"""))

b.append(md(r"""
## 2. Computing the SVD with `np.linalg.svd`

By default NumPy returns the "full" factors. We usually want the compact form,
so pass `full_matrices=False`: then for an $m\times n$ matrix the shapes are
$U:(m,k)$, $S:(k,)$, $V^{\top}:(k,n)$ with $k=\min(m,n)$.

Note that NumPy returns $V^{\top}$ (already transposed), not $V$.
"""))
b.append(code("""
A = np.array([[3.0, 1.0, 1.0],
              [1.0, 3.0, 1.0]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
print("U shape :", U.shape)
print("S (singular values):", S)   # always >= 0, descending
print("Vt shape:", Vt.shape)
"""))

b.append(md(r"""
Reconstruct $A$ from the pieces. Since $S$ is returned as a 1-D array, we turn it
into a diagonal matrix with `np.diag` before multiplying.
"""))
b.append(code("""
A_rebuilt = U @ np.diag(S) @ Vt
print("reconstruction matches A?", np.allclose(A_rebuilt, A))   # True
"""))

b.append(md(r"""
## 3. Low-rank approximation: keep the biggest singular values

The SVD writes $A$ as a sum of rank-1 layers, each weighted by a singular value:

$$A = \sum_{i=1}^{k} \sigma_i\, u_i\, v_i^{\top}.$$

Because the $\sigma_i$ are sorted largest-first, the **top few** terms already
capture most of the matrix. Keeping only the first $r$ terms gives the *best*
rank-$r$ approximation (this is the Eckart–Young theorem).

Let's build a small grayscale "image" with structure and watch the approximation
sharpen as we add singular values.
"""))
b.append(code("""
# A 20x20 "image": a smooth ramp plus a couple of bright stripes.
n = 20
x = np.linspace(0, 1, n)
img = np.outer(x, x)              # smooth gradient (rank 1 on its own)
img[5, :] += 0.8                  # a bright horizontal stripe
img[:, 12] += 0.6                 # a bright vertical stripe

U, S, Vt = np.linalg.svd(img, full_matrices=False)
print("number of singular values:", len(S))
print("first few singular values:", S[:6])
"""))
b.append(code("""
def low_rank(U, S, Vt, r):
    \"\"\"Reconstruct using only the top r singular values.\"\"\"
    return U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

ranks = [1, 2, 4, len(S)]
fig, axes = plt.subplots(1, len(ranks), figsize=(12, 3.2))
for ax, r in zip(axes, ranks):
    approx = low_rank(U, S, Vt, r)
    ax.imshow(approx, cmap='gray')
    ax.set_title(f"rank {r}")
    ax.axis('off')
plt.tight_layout()
plt.show()
"""))

b.append(md(r"""
The rank-1 picture is blurry; by rank 4 it is already very close to the full
image. Let's quantify the shrinking error.
"""))
b.append(code("""
for r in range(1, len(S) + 1):
    approx = low_rank(U, S, Vt, r)
    err = np.linalg.norm(img - approx)     # Frobenius (overall) error
    if r <= 6 or r == len(S):
        print(f"rank {r:2d}:  reconstruction error = {err:.4f}")
"""))

b.append(md(r"""
The error drops fast and reaches (essentially) zero once $r$ equals the true
rank. That is compression: store a few singular values and vectors instead of
the whole matrix.
"""))

b.append(md(r"""
## 4. PCA from scratch — the data

**Principal Component Analysis** finds the directions along which a cloud of
points varies the most. We make a 2-D cloud that is clearly stretched along a
slanted direction.
"""))
b.append(code("""
# 200 points: independent spread, then stretched and rotated into a tilt.
N = 200
base = rng.standard_normal((N, 2)) * np.array([2.0, 0.5])   # wide, then thin
theta = np.deg2rad(35)                                       # rotate 35 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
data = base @ R.T + np.array([5.0, 3.0])                     # rotate + shift

plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=12, alpha=0.6)
plt.gca().set_aspect('equal'); plt.grid(True)
plt.title("Raw 2-D point cloud")
plt.show()
"""))

b.append(md(r"""
## 5. Step 1 — center the data

PCA is about *spread around the mean*, so we first subtract the mean of each
column. The centered cloud sits at the origin.
"""))
b.append(code("""
mean = data.mean(axis=0)            # average of each column (x and y)
X = data - mean                     # centered data
print("original mean:", mean)
print("centered mean:", X.mean(axis=0))   # ~ [0, 0]
"""))

b.append(md(r"""
## 6. Step 2 — covariance matrix and its eigenvectors

The **covariance matrix** $C = \frac{1}{N-1} X^{\top} X$ summarizes how the
coordinates vary together. Its **eigenvectors** are the principal axes and its
**eigenvalues** are the variance along each axis. Because $C$ is symmetric, we
use `np.linalg.eigh`.
"""))
b.append(code("""
C = (X.T @ X) / (len(X) - 1)        # 2x2 covariance matrix
print("covariance matrix:")
print(C)

vals, vecs = np.linalg.eigh(C)      # ascending order
# Reorder so the LARGEST variance comes first.
order = np.argsort(vals)[::-1]
vals = vals[order]
vecs = vecs[:, order]
print("variances (eigenvalues):", vals)
print("principal axes (columns):")
print(vecs)
"""))

b.append(md(r"""
The same axes come straight out of the SVD of the centered data — no covariance
matrix needed. This is why PCA and SVD are two views of one idea.
"""))
b.append(code("""
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# Rows of Vt are the principal axes; variances are S^2 / (N-1).
print("variances from SVD :", S**2 / (len(X) - 1))
print("axes from SVD (rows of Vt):")
print(Vt)
"""))

b.append(md(r"""
## 7. Step 3 — draw the principal axes over the data

Each principal axis is drawn from the mean, scaled by the standard deviation
($\sqrt{\text{variance}}$) so its length reflects how much the data spreads that
way.
"""))
b.append(code("""
plt.figure(figsize=(5, 5))
plt.scatter(data[:, 0], data[:, 1], s=12, alpha=0.5)

colors = ['tab:red', 'tab:green']
for i in range(2):
    axis = vecs[:, i] * np.sqrt(vals[i]) * 2.5   # scale for visibility
    plt.quiver(mean[0], mean[1], axis[0], axis[1],
               angles='xy', scale_units='xy', scale=1,
               color=colors[i], width=0.012,
               label=f"PC{i+1} (var={vals[i]:.2f})")

plt.gca().set_aspect('equal'); plt.grid(True)
plt.legend(loc='upper left')
plt.title("Principal axes over the data")
plt.show()
"""))

b.append(md(r"""
The longer red arrow (PC1) lies along the direction of greatest spread; the
shorter green arrow (PC2) is perpendicular to it.
"""))

b.append(md(r"""
## 8. Step 4 — project to 1-D and report variance explained

Projecting the centered data onto PC1 reduces it from 2 numbers to 1 while
keeping as much spread as possible. The **fraction of variance explained** by a
component is its eigenvalue divided by the total.
"""))
b.append(code("""
pc1 = vecs[:, 0]                 # top principal direction
proj1d = X @ pc1                 # one number per point: its position along PC1

total_var = vals.sum()
explained = vals / total_var
print(f"PC1 explains {explained[0]*100:.1f}% of the variance")
print(f"PC2 explains {explained[1]*100:.1f}% of the variance")
"""))
b.append(code("""
# Visualize the 1-D projection laid out on a line.
plt.figure(figsize=(7, 1.8))
plt.scatter(proj1d, np.zeros_like(proj1d), s=12, alpha=0.5)
plt.yticks([])
plt.xlabel("position along PC1")
plt.title(f"Data reduced to 1-D  ({explained[0]*100:.1f}% variance kept)")
plt.grid(True, axis='x')
plt.show()
"""))

b.append(md(r"""
Most of the spread survives in a single dimension — that is dimensionality
reduction in action, and it is exactly how PCA compresses high-dimensional data
in machine learning.
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

The squared **Frobenius norm** of a matrix equals the sum of the squares of its
singular values: $\|A\|_F^2 = \sum_i \sigma_i^2$. Verify this for a random
$5\times 4$ matrix.
"""))
b.append(md("**Solution:**"))
b.append(code("""
A = rng.standard_normal((5, 4))
S = np.linalg.svd(A, compute_uv=False)   # singular values only

lhs = np.linalg.norm(A)**2               # Frobenius norm, squared
rhs = np.sum(S**2)
print("||A||_F^2       :", lhs)
print("sum of sigma^2  :", rhs)
print("match?", np.allclose(lhs, rhs))
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Make a fresh 2-D cloud stretched mostly along one direction. Use the SVD route
(not the covariance matrix) to compute the principal axes, and report what
percentage of the variance the **first** component explains.
"""))
b.append(md("**Solution:**"))
b.append(code("""
# Strongly anisotropic cloud: big spread in x, tiny in y, then rotate.
cloud = rng.standard_normal((150, 2)) * np.array([3.0, 0.4])
ang = np.deg2rad(20)
Rot = np.array([[np.cos(ang), -np.sin(ang)],
                [np.sin(ang),  np.cos(ang)]])
cloud = cloud @ Rot.T

Xc = cloud - cloud.mean(axis=0)          # center
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
variances = S**2 / (len(Xc) - 1)
explained = variances / variances.sum()

print("variances :", variances)
print(f"PC1 explains {explained[0]*100:.1f}% of the variance")
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. For a random $6\times 6$ matrix, compute its SVD and plot the singular values
   `S` with `plt.plot`. How quickly do they decay?
2. Build a low-rank matrix on purpose: `A = np.outer(u, v) + np.outer(p, q)`
   for random vectors. Confirm via the SVD that it has rank 2 (only two
   singular values are non-zero, up to tiny numerical noise).
3. Take the 20×20 "image" from Section 3 and find the smallest rank $r$ whose
   reconstruction error is below `0.1`. How much storage does that save versus
   the full matrix?
4. Generate a 3-D point cloud that lies (noisily) near a single plane. Use PCA
   to find the two dominant directions and report the combined variance
   explained by the top two principal components.
"""))

save(os.path.join(CH, "05b_svd_pca.ipynb"), b)
