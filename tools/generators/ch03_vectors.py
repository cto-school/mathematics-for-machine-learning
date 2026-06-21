"""Generator for Chapter 03 — Linear Algebra I: Vectors.

Run from anywhere:  python tools/generators/ch03_vectors.py
Produces one notebook in 03-linear-algebra-1-vectors/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "03-linear-algebra-1-vectors")


# ---------------------------------------------------------------------------
# Notebook 03 — Vectors
# ---------------------------------------------------------------------------
c = []

c.append(md(r"""
# Chapter 03 — Linear Algebra I: Vectors

A **vector** is the basic object of linear algebra, and linear algebra is the
language machine learning is written in. A data point (the pixels of an image,
the features of a house, the word counts of a document) is a vector. A model's
weights are a vector. Comparing, combining, and measuring vectors is most of
what an ML algorithm does.

In this notebook we treat a vector two ways at once:

- **algebraically** — a list of numbers, stored as a 1D NumPy array;
- **geometrically** — an arrow in space, which lets us *see* what the algebra
  means.

Run each code cell with **Shift + Enter**. Change the numbers and re-run — you
cannot break anything.
"""))

c.append(md(r"""
## 1. A vector is a 1D NumPy array

On paper we write a vector as a column (or row) of numbers:

$$\mathbf{a} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \qquad
\mathbf{b} = (1,\, 2,\, 2).$$

In NumPy a vector is just a **1D array**. The number of entries is its
*dimension*.
"""))
c.append(code("""
import numpy as np

a = np.array([3, 4])         # a vector in 2D
b = np.array([1, 2, 2])      # a vector in 3D

print("a       =", a)
print("b       =", b)
print("a shape =", a.shape)  # (2,)  -> one axis, two entries
print("a dim   =", a.ndim)   # 1     -> it is one-dimensional
print("len(b)  =", len(b))   # 3     -> b lives in 3D
"""))

c.append(md(r"""
## 2. Addition and scalar multiplication

These are the *only two* operations that define a vector space, and NumPy does
them entry by entry.

**Addition** lines up corresponding entries:
$$\mathbf{a} + \mathbf{b}
= \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}.$$

**Scalar multiplication** stretches (or flips) a vector:
$$s\,\mathbf{a} = \begin{bmatrix} s\,a_1 \\ s\,a_2 \end{bmatrix}.$$
"""))
c.append(code("""
import numpy as np

a = np.array([2.0, 1.0])
b = np.array([1.0, 3.0])

print("a + b      =", a + b)     # [3., 4.]  -> entrywise sum
print("a - b      =", a - b)     # [1., -2.]
print("2 * a      =", 2 * a)     # [4., 2.]  -> scaled by 2
print("-1 * b     =", -1 * b)    # [-1., -3.] -> flipped direction
print("0.5*a + 3*b =", 0.5 * a + 3 * b)  # a 'linear combination'
"""))

c.append(md(r"""
### Seeing the geometry: the parallelogram rule

An arrow from the origin to the point $(a_1, a_2)$ *is* the vector
$\mathbf{a}$. Adding $\mathbf{b}$ means placing $\mathbf{b}$'s tail at
$\mathbf{a}$'s head. We draw arrows with Matplotlib's `quiver`.
"""))
c.append(code("""
import numpy as np
import matplotlib.pyplot as plt

a = np.array([2.0, 1.0])
b = np.array([1.0, 3.0])
s = a + b

# Helper: draw one arrow from a starting point 'tail' along 'vec'.
def arrow(tail, vec, color, label):
    plt.quiver(tail[0], tail[1], vec[0], vec[1],
               angles="xy", scale_units="xy", scale=1,
               color=color, label=label)

plt.figure(figsize=(5, 5))
arrow([0, 0], a, "tab:blue",  "a")
arrow([0, 0], b, "tab:green", "b")
arrow(a, b, "tab:green", None)         # b shifted to a's head (dashed idea)
arrow([0, 0], s, "tab:red",   "a + b") # the diagonal is the sum

plt.xlim(-1, 5); plt.ylim(-1, 5)
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.grid(True); plt.gca().set_aspect("equal")
plt.legend(); plt.title("Vector addition: the parallelogram rule")
plt.show()
"""))

c.append(md(r"""
## 3. The norm: how long is a vector?

The **length** (or **norm**) of a vector measures its magnitude. The usual one
is the **Euclidean** or **$L_2$** norm — just the Pythagorean theorem:

$$\|\mathbf{a}\|_2 = \sqrt{a_1^2 + a_2^2 + \cdots + a_n^2}.$$

There is also the **$L_1$** norm (sum of absolute values), which shows up often
in ML for encouraging sparsity:

$$\|\mathbf{a}\|_1 = |a_1| + |a_2| + \cdots + |a_n|.$$
"""))
c.append(code("""
import numpy as np

a = np.array([3.0, 4.0])

# L2 norm three equivalent ways:
print("np.linalg.norm   :", np.linalg.norm(a))         # 5.0
print("by hand (sqrt)   :", np.sqrt(np.sum(a**2)))     # 5.0
print("sqrt of a . a    :", np.sqrt(a @ a))            # 5.0  (dot with itself)

# L1 norm: pass ord=1
print("L1 norm          :", np.linalg.norm(a, ord=1))  # |3| + |4| = 7.0
print("L1 by hand       :", np.sum(np.abs(a)))         # 7.0
"""))

c.append(md(r"""
## 4. Unit vectors: keeping the direction, dropping the length

A **unit vector** has length 1. To get the unit vector pointing the same way as
$\mathbf{a}$, divide by its norm — this is called **normalizing**:

$$\hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|}.$$

Unit vectors are how we talk about *direction* alone, independent of magnitude.
"""))
c.append(code("""
import numpy as np

a = np.array([3.0, 4.0])
a_hat = a / np.linalg.norm(a)   # normalize

print("unit vector  :", a_hat)              # [0.6, 0.8]
print("its length   :", np.linalg.norm(a_hat))  # 1.0 (up to rounding)
"""))

c.append(md(r"""
## 5. The dot product

The **dot product** of two vectors multiplies matching entries and adds them up,
returning a single number (a *scalar*):

$$\mathbf{a}\cdot\mathbf{b}
= a_1 b_1 + a_2 b_2 + \cdots + a_n b_n.$$

In NumPy use `np.dot(a, b)` or the `@` operator. This one little number is the
workhorse of ML: every neuron computes a dot product of inputs and weights.
"""))
c.append(code("""
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print("np.dot(a, b) :", np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32
print("a @ b        :", a @ b)          # 32  (same thing)
print("by hand      :", np.sum(a * b))  # 32  (entrywise product, summed)
"""))

c.append(md(r"""
### Geometric meaning of the dot product

The dot product secretly encodes the **angle** between two vectors:

$$\mathbf{a}\cdot\mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\,\cos\theta.$$

So the sign of $\mathbf{a}\cdot\mathbf{b}$ tells you whether the vectors point in
*roughly the same* direction (positive), *roughly opposite* (negative), or are
*perpendicular* (zero).
"""))

c.append(md(r"""
## 6. Angle between vectors

Rearranging the formula above gives the angle directly:

$$\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}.$$

We compute $\cos\theta$, then `np.arccos` recovers $\theta$ (in radians;
multiply by $180/\pi$ for degrees).
"""))
c.append(code("""
import numpy as np

a = np.array([1.0, 0.0])
b = np.array([1.0, 1.0])

cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)    # convert radians -> degrees

print("cos(theta)     :", cos_theta)   # 0.7071...
print("theta (radians):", theta_rad)   # 0.7853... = pi/4
print("theta (degrees):", theta_deg)   # 45.0
"""))

c.append(md(r"""
## 7. Cosine similarity

In ML we often care only about *direction*, not length — for example, two
documents are "similar" if their word-count vectors point the same way, even if
one document is longer. The **cosine similarity** is exactly $\cos\theta$:

$$\text{cossim}(\mathbf{a}, \mathbf{b})
= \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}.$$

It ranges from $+1$ (identical direction) through $0$ (perpendicular) to $-1$
(opposite direction).
"""))
c.append(code("""
import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)   # accept plain lists too
    b = np.asarray(b, dtype=float)
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

u = np.array([2.0, 1.0])
print("same direction  :", cosine_similarity(u, 3 * u))   #  1.0
print("perpendicular   :", cosine_similarity([1, 0], [0, 1]))  # 0.0
print("opposite        :", cosine_similarity(u, -u))      # -1.0
print("somewhat similar:", cosine_similarity([1, 1], [1, 0]))  # 0.707
"""))

c.append(md(r"""
## 8. Orthogonality

Two vectors are **orthogonal** (perpendicular) exactly when their dot product is
zero, because $\cos 90^\circ = 0$:

$$\mathbf{a}\cdot\mathbf{b} = 0 \quad\Longleftrightarrow\quad
\mathbf{a}\perp\mathbf{b}.$$

Orthogonality means the vectors share no common direction — a cornerstone idea
behind coordinate axes, PCA, and least squares.
"""))
c.append(code("""
import numpy as np

a = np.array([1.0, 0.0])
b = np.array([0.0, 1.0])
c_ = np.array([1.0, 1.0])

# Floating-point dot products are rarely *exactly* 0, so test with a tolerance.
def is_orthogonal(x, y, tol=1e-9):
    return abs(np.dot(x, y)) < tol

print("a . b =", np.dot(a, b), "-> orthogonal?", is_orthogonal(a, b))   # 0 -> True
print("a . c =", np.dot(a, c_), "-> orthogonal?", is_orthogonal(a, c_)) # 1 -> False
"""))

c.append(md(r"""
## 9. Projection of one vector onto another

The **projection** of $\mathbf{a}$ onto $\mathbf{b}$ is the "shadow" of
$\mathbf{a}$ along the direction of $\mathbf{b}$ — the part of $\mathbf{a}$ that
lies in $\mathbf{b}$'s direction:

$$\text{proj}_{\mathbf{b}}\,\mathbf{a}
= \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\;\mathbf{b}.$$

The leftover piece, $\mathbf{a} - \text{proj}_{\mathbf{b}}\,\mathbf{a}$, is
orthogonal to $\mathbf{b}$. Projection is the engine behind least-squares fitting.
"""))
c.append(code("""
import numpy as np

a = np.array([3.0, 3.0])
b = np.array([4.0, 0.0])

proj = (a @ b) / (b @ b) * b      # projection of a onto b
resid = a - proj                  # the orthogonal leftover

print("proj_b(a)        :", proj)            # [3., 0.]
print("residual a - proj:", resid)           # [0., 3.]
print("residual . b     :", resid @ b)       # 0.0  -> confirms orthogonality
"""))
c.append(code("""
import numpy as np
import matplotlib.pyplot as plt

a = np.array([3.0, 3.0])
b = np.array([4.0, 0.0])
proj = (a @ b) / (b @ b) * b

plt.figure(figsize=(5, 5))
plt.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
           color="tab:blue", label="a")
plt.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
           color="tab:green", label="b")
plt.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1,
           color="tab:red", label="proj_b(a)")
# dashed line from a down to its shadow shows the right angle
plt.plot([a[0], proj[0]], [a[1], proj[1]], "k--", lw=1)

plt.xlim(-1, 5); plt.ylim(-1, 5)
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.grid(True); plt.gca().set_aspect("equal")
plt.legend(); plt.title("Projection of a onto b")
plt.show()
"""))

c.append(md(r"""
## 10. Linear combinations and span

A **linear combination** of vectors mixes them with scalar weights:

$$s\,\mathbf{v}_1 + t\,\mathbf{v}_2.$$

The **span** of a set of vectors is *all* the points you can reach this way as
the weights vary over every real number. Two non-parallel vectors in 2D span the
*whole plane*; a single vector spans only the line through it. Below we scatter
many random combinations of two vectors to *see* their span filling the plane.
"""))
c.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)     # reproducible random numbers

v1 = np.array([2.0, 1.0])
v2 = np.array([-1.0, 1.0])

# pick 400 random weight pairs (s, t) in [-2, 2]
weights = rng.uniform(-2, 2, size=(400, 2))
points = weights[:, [0]] * v1 + weights[:, [1]] * v2   # each row: s*v1 + t*v2

plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.4, color="tab:purple")
plt.quiver(0, 0, v1[0], v1[1], angles="xy", scale_units="xy", scale=1,
           color="tab:blue", label="v1")
plt.quiver(0, 0, v2[0], v2[1], angles="xy", scale_units="xy", scale=1,
           color="tab:green", label="v2")
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Random linear combinations of v1, v2 fill the plane")
plt.show()
"""))

c.append(md(r"""
## 11. Distance between two points

The distance between points $\mathbf{a}$ and $\mathbf{b}$ is the length of the
vector joining them:

$$d(\mathbf{a},\mathbf{b}) = \|\mathbf{a}-\mathbf{b}\|_2
 = \sqrt{\textstyle\sum_i (a_i-b_i)^2}.$$

This Euclidean distance is the backbone of k-nearest-neighbours, clustering, and
many similarity measures in ML.
"""))
c.append(code("""
import numpy as np

a = np.array([1.0, 2.0])
b = np.array([4.0, 6.0])

dist = np.linalg.norm(a - b)            # length of the difference vector
print("distance a to b :", dist)        # sqrt(9 + 16) = 5.0
print("by hand          :", np.sqrt(np.sum((a - b)**2)))
"""))

c.append(md(r"""
## 12. From one vector to a dataset: row-wise operations

Real data is a *stack* of vectors: a matrix $X$ whose every **row** is one data
point. Broadcasting (Ch. 01) lets us measure all rows at once — the key is the
`axis` argument, where `axis=1` works **across each row**.
"""))
c.append(code("""
import numpy as np

# 4 data points in 2-D, one per row
X = np.array([[0.0, 0.0],
              [3.0, 4.0],
              [1.0, 1.0],
              [-2.0, 2.0]])

# Length (norm) of every row at once:
row_norms = np.linalg.norm(X, axis=1)       # one number per row
print("row norms:", row_norms)              # [0. 5. 1.414 2.828]

# Distance from every point to a query point q (broadcasting + axis):
q = np.array([1.0, 0.0])
dists = np.linalg.norm(X - q, axis=1)       # subtract q from each row, then norm
print("distances to q :", dists)
print("nearest row    :", np.argmin(dists)) # index of the closest point (a kNN step!)
"""))

# ---- Exercise 1 (with solution) ----
c.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Let $\mathbf{a} = (1, 2, 2)$. Compute its $L_2$ norm, then build the unit vector
$\hat{\mathbf{a}}$ and verify its length is 1.
"""))
c.append(md("**Solution:**"))
c.append(code("""
import numpy as np

a = np.array([1.0, 2.0, 2.0])
norm = np.linalg.norm(a)          # sqrt(1 + 4 + 4) = 3.0
a_hat = a / norm

print("||a||        :", norm)             # 3.0
print("unit vector  :", a_hat)            # [0.333, 0.667, 0.667]
print("its length   :", np.linalg.norm(a_hat))  # 1.0
"""))

# ---- Exercise 2 (with solution) ----
c.append(md(r"""
## ✍️ Exercise 2 (solution included)

Find the angle (in degrees) between $\mathbf{a} = (1, 2)$ and
$\mathbf{b} = (2, 1)$. Are they orthogonal?
"""))
c.append(md("**Solution:**"))
c.append(code("""
import numpy as np

a = np.array([1.0, 2.0])
b = np.array([2.0, 1.0])

cos_theta = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
print("dot product :", a @ b)                 # 4 (not zero -> NOT orthogonal)
print("cos(theta)  :", cos_theta)             # 0.8
print("angle (deg) :", np.degrees(np.arccos(cos_theta)))  # 36.87 degrees
"""))

# ---- Exercise 3 (with solution) ----
c.append(md(r"""
## ✍️ Exercise 3 (solution included)

Project $\mathbf{a} = (2, 3)$ onto $\mathbf{b} = (1, 0)$. What do you expect
geometrically (think about what $(1,0)$ points along), and does the residual come
out orthogonal to $\mathbf{b}$?
"""))
c.append(md("**Solution:**"))
c.append(code("""
import numpy as np

a = np.array([2.0, 3.0])
b = np.array([1.0, 0.0])

proj = (a @ b) / (b @ b) * b      # since b is the x-axis, this grabs a's x-part
resid = a - proj

print("proj_b(a)   :", proj)        # [2., 0.]  -> just the x-component of a
print("residual    :", resid)       # [0., 3.]  -> the y-component
print("resid . b   :", resid @ b)   # 0.0 -> orthogonal, as expected
"""))

# ---- Homework (no solutions) ----
c.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. For $\mathbf{u} = (4, -3)$ and $\mathbf{v} = (1, 2)$, compute
   $\mathbf{u} + \mathbf{v}$, $2\mathbf{u} - \mathbf{v}$, $\|\mathbf{u}\|_2$,
   and $\|\mathbf{u}\|_1$ in NumPy.
2. Write a function `cosine_similarity(a, b)` and use it to decide which of
   $\mathbf{p} = (3, 4)$ and $\mathbf{q} = (-3, 4)$ is more similar in direction
   to $\mathbf{r} = (1, 1)$.
3. Find a non-zero vector orthogonal to $\mathbf{w} = (2, 5)$ in 2D, and confirm
   with a dot product. *(Hint: swap the entries and flip one sign.)*
4. Project $\mathbf{a} = (1, 2, 2)$ onto $\mathbf{b} = (0, 0, 1)$, then verify
   that `a` equals `proj + residual` and that the residual is orthogonal to
   $\mathbf{b}$.
5. Given the data matrix `X = np.array([[1, 0], [4, 5], [2, 1], [0, 3]])` and a
   query point `q = (2, 2)`, compute the Euclidean distance from `q` to every
   row of `X` in one expression, and report which point is nearest.
"""))

save(os.path.join(CH, "03_vectors.ipynb"), c)
