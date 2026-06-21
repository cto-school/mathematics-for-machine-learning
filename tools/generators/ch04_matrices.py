"""Generator for Chapter 04 — Linear Algebra II: Matrices & Linear Systems.

Run from anywhere:  python tools/generators/ch04_matrices.py
Produces two notebooks in 04-linear-algebra-2-matrices/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "04-linear-algebra-2-matrices")


# ---------------------------------------------------------------------------
# Notebook 04a — Matrices & linear transformations
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 04a — Matrices & Linear Transformations

A **matrix** is a rectangular grid of numbers. On paper we write

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}.$$

In NumPy a matrix is simply a **2-D array**. In this notebook we learn how to
build matrices, do arithmetic with them, multiply them with `@`, and — most
importantly — *see* what a matrix **does**: a matrix is a machine that
**transforms space** (it rotates, scales, and shears points).

Run each cell with **Shift + Enter**. Change numbers and re-run to build
intuition.
"""))

a.append(md(r"""
## 1. A matrix is a 2-D array

We give `np.array` a list of rows. The **shape** is `(rows, columns)`.
"""))
a.append(code("""
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])    # 2 rows, 3 columns

print(A)
print("shape:", A.shape)     # (2, 3)  ->  2 rows, 3 columns
print("rows :", A.shape[0])  # 2
print("cols :", A.shape[1])  # 3
"""))

a.append(md(r"""
We index with `A[row, col]` (remember: counting starts at **0**). A whole row
or column is grabbed with a colon `:`.
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A[0, 0])   # top-left entry      -> 1
print(A[1, 2])   # row 1, col 2        -> 6
print(A[0, :])   # the whole first row -> [1 2 3]
print(A[:, 1])   # the whole 2nd column-> [2 5]
"""))

a.append(md(r"""
## 2. The transpose $A^\top$

The **transpose** flips a matrix across its diagonal: rows become columns. If
$A$ is $2\times 3$ then $A^\top$ is $3\times 2$, and $(A^\top)_{ij}=A_{ji}$.
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("A =\\n", A)
print("A.T =\\n", A.T)        # .T is the transpose
print("shape of A.T:", A.T.shape)   # (3, 2)
"""))

a.append(md(r"""
## 3. Addition and scalar multiplication

Matrices of the **same shape** add entry by entry. Multiplying by a single
number (a *scalar*) scales every entry. These work exactly like vectors.

$$ (A+B)_{ij} = A_{ij}+B_{ij}, \qquad (cA)_{ij} = c\,A_{ij}. $$
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[10, 20],
              [30, 40]])

print("A + B =\\n", A + B)    # entrywise sum
print("3 * A =\\n", 3 * A)    # scale every entry by 3
print("A - B =\\n", A - B)    # entrywise difference
"""))

a.append(md(r"""
> **Careful:** `A * B` in NumPy is the *entrywise* product, **not** matrix
> multiplication. For real matrix multiplication use `@` (next section).
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[10, 20],
              [30, 40]])

print("A * B (entrywise!) =\\n", A * B)   # NOT the matrix product
"""))

a.append(md(r"""
## 4. Matrix multiplication with `@`

The matrix product $C = AB$ is defined by

$$ C_{ij} = \sum_k A_{ik}\,B_{kj} \quad\text{(row $i$ of $A$ dotted with column $j$ of $B$).} $$

**The dimensions must match:** if $A$ is $m\times n$, then $B$ must be
$n\times p$ (inner numbers equal), and the result is $m\times p$.

$$ \underbrace{(m\times n)}_{A}\;\underbrace{(n\times p)}_{B}=\underbrace{(m\times p)}_{C}. $$
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2],
              [3, 4]])       # 2 x 2
B = np.array([[5, 6],
              [7, 8]])       # 2 x 2

print("A @ B =\\n", A @ B)   # the @ operator IS matrix multiplication
# Check one entry by hand: top-left = 1*5 + 2*7 = 19
print("top-left by hand:", 1*5 + 2*7)
"""))
a.append(code("""
import numpy as np
# Non-square example: (2x3) @ (3x2) -> (2x2)
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # 2 x 3
B = np.array([[1, 0],
              [0, 1],
              [1, 1]])       # 3 x 2

C = A @ B
print("C = A @ B =\\n", C)
print("shape:", C.shape)     # (2, 2)
"""))

a.append(md(r"""
If the inner dimensions **don't** match, NumPy raises an error. This is a good
thing — it catches mistakes. Run this to see the error message:
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # 2 x 3
B = np.array([[1, 2],
              [3, 4]])       # 2 x 2  -- inner dims 3 and 2 disagree

try:
    A @ B
except ValueError as e:
    print("ValueError:", e)
"""))

a.append(md(r"""
**Order matters!** In general $AB \neq BA$. Matrix multiplication is *not*
commutative.
"""))
a.append(code("""
import numpy as np
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[0, 1],
              [1, 0]])       # this B swaps columns / rows

print("A @ B =\\n", A @ B)
print("B @ A =\\n", B @ A)   # different!
"""))

a.append(md(r"""
## 5. The identity matrix $I$

The identity $I$ has 1's on the diagonal and 0's elsewhere. It is the
"do-nothing" matrix: $AI = IA = A$, just like multiplying a number by 1.
Build it with `np.eye(n)`.
"""))
a.append(code("""
import numpy as np
I = np.eye(3)        # 3 x 3 identity
print(I)

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("A @ I = A? ", np.allclose(A @ I, A))   # True
"""))

a.append(md(r"""
## 6. The big idea: a matrix *transforms* space

Multiplying a matrix $A$ by a **vector** $\mathbf{x}$ gives a new vector
$A\mathbf{x}$. Think of $A$ as a **function** that moves every point of the
plane to a new place. Because

$$ A(c\mathbf{x}) = c\,A\mathbf{x}, \qquad A(\mathbf{x}+\mathbf{y}) = A\mathbf{x}+A\mathbf{y}, $$

these are exactly the **linear transformations**. Let's apply one to a single
vector first.
"""))
a.append(code("""
import numpy as np

A = np.array([[2, 0],
              [0, 3]])       # stretches x by 2, y by 3
x = np.array([1, 1])        # the point (1, 1)

print("A @ x =", A @ x)     # -> [2 3]:  x doubled, y tripled
"""))

a.append(md(r"""
### Seeing the transformation on the unit square

The clearest way to understand a $2\times 2$ matrix is to watch what it does to
the **unit square** (corners $(0,0),(1,0),(1,1),(0,1)$). We store the corners as
**columns** of a matrix `S`, then transform them all at once with `A @ S`.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

# Corners of the unit square as COLUMNS (last point repeats the first to close it)
square = np.array([[0, 1, 1, 0, 0],     # x-coordinates
                   [0, 0, 1, 1, 0]])    # y-coordinates

def show_transform(A, title):
    \"\"\"Plot the unit square before (blue) and after (red) applying A.\"\"\"
    out = A @ square                    # transform all corners at once
    plt.figure(figsize=(5, 5))
    plt.plot(square[0], square[1], "b-o", label="before (unit square)")
    plt.plot(out[0],    out[1],    "r-o", label="after  (A @ square)")
    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.gca().set_aspect("equal")       # so squares look square
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()

# A SCALING matrix: x -> 2x, y -> 0.5y
A_scale = np.array([[2.0, 0.0],
                    [0.0, 0.5]])
show_transform(A_scale, "Scaling: wider and shorter")
"""))

a.append(md(r"""
### Rotation

A counter-clockwise rotation by angle $\theta$ is

$$ R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}. $$
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

theta = np.pi / 6            # 30 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

out = R @ square
plt.figure(figsize=(5, 5))
plt.plot(square[0], square[1], "b-o", label="before")
plt.plot(out[0],    out[1],    "r-o", label="after (rotated 30 deg)")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Rotation by 30 degrees")
plt.show()
"""))

a.append(md(r"""
### Shear

A **shear** slides points sideways by an amount proportional to their height:

$$ \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}. $$
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])

A_shear = np.array([[1.0, 1.0],     # k = 1: top edge slides right by 1
                    [0.0, 1.0]])
out = A_shear @ square
plt.figure(figsize=(5, 5))
plt.plot(square[0], square[1], "b-o", label="before")
plt.plot(out[0],    out[1],    "r-o", label="after (shear)")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Shear: the square becomes a parallelogram")
plt.show()
"""))

a.append(md(r"""
### Transforming a cloud of points

The same matrix moves *every* point the same way. Here we transform a circle of
points and watch it become an ellipse.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
t = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(t), np.sin(t)])   # 2 x 200 points on the unit circle

A = np.array([[2.0, 1.0],
              [0.0, 1.0]])
ellipse = A @ circle

plt.figure(figsize=(5, 5))
plt.plot(circle[0],  circle[1],  "b.", label="before (circle)")
plt.plot(ellipse[0], ellipse[1], "r.", label="after (A @ points)")
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("A linear map sends the circle to an ellipse")
plt.show()
"""))

a.append(md(r"""
## 7. Composition of transformations = matrix product

Doing transformation $B$ **then** $A$ on a vector $\mathbf{x}$ is

$$ A(B\mathbf{x}) = (AB)\mathbf{x}. $$

So **applying two matrices in a row is the same as multiplying them first.**
The order is *right-to-left* (the matrix closest to $\mathbf{x}$ acts first).
Let's verify: rotate, then scale.
"""))
a.append(code("""
import numpy as np

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])   # rotate 45 deg
S = np.array([[2.0, 0.0],
              [0.0, 1.0]])                         # then stretch x

x = np.array([1.0, 0.0])

# Apply rotation first, then scaling
step_by_step = S @ (R @ x)
# Combine into ONE matrix first
combined     = (S @ R) @ x

print("S @ (R @ x) =", step_by_step)
print("(S @ R) @ x =", combined)
print("same? ", np.allclose(step_by_step, combined))   # True
"""))

a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.array([[2.0, 0.0],
              [0.0, 1.0]])

after_R  = R @ square          # just rotated
after_SR = (S @ R) @ square    # rotated THEN scaled, via one matrix

plt.figure(figsize=(5, 5))
plt.plot(square[0],   square[1],   "b-o", label="original")
plt.plot(after_R[0],  after_R[1],  "g-o", label="after R (rotate)")
plt.plot(after_SR[0], after_SR[1], "r-o", label="after S@R (rotate+scale)")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Composition: one product does both steps")
plt.show()
"""))

# ---- Exercise 1 ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Build the matrix $A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$ and the
vector $\mathbf{x} = (3, 4)$. Compute $A\mathbf{x}$ by hand, then check with
NumPy. Also report the shapes of $A$, $\mathbf{x}$, and $A\mathbf{x}$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
import numpy as np
A = np.array([[1, 2],
              [0, 1]])
x = np.array([3, 4])

# By hand: row 0 -> 1*3 + 2*4 = 11 ;  row 1 -> 0*3 + 1*4 = 4
print("A @ x =", A @ x)            # [11  4]
print("shape A   :", A.shape)      # (2, 2)
print("shape x   :", x.shape)      # (2,)
print("shape A@x :", (A @ x).shape)  # (2,)
"""))

# ---- Exercise 2 ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

Show numerically that rotating by $90^\circ$ twice equals rotating by
$180^\circ$ (which flips a vector to its negative). Build $R = R(90^\circ)$ and
check that $R\,R$ sends $\mathbf{x}=(1,0)$ to $(-1,0)$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
import numpy as np
theta = np.pi / 2            # 90 degrees
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

x = np.array([1.0, 0.0])
print("R @ x      =", R @ x)        # [0, 1]  (pointing up)
print("R @ R @ x  =", R @ (R @ x))  # [-1, 0] (pointing left)

# R @ R should equal the 180-degree rotation = -I
print("R @ R =\\n", np.round(R @ R, 10))
print("equals -I? ", np.allclose(R @ R, -np.eye(2)))
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Create a $3\times 3$ matrix of your choice and a compatible $3\times 1$
   vector. Compute the product and verify one entry by hand.
2. Pick two $2\times 2$ matrices $A$ and $B$ and confirm that, in general,
   $AB \neq BA$. Can you find a *special* pair where $AB = BA$?
3. The **reflection across the $x$-axis** is
   $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$. Apply it to the unit square
   and plot before/after.
4. Build a rotation $R(60^\circ)$ and a shear $H=\begin{bmatrix}1&0.5\\0&1\end{bmatrix}$.
   Plot the unit square under $H R$ and under $R H$. Are the results the same?
   Explain using the fact that matrix multiplication is not commutative.
"""))

save(os.path.join(CH, "04a_matrices.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 04b — Linear systems
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 04b — Linear Systems: Solving $A\mathbf{x}=\mathbf{b}$

A **linear system** is a set of linear equations, for example

$$ \begin{aligned} 2x + y &= 5,\\ x - y &= 1. \end{aligned} $$

We can pack it into a single matrix equation $A\mathbf{x}=\mathbf{b}$:

$$ \underbrace{\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}}_{A}
   \underbrace{\begin{bmatrix} x \\ y \end{bmatrix}}_{\mathbf{x}}
   = \underbrace{\begin{bmatrix} 5 \\ 1 \end{bmatrix}}_{\mathbf{b}}. $$

In this notebook we solve such systems with NumPy, meet the **inverse**,
**determinant**, and **rank**, and see the *geometry*: solutions are where lines
intersect.
"""))

b.append(md(r"""
## 1. Solving a system with `np.linalg.solve`

Give NumPy the matrix $A$ and the right-hand side $\mathbf{b}$; it returns the
$\mathbf{x}$ that solves $A\mathbf{x}=\mathbf{b}$.
"""))
b.append(code("""
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, -1.0]])
b = np.array([5.0, 1.0])

x = np.linalg.solve(A, b)
print("solution x =", x)          # [2. 1.]  ->  x=2, y=1

# Always sanity-check: does A @ x give back b?
print("A @ x =", A @ x)           # [5. 1.]
print("matches b? ", np.allclose(A @ x, b))
"""))

b.append(md(r"""
## 2. The inverse $A^{-1}$ — and why `solve` is better

If $A$ is square and "nice", there is an **inverse** $A^{-1}$ with
$A^{-1}A = AA^{-1} = I$. Then in principle $\mathbf{x}=A^{-1}\mathbf{b}$.
NumPy gives it with `np.linalg.inv`.
"""))
b.append(code("""
import numpy as np
A = np.array([[2.0, 1.0],
              [1.0, -1.0]])
b = np.array([5.0, 1.0])

Ainv = np.linalg.inv(A)
print("A^-1 =\\n", Ainv)
print("A^-1 @ A =\\n", np.round(Ainv @ A, 10))   # the identity

x = Ainv @ b
print("x via inverse =", x)        # same [2. 1.]
"""))

b.append(md(r"""
> **Why prefer `solve` over `inv`?** Computing the full inverse does *more*
> work than needed and is **less accurate** (it accumulates rounding error).
> `np.linalg.solve(A, b)` goes straight to the answer faster and more reliably.
> **Rule of thumb: never write `inv(A) @ b`; write `solve(A, b)`.**
"""))
b.append(code("""
import numpy as np
A = np.array([[2.0, 1.0],
              [1.0, -1.0]])
b = np.array([5.0, 1.0])

print("solve :", np.linalg.solve(A, b))   # preferred
print("inv   :", np.linalg.inv(A) @ b)    # works, but discouraged
"""))

b.append(md(r"""
## 3. The determinant: area/volume scaling

The **determinant** $\det(A)$ measures how much the transformation $A$ scales
**area** (in 2-D) or **volume** (in 3-D). A unit square has area 1; after
applying $A$ its area becomes $|\det(A)|$.

- $|\det A| > 1$: the map *expands* area.
- $|\det A| < 1$: it *shrinks* area.
- $\det A < 0$: it also *flips* orientation (a mirror reflection).
"""))
b.append(code("""
import numpy as np

A = np.array([[2.0, 0.0],
              [0.0, 3.0]])     # stretches x by 2, y by 3
print("det(A) =", np.linalg.det(A))   # 6.0  -> area multiplied by 6

R = np.array([[0.0, -1.0],
              [1.0,  0.0]])    # a 90-degree rotation
print("det(R) =", np.linalg.det(R))   # 1.0  -> rotations preserve area
"""))

b.append(md(r"""
### Visual check: area really scales by $|\det A|$
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

square = np.array([[0, 1, 1, 0, 0],
                   [0, 0, 1, 1, 0]])
A = np.array([[2.0, 1.0],
              [0.0, 1.5]])
out = A @ square

print("det(A) =", np.linalg.det(A))   # 3.0 -> new area is 3

plt.figure(figsize=(5, 5))
plt.fill(square[0], square[1], alpha=0.3, label="unit square (area 1)")
plt.fill(out[0],    out[1],    alpha=0.3, label="image (area = |det A|)")
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Determinant = area-scaling factor")
plt.show()
"""))

b.append(md(r"""
## 4. Singular matrices: when $\det A = 0$

If $\det(A)=0$, the matrix **collapses** area to zero — it squashes the plane
onto a line (or a point). Such a matrix is called **singular**: it has **no
inverse**, and $A\mathbf{x}=\mathbf{b}$ may have *no* solution or *infinitely
many*. `np.linalg.solve` will refuse (it raises an error).
"""))
b.append(code("""
import numpy as np

A = np.array([[1.0, 2.0],
              [2.0, 4.0]])     # row 2 is exactly 2 x row 1
print("det(A) =", np.linalg.det(A))   # 0.0 (up to tiny rounding) -> singular

try:
    np.linalg.inv(A)
except np.linalg.LinAlgError as e:
    print("inv failed:", e)

try:
    np.linalg.solve(A, np.array([1.0, 0.0]))
except np.linalg.LinAlgError as e:
    print("solve failed:", e)
"""))

b.append(md(r"""
## 5. Rank: how many independent directions

The **rank** of $A$ is the number of linearly independent rows (equivalently,
columns) — how many genuinely different directions the matrix keeps. For a
square $n\times n$ matrix:

- **full rank** ($\text{rank}=n$): invertible, $\det\neq 0$, unique solution.
- **rank $< n$**: singular, $\det = 0$, no/infinitely-many solutions.

Use `np.linalg.matrix_rank`.
"""))
b.append(code("""
import numpy as np

full = np.array([[2.0, 1.0],
                 [1.0, -1.0]])
deficient = np.array([[1.0, 2.0],
                      [2.0, 4.0]])   # second row depends on the first

print("rank(full)      =", np.linalg.matrix_rank(full))       # 2  (full rank)
print("rank(deficient) =", np.linalg.matrix_rank(deficient))  # 1  (rank deficient)
"""))

b.append(md(r"""
## 6. Geometry: no solution vs. infinitely many

Each equation $a x + b y = c$ is a **line** in the plane. Solving a $2\times 2$
system means finding where the two lines **meet**.

- **One solution:** the lines cross at a single point (the normal case).
- **No solution:** the lines are **parallel but different** — they never meet.
- **Infinitely many:** the two equations describe the **same line**.

Let's draw all three.
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 4, 100)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# --- (a) Unique solution: 2x+y=5 and x-y=1 cross at (2,1) ---
axes[0].plot(x, 5 - 2*x, label="2x + y = 5")
axes[0].plot(x, x - 1,   label="x - y = 1")
axes[0].plot(2, 1, "ko")               # the intersection
axes[0].set_title("Unique solution")

# --- (b) No solution: two PARALLEL lines (same slope, different intercept) ---
axes[1].plot(x, 2 - x,     label="x + y = 2")
axes[1].plot(x, 4 - x,     label="x + y = 4")
axes[1].set_title("No solution (parallel)")

# --- (c) Infinitely many: the SAME line written twice ---
axes[2].plot(x, 2 - x,        label="x + y = 2")
axes[2].plot(x, 2 - x, "--",  label="2x + 2y = 4 (same line)")
axes[2].set_title("Infinitely many (identical)")

for ax in axes:
    ax.set_aspect("equal"); ax.grid(True); ax.legend(); ax.set_xlim(-1, 4)
plt.show()
"""))

b.append(md(r"""
We can detect these cases numerically with the **determinant** and **rank** of
$A$ (and of the *augmented* matrix $[A\,|\,\mathbf{b}]$):
"""))
b.append(code("""
import numpy as np

# No-solution system (parallel lines): x+y=2, x+y=4
A = np.array([[1.0, 1.0],
              [1.0, 1.0]])
b = np.array([2.0, 4.0])

print("det(A)            =", np.linalg.det(A))            # 0 -> singular
print("rank(A)           =", np.linalg.matrix_rank(A))    # 1
aug = np.column_stack([A, b])                             # [A | b]
print("rank([A|b])       =", np.linalg.matrix_rank(aug))  # 2
print("rank(A) < rank([A|b])  -> NO solution")
"""))

b.append(md(r"""
## 7. A fully worked $2\times 2$ example, visualized

Solve

$$ \begin{aligned} 3x + 2y &= 12,\\ x - y &= 1, \end{aligned} $$

both algebraically (`solve`) and graphically (intersection of two lines).
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3.0, 2.0],
              [1.0, -1.0]])
b = np.array([12.0, 1.0])

# 1) Solve it
sol = np.linalg.solve(A, b)
print("solution (x, y) =", sol)          # [2.8 1.8]
print("check A @ x:", A @ sol, " vs b:", b)

# 2) Show it as intersecting lines
x = np.linspace(-1, 5, 100)
line1 = (12 - 3*x) / 2      # from 3x + 2y = 12  ->  y = (12 - 3x)/2
line2 = x - 1               # from  x -  y = 1   ->  y = x - 1

plt.figure(figsize=(5, 5))
plt.plot(x, line1, label="3x + 2y = 12")
plt.plot(x, line2, label="x - y = 1")
plt.plot(sol[0], sol[1], "ko", markersize=9, label=f"solution ({sol[0]:.1f}, {sol[1]:.1f})")
plt.axhline(0, color="gray", lw=0.5); plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Solution = where the two lines cross")
plt.show()
"""))

# ---- Exercise 1 ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Solve the system

$$ \begin{aligned} 4x + 3y &= 10,\\ 2x - y &= 0, \end{aligned} $$

using `np.linalg.solve`. Verify your answer by plugging it back in, and report
$\det(A)$ to confirm the system has a unique solution.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import numpy as np
A = np.array([[4.0, 3.0],
              [2.0, -1.0]])
b = np.array([10.0, 0.0])

x = np.linalg.solve(A, b)
print("solution =", x)                 # [1. 2.]
print("check A @ x =", A @ x)          # [10. 0.] -> matches b
print("det(A) =", np.linalg.det(A))    # -10 (nonzero) -> unique solution
"""))

# ---- Exercise 2 ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Consider $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$. Compute
$\det(A)$ and $\text{rank}(A)$. Is $A$ invertible? Then decide, using ranks,
whether $A\mathbf{x} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$ has solutions, and
whether $A\mathbf{x} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}$ does.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import numpy as np
A = np.array([[1.0, 2.0],
              [2.0, 4.0]])

print("det(A)  =", np.linalg.det(A))            # 0 -> singular, NOT invertible
print("rank(A) =", np.linalg.matrix_rank(A))    # 1

# Case b = [3, 6]: it lies ON the line -> infinitely many solutions
b1 = np.array([3.0, 6.0])
aug1 = np.column_stack([A, b1])
print("b=[3,6]: rank([A|b]) =", np.linalg.matrix_rank(aug1),
      "== rank(A) -> infinitely many solutions")

# Case b = [3, 7]: inconsistent -> no solution
b2 = np.array([3.0, 7.0])
aug2 = np.column_stack([A, b2])
print("b=[3,7]: rank([A|b]) =", np.linalg.matrix_rank(aug2),
      ">  rank(A) -> NO solution")
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Solve $\begin{aligned} x + 2y &= 4 \\ 3x - y &= 5 \end{aligned}$ with
   `np.linalg.solve`, verify by substitution, and plot the two lines with their
   intersection marked.
2. For $A = \begin{bmatrix} 1 & 1 \\ 1 & 1.0001 \end{bmatrix}$, compute
   $\det(A)$. It is *tiny* but nonzero — such a matrix is "nearly singular".
   Solve $A\mathbf{x}=(2,2)$ and then $A\mathbf{x}=(2,2.001)$; notice how a tiny
   change in $\mathbf{b}$ swings the answer a lot.
3. Build a $3\times 3$ system of your choice that has a unique solution
   (check $\det \neq 0$), solve it, and verify $A\mathbf{x}=\mathbf{b}$.
4. Construct a $2\times 2$ system whose two lines are **parallel** (no
   solution). Confirm with `det`, `matrix_rank(A)`, and
   `matrix_rank([A | b])`, and plot the two parallel lines.
"""))

save(os.path.join(CH, "04b_linear_systems.ipynb"), b)
