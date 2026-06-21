"""Generator for Chapter 01 — NumPy for Mathematics.

Run from anywhere:  python tools/generators/ch01_numpy.py
Produces two notebooks in 01-numpy-for-mathematics/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "01-numpy-for-mathematics")


# ---------------------------------------------------------------------------
# Notebook 01a — Arrays: the basics
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 01a — NumPy Arrays: the Basics

In Chapter 00 we built lists of numbers with loops and found it clumsy. **NumPy**
fixes this. Its central object is the **array** (`ndarray`): a grid of numbers
that behaves like a vector or a matrix.

The big idea: an array lets you write math on *whole collections of numbers at
once*, in notation that mirrors the page. A vector $\mathbf{v}\in\mathbb{R}^n$
is a 1D array; a matrix $A\in\mathbb{R}^{m\times n}$ is a 2D array.

Run each cell with **Shift + Enter**. You cannot break anything.
"""))

a.append(md(r"""
## 1. Importing NumPy

By universal convention we import NumPy as `np`. Do this once at the top of a
notebook.
"""))
a.append(code("""
import numpy as np      # the universal nickname for NumPy

print(np.__version__)   # which version is installed
"""))

a.append(md(r"""
## 2. Making an array from a list

The simplest way to build an array is `np.array(...)` from a Python list.
Think of this as writing down a vector $\mathbf{v} = (2, 3, 5, 7)$.
"""))
a.append(code("""
v = np.array([2, 3, 5, 7])   # a 1D array = a vector
print(v)
print(type(v))               # <class 'numpy.ndarray'>
"""))

a.append(md(r"""
A list of lists becomes a 2D array — a **matrix**. Each inner list is a row.

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$$
"""))
a.append(code("""
A = np.array([[1, 2, 3],
              [4, 5, 6]])    # two rows, three columns
print(A)
"""))

a.append(md(r"""
## 3. Building arrays without typing every number

Typing numbers by hand does not scale. NumPy has factory functions:

| Function | What it makes |
|----------|---------------|
| `np.arange(start, stop, step)` | evenly spaced values, like `range` (stop excluded) |
| `np.linspace(start, stop, num)` | `num` points evenly spaced, **stop included** |
| `np.zeros(shape)` | all zeros |
| `np.ones(shape)` | all ones |
| `np.full(shape, value)` | all equal to `value` |
| `np.eye(n)` | the $n\times n$ identity matrix $I$ |
"""))
a.append(code("""
print(np.arange(0, 10, 2))        # 0,2,4,6,8  (10 is excluded)
print(np.linspace(0, 1, 5))       # 5 points from 0 to 1 INCLUSIVE
"""))
a.append(code("""
print(np.zeros(4))                # vector of zeros
print(np.ones((2, 3)))            # 2x3 matrix of ones  (note: shape is a tuple)
print(np.full((2, 2), 7))         # 2x2 matrix of 7s
print(np.eye(3))                  # 3x3 identity matrix
"""))

a.append(md(r"""
**`arange` vs `linspace`.** Use `arange` when you care about the *step size*; use
`linspace` when you care about the *number of points* (and want the endpoint
included). For plotting a smooth function, `linspace` is almost always what you
want.
"""))

a.append(md(r"""
## 4. The anatomy of an array: shape, ndim, dtype

Every array carries three useful descriptions:

- `.shape` — a tuple of sizes along each direction, e.g. `(2, 3)` = 2 rows, 3 cols
- `.ndim`  — the number of dimensions (1 for a vector, 2 for a matrix)
- `.dtype` — the type of the numbers (e.g. `int64`, `float64`)
"""))
a.append(code("""
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("shape :", A.shape)   # (2, 3)
print("ndim  :", A.ndim)    # 2
print("dtype :", A.dtype)   # int64 (the numbers are whole)
print("size  :", A.size)    # 6  total number of elements
"""))
a.append(code("""
# A single decimal makes the WHOLE array floating-point:
w = np.array([1, 2, 3.0])
print(w)            # [1. 2. 3.]
print(w.dtype)      # float64

# You can also ask for a dtype explicitly:
z = np.array([1, 2, 3], dtype=float)
print(z, z.dtype)
"""))

a.append(md(r"""
## 5. Indexing: reading one element

Indexing starts at **0** (just like Python lists). For a 2D array use
`A[row, col]`.
"""))
a.append(code("""
v = np.array([10, 20, 30, 40, 50])
print(v[0])     # first element -> 10
print(v[-1])    # last element  -> 50

A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(A[0, 0])  # row 0, col 0 -> 1
print(A[1, 2])  # row 1, col 2 -> 6
"""))

a.append(md(r"""
## 6. Slicing: reading many elements at once

Slicing uses `start:stop` (stop excluded), exactly like lists. For 2D arrays you
slice each direction, separated by a comma. A lone `:` means "everything in this
direction".
"""))
a.append(code("""
v = np.array([10, 20, 30, 40, 50])
print(v[1:4])    # elements 1,2,3 -> [20 30 40]
print(v[:3])     # first three    -> [10 20 30]
print(v[::2])    # every 2nd one  -> [10 30 50]
"""))
a.append(code("""
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(A[0, :])     # whole first ROW    -> [1 2 3]
print(A[:, 1])     # whole second COL   -> [2 5 8]
print(A[0:2, 1:3]) # top-right 2x2 block
"""))

a.append(md(r"""
## 7. Reshaping: same numbers, new layout

`reshape` rearranges the same data into a different shape. The number of
elements must match. A handy trick: pass `-1` for "figure this dimension out for
me".
"""))
a.append(code("""
v = np.arange(12)          # 0,1,...,11  -> shape (12,)
print(v)

M = v.reshape(3, 4)        # 3 rows, 4 cols
print(M)

print(v.reshape(2, -1))    # 2 rows, NumPy infers 6 cols
"""))

a.append(md(r"""
## 8. Stacking arrays together

You can glue arrays edge-to-edge. `np.vstack` stacks vertically (adds rows);
`np.hstack` stacks horizontally (adds columns).
"""))
a.append(code("""
r1 = np.array([1, 2, 3])
r2 = np.array([4, 5, 6])

print(np.vstack([r1, r2]))   # stack as two rows -> 2x3 matrix
print(np.hstack([r1, r2]))   # join end to end   -> length-6 vector
"""))

a.append(md(r"""
## 9. Boolean masks: selecting by a condition

Comparing an array with a number gives a **boolean array** (True/False for each
entry). Indexing with that mask keeps only the `True` entries — this is how you
*filter* data. `np.where` picks between two values per entry, and
`np.any`/`np.all` summarize a whole array.
"""))
a.append(code("""
import numpy as np

v = np.array([-2, 5, -1, 3, 0, 8])

mask = v > 0                 # a boolean array
print(mask)                  # [False  True False  True False  True]
print(v[mask])               # [5 3 8]  -> only the positive entries
print(v[v > 0])              # the same, in one step

print("how many > 0 :", np.sum(v > 0))   # True counts as 1 -> 3
print("any negative?:", np.any(v < 0))   # True
print("all >= 0?    :", np.all(v >= 0))  # False
"""))
a.append(code("""
import numpy as np
v = np.array([-2, 5, -1, 3, 0, 8])

# np.where(condition, value_if_true, value_if_false).
# Replacing negatives with 0 is exactly the ReLU used in neural networks!
print(np.where(v > 0, v, 0))   # [0 5 0 3 0 8]

# You can also ASSIGN through a mask:
w = v.copy()
w[w < 0] = 0                   # set every negative entry to 0
print(w)                       # [0 5 0 3 0 8]
"""))

a.append(md(r"""
## 10. Views vs. copies (an important gotcha)

A slice of an array is a **view**: it shares memory with the original, so editing
the slice edits the original too. When you need an independent array, use
`.copy()`.
"""))
a.append(code("""
import numpy as np

a = np.array([10, 20, 30, 40])
s = a[1:3]          # a VIEW into a (not a fresh array)
s[0] = 999          # change the view ...
print("a changed!:", a)     # [ 10 999  30  40]  -> the original changed

b = a[1:3].copy()   # an independent COPY
b[0] = -1
print("a is safe :", a)     # original untouched
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Build the $3\times 3$ matrix

$$B = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

using `np.arange` and `reshape` (do **not** type the nine numbers by hand). Then
print its shape, its middle column, and its last row.
"""))
a.append(md("**Solution:**"))
a.append(code("""
import numpy as np

B = np.arange(1, 10).reshape(3, 3)   # 1..9 laid out 3x3
print(B)
print("shape       :", B.shape)      # (3, 3)
print("middle col  :", B[:, 1])      # [2 5 8]
print("last row    :", B[-1, :])     # [7 8 9]
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

Create 11 evenly spaced points on the interval $[0, 1]$ (including both
endpoints). Then pull out (a) the first 5 points and (b) every other point.
*Hint:* this is a job for `linspace`, not `arange`.
"""))
a.append(md("**Solution:**"))
a.append(code("""
import numpy as np

x = np.linspace(0, 1, 11)   # 0.0, 0.1, ..., 1.0
print(x)
print("first 5     :", x[:5])     # [0.  0.1 0.2 0.3 0.4]
print("every other :", x[::2])    # [0.  0.2 0.4 0.6 0.8 1. ]
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Make a length-10 array of all `5`s in two different ways (one using
   `np.full`, one using `np.ones`).
2. Build a $4\times 4$ matrix containing the numbers $0,1,\dots,15$ in
   row-by-row order, then extract its $2\times 2$ bottom-right corner.
3. Create the vector $(0, 0.25, 0.5, 0.75, 1.0)$ twice: once with `arange` and
   once with `linspace`. Confirm both give the same numbers.
4. Take `A = np.arange(1, 13).reshape(3, 4)`. Use slicing to print its 2nd row,
   its last column, and every element in even-indexed columns ($0, 2, \dots$).
5. With `rng = np.random.default_rng(0)` and `v = rng.integers(-10, 10, size=20)`,
   use a boolean mask to (a) keep only the even entries, and (b) replace every
   negative entry with 0 using `np.where`.
"""))

save(os.path.join(CH, "01a_arrays_basics.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 01b — Vectorization & broadcasting
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 01b — Vectorization & Broadcasting

This is where NumPy earns its keep. The theme of this whole notebook:

> **Turn an equation directly into array code.**

If a formula says "do this to every entry", you do **not** write a loop — you
write the formula once and NumPy applies it to the entire array at once. This is
called **vectorization**. It is shorter, reads like math, and runs *much* faster.
"""))

b.append(md(r"""
## 1. Elementwise arithmetic

Arithmetic on arrays happens **elementwise**: `u + v` adds matching entries,
`u * v` multiplies matching entries, and so on. This is exactly vector addition
and scalar multiplication from linear algebra.
"""))
b.append(code("""
import numpy as np

u = np.array([1, 2, 3])
v = np.array([10, 20, 30])

print(u + v)     # [11 22 33]   entry by entry
print(v - u)     # [ 9 18 27]
print(u * v)     # [10 40 90]   elementwise product (NOT the dot product)
print(v / u)     # [10. 10. 10.]
print(u ** 2)    # [1 4 9]      square every entry
"""))
b.append(code("""
# A scalar applies to every entry (this is "broadcasting", section 4):
print(3 * u)     # [3 6 9]   scalar multiple of a vector
print(u + 100)   # [101 102 103]
"""))

b.append(md(r"""
## 2. Universal functions (ufuncs): math on whole arrays

NumPy ships fast, elementwise versions of the usual math functions. They are
called **ufuncs**. Feed in an array, get back an array of the same shape.

$\sin,\ \cos,\ \exp,\ \log,\ \sqrt,\ \dots$ all work this way.
"""))
b.append(code("""
import numpy as np

x = np.array([0.0, 1.0, 4.0, 9.0])

print(np.sqrt(x))   # [0. 1. 2. 3.]    square root of each entry
print(np.exp(x))    # e^x for each entry
print(np.log1p(x))  # log(1+x), safe near 0
"""))
b.append(code("""
# Evaluate sin at a few special angles, all at once:
angles = np.array([0, np.pi / 6, np.pi / 4, np.pi / 2])
print(np.sin(angles))    # [0.   0.5  0.707 1. ]  (approx)
print(np.cos(angles))
"""))

b.append(md(r"""
**Turning an equation into code.** Suppose $g(x) = \sqrt{x}\,e^{-x}$. On paper
that's one line; in NumPy it is also one line, and it works on a whole array of
$x$ values simultaneously:
"""))
b.append(code("""
import numpy as np

x = np.linspace(0, 4, 9)         # nine x-values
g = np.sqrt(x) * np.exp(-x)      # the formula, applied to ALL of them at once
for xi, gi in zip(x, g):
    print(f"g({xi:.1f}) = {gi:.4f}")
"""))

b.append(md(r"""
## 3. Vectorization vs. Python loops (and why speed matters)

The same computation can be done with a slow Python loop or with one vectorized
expression. Let's check they agree, then time them.
"""))
b.append(code("""
import numpy as np

N = 1_000_000
x = np.linspace(0, 10, N)

# --- Way 1: a plain Python loop (slow) ---
y_loop = np.empty(N)
for i in range(N):
    y_loop[i] = np.sin(x[i]) ** 2     # work on one element at a time

# --- Way 2: vectorized (fast) ---
y_vec = np.sin(x) ** 2                 # one expression, whole array

# Do they give the same numbers? (allclose checks "equal up to tiny rounding")
print("same result?", np.allclose(y_loop, y_vec))
"""))
b.append(code("""
import numpy as np
import time

N = 1_000_000
x = np.linspace(0, 10, N)

# Time the loop
t0 = time.perf_counter()
y_loop = np.empty(N)
for i in range(N):
    y_loop[i] = np.sin(x[i]) ** 2
t1 = time.perf_counter()

# Time the vectorized version
t2 = time.perf_counter()
y_vec = np.sin(x) ** 2
t3 = time.perf_counter()

loop_time = t1 - t0
vec_time = t3 - t2
print(f"loop       : {loop_time:.4f} s")
print(f"vectorized : {vec_time:.4f} s")
print(f"speedup    : about {loop_time / vec_time:.0f}x faster")
"""))
b.append(md(r"""
The vectorized version is typically **tens to hundreds of times faster** — and
shorter to write. The lesson: *if you find yourself writing a `for` loop over the
entries of an array, look for the vectorized expression instead.*
"""))

b.append(md(r"""
## 4. Broadcasting: combining different shapes

What if shapes don't match? NumPy **broadcasts**: it stretches the smaller array
across the larger one when their shapes are compatible. You already saw the
simplest case, `3 * u`, where the scalar `3` was applied to every entry.

**The rule (read shapes from the right):** two dimensions are compatible when
they are **equal**, or **one of them is 1**. A size-1 (or missing) dimension is
stretched to match.
"""))
b.append(code("""
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
row = np.array([10, 20, 30])   # shape (3,)

# 'row' is stretched down to both rows of A:
print(A + row)
# [[11 22 33]
#  [14 25 36]]
"""))
b.append(code("""
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2, 3)
col = np.array([[100],
                [200]])            # shape (2, 1)

# 'col' is stretched across the three columns:
print(A + col)
# [[101 102 103]
#  [204 205 206]]
"""))
b.append(md(r"""
A classic use: build a **multiplication table** (an outer product) by combining a
column vector and a row vector. The $(i,j)$ entry becomes $i \cdot j$.
"""))
b.append(code("""
import numpy as np

i = np.arange(1, 6).reshape(5, 1)   # column 1..5, shape (5, 1)
j = np.arange(1, 6).reshape(1, 5)   # row    1..5, shape (1, 5)

table = i * j                        # broadcasting -> 5x5 product table
print(table)
"""))

b.append(md(r"""
## 5. Reductions: collapsing an array to a summary

A **reduction** boils an array down to fewer numbers: `sum`, `mean`, `std`
(standard deviation), `min`, `max`. With no `axis`, it uses the whole array.
"""))
b.append(code("""
import numpy as np

x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
print("sum  :", x.sum())
print("mean :", x.mean())
print("std  :", x.std())     # population standard deviation
print("min  :", x.min(), " max :", x.max())
"""))
b.append(md(r"""
For a 2D array, `axis` chooses the direction to collapse:

- `axis=0` collapses **down the rows** → one value per **column**.
- `axis=1` collapses **across the columns** → one value per **row**.
"""))
b.append(code("""
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)

print("total          :", A.sum())          # 21  (everything)
print("column sums    :", A.sum(axis=0))    # [5 7 9]   (down the rows)
print("row sums       :", A.sum(axis=1))    # [ 6 15]   (across the columns)
print("column means   :", A.mean(axis=0))   # [2.5 3.5 4.5]
"""))

b.append(md(r"""
## 6. The payoff: an equation becomes a plot

Now we combine everything. To plot $f(x) = \sin(x)$ on $[0, 2\pi]$ we make a fine
grid with `linspace`, apply the ufunc `np.sin` to the whole grid, and plot. No
loops anywhere — the code reads like the math.
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)   # 200 points across [0, 2*pi]
y = np.sin(x)                        # apply sin to ALL of them at once

plt.plot(x, y)
plt.title(r"$y = \\sin(x)$")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.axhline(0, color="gray", linewidth=0.8)   # the x-axis
plt.grid(True)
plt.show()
"""))
b.append(md(r"""
Compare this to the list-and-loop version from Chapter 00b. Same picture, far
less code — and it generalizes: swap `np.sin` for any ufunc and you instantly
plot a different function.
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Without any loop, evaluate the **Gaussian / bell curve**

$$f(x) = e^{-x^2}$$

on 201 points over $[-3, 3]$, and report its maximum value and where (which
$x$) the maximum occurs. *Hint:* `np.argmax` gives the **index** of the largest
entry.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import numpy as np

x = np.linspace(-3, 3, 201)
f = np.exp(-x ** 2)            # the formula, vectorized

print("max value :", f.max())             # 1.0
i = np.argmax(f)                           # index of the largest entry
print("at x      :", x[i])                 # 0.0  (the peak sits at x=0)
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

You are given a small data matrix where each **row** is a sample and each
**column** is a feature. **Standardize** each column so it has mean 0 and
standard deviation 1, i.e. replace each entry by

$$z = \frac{x - \mu_{\text{col}}}{\sigma_{\text{col}}}.$$

Do it in one broadcasted expression (no loops). Then verify the new column means
are ~0 and column standard deviations are ~1.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(loc=5.0, scale=2.0, size=(6, 3))   # 6 samples, 3 features

mu = X.mean(axis=0)     # one mean per column,  shape (3,)
sd = X.std(axis=0)      # one std  per column,  shape (3,)

Z = (X - mu) / sd       # broadcasting subtracts/divides per column

print("new column means:", np.round(Z.mean(axis=0), 10))  # ~ [0 0 0]
print("new column stds :", np.round(Z.std(axis=0), 10))   # ~ [1 1 1]
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. On a grid `x = np.linspace(0, 2*np.pi, 300)`, plot $\sin(x)$ and $\cos(x)$ on
   the **same** axes. (Call `plt.plot` twice before `plt.show()`.)
2. Verify the identity $\sin^2 x + \cos^2 x = 1$ numerically: compute
   `np.sin(x)**2 + np.cos(x)**2` on a grid and check it is `1` everywhere with
   `np.allclose`.
3. Build the $10\times 10$ multiplication table using broadcasting, then use a
   reduction to find the sum of all its entries and the largest value in each
   row.
4. Implement the **sigmoid** $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ as a one-line
   vectorized function, evaluate it on `np.linspace(-6, 6, 200)`, and plot it.
   Where does the curve cross $y = 0.5$?
"""))

save(os.path.join(CH, "01b_vectorization_broadcasting.ipynb"), b)
