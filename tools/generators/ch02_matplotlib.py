"""Generator for Chapter 02 — Visualization with Matplotlib.

Run from anywhere:  python tools/generators/ch02_matplotlib.py
Produces two notebooks in 02-visualization-with-matplotlib/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "02-visualization-with-matplotlib")


# ---------------------------------------------------------------------------
# Notebook 02a — Plotting functions and data
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 02a — Plotting Functions & Data

A picture turns a formula into intuition. In this notebook we use
**Matplotlib**, the standard plotting library, together with **NumPy** (from
Chapter 01) to draw functions and data.

The recipe is almost always the same three steps:

1. Build an array of $x$ values with `np.linspace`.
2. Compute the matching $y$ values with a vectorised formula, e.g. `np.sin(x)`.
3. Hand `(x, y)` to `plt.plot` and decorate the picture (title, labels, grid).

Run each cell with **Shift + Enter**. Edit and re-run — you cannot break
anything.
"""))

a.append(md(r"""
## 1. The setup line

By convention we import NumPy as `np` and the plotting interface as `plt`.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

# A reproducible random-number generator (same numbers every run).
rng = np.random.default_rng(0)
"""))

a.append(md(r"""
## 2. Your first plot: a function on a grid

To draw $f(x) = \sin(x)$ we need many closely-spaced $x$ values.
`np.linspace(a, b, n)` returns `n` evenly spaced points from `a` to `b`
(both endpoints included). NumPy then evaluates `np.sin` on all of them at once.
"""))
a.append(code("""
x = np.linspace(0, 2 * np.pi, 200)   # 200 points from 0 to 2*pi
y = np.sin(x)                        # one y for every x, all at once

plt.plot(x, y)        # connect the (x, y) points with a line
plt.show()            # render the figure
"""))

a.append(md(r"""
The smooth curve is really 200 tiny straight segments — with enough points the
eye sees a smooth sine wave. More points = smoother curve.
"""))

a.append(md(r"""
## 3. Decorating a plot: title, labels, grid

A plot without labels is a riddle. Always say **what** is on each axis.
"""))
a.append(code("""
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x)

plt.plot(x, y)
plt.title("The sine function")   # heading above the plot
plt.xlabel("x")                  # label the horizontal axis
plt.ylabel("sin(x)")             # label the vertical axis
plt.grid(True)                   # add a light reference grid
plt.show()
"""))

a.append(md(r"""
## 4. Several curves on one axes + a legend

Just call `plt.plot` more than once before `plt.show()`. Give each curve a
`label`, then call `plt.legend()` so the reader knows which is which.
"""))
a.append(code("""
x = np.linspace(0, 2 * np.pi, 200)

plt.plot(x, np.sin(x), label="sin(x)")
plt.plot(x, np.cos(x), label="cos(x)")
plt.title("Sine and cosine")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()          # show the box matching colors to labels
plt.grid(True)
plt.show()
"""))

a.append(md(r"""
## 5. Line styles and colors

You can control how each line looks. A short format string after the data sets
color and style, e.g. `"r--"` = red dashed. You can also pass keywords like
`color`, `linestyle`, and `linewidth`.

| Code | Meaning |
|------|---------|
| `"r"` `"g"` `"b"` `"k"` | red, green, blue, black |
| `"-"` `"--"` `":"` `"-."` | solid, dashed, dotted, dash-dot |
| `"o"` `"s"` `"^"` | circle, square, triangle markers |
"""))
a.append(code("""
x = np.linspace(0, 2 * np.pi, 200)

plt.plot(x, np.sin(x), "r--", label="sin (red dashed)")
plt.plot(x, np.cos(x), color="navy", linestyle=":", linewidth=2,
         label="cos (blue dotted)")
plt.title("Controlling color and style")
plt.xlabel("x")
plt.ylabel("value")
plt.legend()
plt.grid(True)
plt.show()
"""))

a.append(md(r"""
## 6. Scatter plots: showing data points

When you have *data* (not a smooth function) you usually want dots, not a line.
`plt.scatter(x, y)` draws one marker per point. Here we make a noisy version of
the line $y = 2x + 1$ and scatter it against the true line.
"""))
a.append(code("""
x = np.linspace(0, 10, 40)
true_y = 2 * x + 1                     # the underlying straight line
noise = rng.normal(0, 1.5, size=x.shape)   # random wobble, mean 0
data_y = true_y + noise                # measured/noisy data

plt.scatter(x, data_y, color="black", label="noisy data")
plt.plot(x, true_y, "r-", label="true line  y = 2x + 1")
plt.title("Data scattered around a line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
"""))

a.append(md(r"""
## 7. Subplots: several panels in one figure

`plt.subplots(rows, cols)` returns a **figure** and an array of **axes**. Each
axes is its own little plot. Calling methods on an axes (`ax.plot`,
`ax.set_title`, ...) draws into that specific panel.
"""))
a.append(code("""
x = np.linspace(-3, 3, 200)

# A figure with 1 row, 2 columns of plots.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

axes[0].plot(x, x**2, color="teal")
axes[0].set_title("y = x^2")
axes[0].grid(True)

axes[1].plot(x, x**3, color="crimson")
axes[1].set_title("y = x^3")
axes[1].grid(True)

fig.suptitle("Two panels side by side")
fig.tight_layout()    # stop the titles/labels from overlapping
plt.show()
"""))

a.append(md(r"""
## 8. Histograms: the shape of a distribution

A **histogram** sorts numbers into bins and draws a bar for how many landed in
each bin. It reveals the *shape* of a dataset. Here we draw 5000 samples from a
standard normal (bell curve) and let `plt.hist` count them.
"""))
a.append(code("""
samples = rng.normal(0, 1, size=5000)   # 5000 draws from N(0, 1)

plt.hist(samples, bins=40, color="slateblue", edgecolor="white")
plt.title("Histogram of 5000 normal samples")
plt.xlabel("value")
plt.ylabel("count")
plt.grid(True, axis="y")
plt.show()
"""))

a.append(md(r"""
## 9. Bar charts: comparing categories

`plt.bar` draws one bar per category — handy for counts and, later, for the
probabilities a classifier assigns to each class.
"""))
a.append(code("""
classes = ["cat", "dog", "bird", "fish"]
probs   = [0.55, 0.30, 0.10, 0.05]

plt.bar(classes, probs, color="steelblue")
plt.title("Predicted probability per class")
plt.ylabel("probability")
plt.ylim(0, 1)
plt.show()
"""))

a.append(md(r"""
## 10. Annotations and shading

Two finishing touches you'll use a lot: `plt.fill_between` shades a region (the
area under a curve, or a confidence band), and `plt.annotate` points an arrow at
an interesting spot.
"""))
a.append(code("""
x = np.linspace(-3, 3, 200)
y = np.exp(-x**2)

plt.plot(x, y, color="darkorange")
plt.fill_between(x, y, alpha=0.3, color="orange")   # shade under the curve
plt.annotate("peak at x = 0", xy=(0, 1), xytext=(1.0, 0.85),
             arrowprops=dict(arrowstyle="->"))
plt.title("fill_between + annotate")
plt.xlabel("x"); plt.ylabel("y")
plt.show()
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

On the interval $[-2, 2]$ plot the parabola $f(x) = x^2$ and the line
$g(x) = 2x - 1$ on the **same axes**. Add a title, axis labels, a legend, and a
grid. (The line is the tangent to the parabola at $x = 1$ — notice they touch.)
"""))
a.append(md("**Solution:**"))
a.append(code("""
x = np.linspace(-2, 2, 200)

plt.plot(x, x**2, label="f(x) = x^2")
plt.plot(x, 2 * x - 1, "r--", label="g(x) = 2x - 1")
plt.title("A parabola and its tangent at x = 1")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

Make a figure with **two stacked panels** (2 rows, 1 column). In the top panel
draw a histogram of 2000 samples from a normal distribution with mean 5 and
standard deviation 2. In the bottom panel draw a histogram of 2000 samples from
a **uniform** distribution on $[0, 10]$ (use `rng.uniform(0, 10, size=2000)`).
Give each panel a title.
"""))
a.append(md("**Solution:**"))
a.append(code("""
normal_samples  = rng.normal(5, 2, size=2000)
uniform_samples = rng.uniform(0, 10, size=2000)

fig, axes = plt.subplots(2, 1, figsize=(7, 6))

axes[0].hist(normal_samples, bins=30, color="seagreen", edgecolor="white")
axes[0].set_title("Normal(mean=5, sd=2)")

axes[1].hist(uniform_samples, bins=30, color="orange", edgecolor="white")
axes[1].set_title("Uniform on [0, 10]")

fig.tight_layout()
plt.show()
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Plot $\tan(x)$ on $[-1.4, 1.4]$. Use `plt.ylim(-10, 10)` so the vertical
   asymptotes don't squash the picture.
2. On one axes, plot $x$, $x^2$, and $x^3$ for $x \in [0, 1]$, each with a
   different color and a legend. Which grows fastest near $x = 1$?
3. Generate `x = np.linspace(0, 2*np.pi, 50)` and noisy data
   `y = np.sin(x) + rng.normal(0, 0.2, size=50)`. Scatter the data and overlay
   the true curve `np.sin(x)`.
4. Draw a $2 \times 2$ grid of subplots showing $\sin$, $\cos$, $\sin(2x)$, and
   $\cos(2x)$ on $[0, 2\pi]$, each panel titled.
"""))

save(os.path.join(CH, "02a_plotting_functions.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 02b — Surfaces, contours, and fields
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 02b — Surfaces, Contours & Vector Fields

Many objects in machine learning are functions of **two** variables, like a
loss surface $L(w_1, w_2)$. This notebook shows the standard ways to picture a
function $f(x, y)$ and a vector field, all built on one key idea: the **grid**.
"""))

b.append(md(r"""
## 1. The grid: `np.meshgrid`

To evaluate $f(x, y)$ over a rectangle we need every combination of an $x$ and
a $y$. `np.meshgrid` takes a 1D list of $x$ values and a 1D list of $y$ values
and returns two 2D arrays `X` and `Y` such that `(X[i,j], Y[i,j])` is one grid
point. Then a vectorised formula gives `Z = f(X, Y)` for the whole grid at once.
"""))
a_setup = """
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
"""
b.append(code(a_setup))
b.append(code("""
# A small grid so we can SEE the arrays.
xs = np.linspace(0, 2, 3)   # [0, 1, 2]
ys = np.linspace(0, 1, 2)   # [0, 1]
X, Y = np.meshgrid(xs, ys)

print("X =\\n", X)   # x-coordinate at each grid point
print("Y =\\n", Y)   # y-coordinate at each grid point
"""))

b.append(md(r"""
## 2. A function of two variables

Let
$$f(x, y) = \exp\!\big(-(x^2 + y^2)\big),$$
a smooth "bump" centered at the origin. We build a fine grid and evaluate it.
"""))
b.append(code("""
xs = np.linspace(-2, 2, 200)
ys = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(xs, ys)

Z = np.exp(-(X**2 + Y**2))   # f(x, y) on the whole grid at once
print("Z has shape", Z.shape)   # one value per grid point
"""))

b.append(md(r"""
## 3. Contour lines: `plt.contour`

A **contour line** connects points where $f$ has the same value — exactly like
height lines on a topographic map. Closely-spaced contours mean a steep region.
"""))
b.append(code("""
plt.contour(X, Y, Z, levels=12)   # 12 evenly spaced level curves
plt.title("Contour lines of f(x, y) = exp(-(x^2 + y^2))")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")        # equal scaling so circles look like circles
plt.colorbar(label="f value")
plt.show()
"""))

b.append(md(r"""
## 4. Filled contours: `plt.contourf`

`contourf` (the **f** is for *filled*) shades the bands between contours, which
reads more like a heat map. A colorbar decodes color into value.
"""))
b.append(code("""
plt.contourf(X, Y, Z, levels=20, cmap="viridis")
plt.title("Filled contours of the bump")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="f value")
plt.show()
"""))

b.append(md(r"""
## 5. A 3D surface: `ax.plot_surface`

The same `X, Y, Z` can be drawn as a true surface. 3D drawing needs an axes
created with `projection="3d"`. (Behind the scenes this uses
`mpl_toolkits.mplot3d`; modern Matplotlib enables it automatically when you ask
for the 3D projection.)
"""))
b.append(code("""
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")   # make a 3D axes

# Draw the surface; cmap colors it by height.
surf = ax.plot_surface(X, Y, Z, cmap="viridis")

ax.set_title("Surface z = exp(-(x^2 + y^2))")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.colorbar(surf, shrink=0.6, label="z")
plt.show()
"""))

b.append(md(r"""
A contour plot is really this surface viewed straight from above. Contours are
cheap, exact, and easy to read; the 3D view is good for an intuitive first look.
"""))

b.append(md(r"""
## 6. A vector field: `plt.quiver` (gradients)

A **vector field** attaches an arrow to each point. A natural example is the
**gradient** $\nabla f = \left(\dfrac{\partial f}{\partial x},
\dfrac{\partial f}{\partial y}\right)$, which points in the direction of
steepest increase. For our bump,
$$\nabla f = \big(-2x\,f,\; -2y\,f\big).$$
We compute it on a *coarse* grid (too many arrows is unreadable) and draw it
with `quiver`.
"""))
b.append(code("""
# Coarse grid just for the arrows.
xs = np.linspace(-2, 2, 16)
ys = np.linspace(-2, 2, 16)
Xc, Yc = np.meshgrid(xs, ys)
Fc = np.exp(-(Xc**2 + Yc**2))

U = -2 * Xc * Fc    # df/dx
V = -2 * Yc * Fc    # df/dy

plt.figure()                                  # fresh 2D figure
plt.contour(X, Y, Z, levels=12, alpha=0.5)   # contours for context
plt.quiver(Xc, Yc, U, V, color="black")       # arrows = gradient
plt.title("Gradient field of the bump (arrows point uphill)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
"""))

b.append(md(r"""
Notice the arrows all point **inward toward the peak** at the origin and shrink
to nothing far away, where the bump is flat. Following them uphill is exactly
what *gradient ascent* does.
"""))

b.append(md(r"""
## 7. Heatmap of a matrix: `plt.imshow`

`imshow` draws a 2D array as a grid of colored pixels — perfect for visualising
a matrix. Each entry becomes one colored cell; `colorbar` gives the scale.
"""))
b.append(code("""
# A 6x6 matrix whose entry (i, j) is i * j (a multiplication table).
i = np.arange(6).reshape(6, 1)   # column 0..5
j = np.arange(6).reshape(1, 6)   # row    0..5
M = i * j                        # outer product via broadcasting

plt.imshow(M, cmap="magma")
plt.title("Heatmap of the matrix M[i, j] = i * j")
plt.xlabel("column j")
plt.ylabel("row i")
plt.colorbar(label="value")
plt.show()
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Build a grid on $[-3, 3] \times [-3, 3]$ and draw **filled contours** of the
saddle function
$$f(x, y) = x^2 - y^2.$$
Add a colorbar and axis labels. (You should see a saddle: high along the
$x$-axis, low along the $y$-axis.)
"""))
b.append(md("**Solution:**"))
b.append(code("""
xs = np.linspace(-3, 3, 200)
ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = X**2 - Y**2          # the saddle

plt.contourf(X, Y, Z, levels=25, cmap="coolwarm")
plt.title("Saddle f(x, y) = x^2 - y^2")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="f value")
plt.show()
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

For the same saddle $f(x, y) = x^2 - y^2$, draw its gradient field with
`quiver` on a coarse $15 \times 15$ grid. The gradient is
$$\nabla f = (2x,\; -2y).$$
Overlay it on the contour lines for context.
"""))
b.append(md("**Solution:**"))
b.append(code("""
# Fine grid for contours.
xf = np.linspace(-3, 3, 200)
Xf, Yf = np.meshgrid(xf, xf)
Zf = Xf**2 - Yf**2

# Coarse grid for arrows.
xc = np.linspace(-3, 3, 15)
Xc, Yc = np.meshgrid(xc, xc)
U = 2 * Xc      # df/dx
V = -2 * Yc     # df/dy

plt.contour(Xf, Yf, Zf, levels=15, alpha=0.5)
plt.quiver(Xc, Yc, U, V, color="black")
plt.title("Gradient field of the saddle")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Plot filled contours of $f(x, y) = \sin(x)\cos(y)$ on
   $[-\pi, \pi] \times [-\pi, \pi]$. Add a colorbar.
2. Draw the 3D surface of $g(x, y) = x^2 + y^2$ (a bowl) on
   $[-2, 2]^2$ with `projection="3d"`. Label all three axes.
3. The gradient of the bowl $g = x^2 + y^2$ is $\nabla g = (2x, 2y)$. Draw its
   `quiver` field on a coarse grid and confirm the arrows point **outward**,
   away from the minimum at the origin.
4. Make a random $10 \times 10$ matrix with `rng.normal(0, 1, size=(10, 10))`
   and display it with `imshow` and a colorbar. Try two different `cmap`
   choices (e.g. `"gray"` and `"coolwarm"`).
"""))

save(os.path.join(CH, "02b_surfaces_contours_fields.ipynb"), b)
