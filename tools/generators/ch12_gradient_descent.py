"""Generator for Chapter 12 — Gradient Descent & Optimization.

Run from anywhere:  python tools/generators/ch12_gradient_descent.py
Produces one notebook in 12-gradient-descent-optimization/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "12-gradient-descent-optimization")


# ---------------------------------------------------------------------------
# Notebook 12 — Gradient Descent & Optimization
# ---------------------------------------------------------------------------
nb = []

nb.append(md(r"""
# Chapter 12 — Gradient Descent & Optimization

> **The workhorse of machine learning.** In Chapter 06 you minimized functions
> of one variable, and in Chapter 07 you met the **gradient** $\nabla J$ — the
> vector that points uphill on a loss landscape. This chapter turns that idea
> into the single algorithm that trains almost every model: **gradient descent**.

The whole story is one update rule, repeated:

$$\boxed{\;\theta \;\leftarrow\; \theta \;-\; \eta\,\nabla J(\theta)\;}$$

Read it as *"stand on the loss surface, look for the downhill direction
$-\nabla J$, and take a small step of size $\eta$ in that direction."* Repeat
until you reach the bottom. That is the entire idea.

In this notebook we will:

- run gradient descent on a simple 1D convex function and **plot the path**;
- study the **learning rate** $\eta$ — too small (slow), good, too large (diverges);
- apply gradient descent to **linear regression** built from scratch, and confirm
  it converges to the closed-form solution (the *normal equation*);
- see why **feature scaling** dramatically speeds convergence (round vs stretched
  contours);
- implement **batch**, **stochastic**, and **mini-batch** gradient descent and
  compare their loss curves;
- add **momentum** to accelerate descent (optional).

Run every code cell with **Shift + Enter**. Edit and re-run freely — you cannot
break anything.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 0. Setup

We use only **NumPy** (fast array math) and **Matplotlib** (plots). We fix the
randomness with `np.random.default_rng(0)` so every run looks identical.
"""))
nb.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)                  # reproducible randomness
np.set_printoptions(precision=4, suppress=True) # tidy array printing
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 1. The update rule: walk downhill

Imagine a hiker in fog on a hillside, wanting to reach the lowest valley. They
cannot see far, but they *can* feel the slope under their feet. A sensible plan:
feel which way is steepest **downhill**, take a small step that way, and repeat.

That is gradient descent. The "slope under your feet" is the gradient
$\nabla J(\theta)$ (which points *uphill*), so we step the **opposite** way:

$$\theta_{k+1} = \theta_k - \eta\,\nabla J(\theta_k).$$

- $\theta$ — the parameters we are tuning (here just one number; later a vector).
- $J(\theta)$ — the **cost** (or *loss*): how wrong the model is. Smaller is better.
- $\nabla J(\theta)$ — the gradient: the steepest-uphill direction.
- $\eta$ (eta) — the **learning rate**: how big a step we take.

Let's start with the simplest possible landscape, a 1D bowl (parabola):

$$J(\theta) = (\theta - 3)^2, \qquad J'(\theta) = 2(\theta - 3).$$

Its minimum is obviously at $\theta = 3$. Let's make gradient descent *discover*
that, pretending we don't know the answer.
"""))
nb.append(code("""
def J(theta):
    return (theta - 3.0) ** 2        # cost: a bowl with minimum at theta = 3

def dJ(theta):
    return 2.0 * (theta - 3.0)       # derivative (the 1D gradient)

print("J(3) =", J(3.0))             # 0.0  -> bottom of the bowl
print("J(0) =", J(0.0))             # 9.0
print("J'(0) =", dJ(0.0))           # -6  -> negative slope: downhill is to the RIGHT
"""))

nb.append(md(r"""
At $\theta = 0$ the derivative is $-6$ (negative), meaning the function is
*decreasing* — downhill is to the right. The update $\theta - \eta\,J'(\theta)$
subtracts a negative number, so $\theta$ moves right, toward the minimum.
Exactly what we want.
"""))
nb.append(code("""
def gradient_descent_1d(dcost, start, lr=0.1, n_steps=20):
    \"\"\"Run 1D gradient descent; return every theta visited (the 'path').\"\"\"
    theta = float(start)
    path = [theta]
    for _ in range(n_steps):
        theta = theta - lr * dcost(theta)   # <-- the one rule that matters
        path.append(theta)
    return np.array(path)

path = gradient_descent_1d(dJ, start=0.0, lr=0.1, n_steps=20)
print("start :", path[0])
print("end   :", path[-1], " (should be near the minimum theta = 3)")
"""))

nb.append(md(r"""
### Plotting the path

Left: the parabola with each visited point marked — watch the iterates slide
down into the valley. Right: the **loss curve**, $J$ at each iteration, dropping
toward zero. In real training you stare at exactly this curve to judge progress.
"""))
nb.append(code("""
ths = np.linspace(-1, 7, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: the bowl with the descent path
ax1.plot(ths, J(ths), color="steelblue", label="J(theta)")
ax1.plot(path, J(path), "o-", color="crimson", markersize=5, label="GD steps")
ax1.plot(3, 0, "k*", markersize=15, label="true minimum")
ax1.set_title("Walking downhill on J(theta) = (theta - 3)^2")
ax1.set_xlabel("theta"); ax1.set_ylabel("J(theta)")
ax1.legend(); ax1.grid(True)

# Right: loss vs iteration
ax2.plot(J(path), "o-", color="navy", markersize=5)
ax2.set_title("Loss vs iteration")
ax2.set_xlabel("iteration"); ax2.set_ylabel("J(theta)")
ax2.grid(True)

plt.tight_layout()
plt.show()
"""))

nb.append(md(r"""
Notice the steps are **big when far from the bottom** (steep slope $\Rightarrow$
large gradient $\Rightarrow$ large step) and **shrink near the minimum** (the
slope flattens). Gradient descent automatically slows down as it arrives — no
special handling needed.
"""))

# ---- Exercise 1 ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Minimize $J(\theta) = (\theta + 5)^2 + 2$ with gradient descent.

1. Write its derivative $J'(\theta)$ by hand.
2. Run `gradient_descent_1d` from `start = 10.0` with `lr = 0.1` for 30 steps.
3. Where does it end up? What is the smallest value of $J$?
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
# J'(theta) = 2*(theta + 5); minimum is at theta = -5, where J = 2.
def dJ2(theta):
    return 2.0 * (theta + 5.0)

p = gradient_descent_1d(dJ2, start=10.0, lr=0.1, n_steps=30)
print("end theta :", p[-1], " (should be near -5)")
print("min J     :", (p[-1] + 5.0) ** 2 + 2.0, " (should be near 2)")
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 2. The learning rate $\eta$: slow, good, and diverging

The learning rate is the single most important knob in gradient descent. Using
the same bowl $J(\theta) = (\theta-3)^2$ and the same start, we compare three
choices:

- **too small** ($\eta = 0.01$) — each step barely moves; convergence crawls;
- **good** ($\eta = 0.1$) — smooth, fast descent to the minimum;
- **too large** ($\eta = 1.01$) — the steps *overshoot* the minimum and grow,
  so the loss **diverges** (blows up).

For this quadratic the update is $\theta_{k+1} - 3 = (1 - 2\eta)(\theta_k - 3)$,
so the distance to the minimum is multiplied by $|1 - 2\eta|$ each step. That is
why $\eta < 1$ shrinks the error but $\eta > 1$ makes it explode.
"""))
nb.append(code("""
fig, ax = plt.subplots(figsize=(7.5, 4.5))

for lr, name in [(0.01, "too small"), (0.1, "good"), (1.01, "too large")]:
    p = gradient_descent_1d(dJ, start=0.0, lr=lr, n_steps=25)
    losses = J(p)
    ax.plot(losses, "o-", markersize=4, label=f"lr = {lr} ({name})")

ax.set_title("Effect of the learning rate (overlaid loss curves)")
ax.set_xlabel("iteration"); ax.set_ylabel("loss J(theta)")
ax.set_yscale("log")        # log scale: the diverging curve would dwarf the others
ax.legend(); ax.grid(True)
plt.show()
"""))

nb.append(md(r"""
On the log scale the picture is clear:

- the **too-small** curve drifts down gently (it would get there *eventually*);
- the **good** curve plunges straight to a tiny loss;
- the **too-large** curve climbs — the loss is *increasing*, a sure sign $\eta$
  is past the stability limit.

Tuning $\eta$ is a daily concern in machine learning. A common workflow: start
small enough to be stable, then increase it as far as you can before it diverges.
"""))

# ---- Exercise 2 ----
nb.append(md(r"""
---
## ✍️ Exercise 2 (solution included)

For our bowl, the error is multiplied by $|1 - 2\eta|$ each step, so descent is
stable only when $|1 - 2\eta| < 1$, i.e. $0 < \eta < 1$.

1. Predict: does $\eta = 0.5$ converge, and if so, how fast?
2. Run it from `start = 0.0` for 10 steps and print the path to check.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
# At eta = 0.5 the factor |1 - 2*eta| = 0, so it lands EXACTLY on the minimum
# in a single step!
p = gradient_descent_1d(dJ, start=0.0, lr=0.5, n_steps=10)
print(p)            # 0, then 3, 3, 3, ...  -> one step to the bottom
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 3. Gradient descent for linear regression (from scratch)

Now a real model. **Linear regression** fits a straight line $y \approx w x + b$
to data by choosing the slope $w$ and intercept $b$ that minimize the
**mean squared error (MSE)** cost:

$$J(w, b) = \frac{1}{n}\sum_{i=1}^{n}\big(w x_i + b - y_i\big)^2.$$

This is a function of **two** parameters, so $\theta = (w, b)$ and we need the
*gradient* (Chapter 07). Differentiating the MSE:

$$\frac{\partial J}{\partial w} = \frac{2}{n}\sum_i (w x_i + b - y_i)\,x_i,
\qquad
\frac{\partial J}{\partial b} = \frac{2}{n}\sum_i (w x_i + b - y_i).$$

Let's manufacture some noisy data from a known line so we know the right answer.
"""))
nb.append(code("""
# True line: y = 2*x + 1, plus a little noise
n = 80
x = rng.uniform(-3, 3, size=n)
y = 2.0 * x + 1.0 + rng.normal(0, 1.0, size=n)   # noisy observations

plt.figure(figsize=(6, 4))
plt.scatter(x, y, s=18, alpha=0.7, label="data")
plt.title("Noisy data from y = 2x + 1")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
### The closed-form target (normal equation)

Linear regression is special: its minimum can be found *exactly* with a formula,
the **normal equation** from linear algebra. If we stack a column of ones next to
$x$ to form the design matrix $X$ (so the intercept is just another weight), the
best parameters are

$$\theta^\star = (X^\top X)^{-1} X^\top y.$$

We will use this as the **target**: gradient descent should converge to it.
"""))
nb.append(code("""
# Design matrix: column of x's and a column of ones (for the intercept b)
X = np.column_stack([x, np.ones_like(x)])      # shape (n, 2): columns [x, 1]

# Normal equation (closed-form least-squares solution)
theta_star = np.linalg.solve(X.T @ X, X.T @ y)
print("closed-form  w, b =", theta_star)        # should be near [2, 1]
"""))

nb.append(md(r"""
### Now solve the same problem with gradient descent

We write the cost and its gradient using the design matrix $X$, then run the
**same** update rule $\theta \leftarrow \theta - \eta\,\nabla J(\theta)$.
"""))
nb.append(code("""
def cost(theta, X, y):
    residual = X @ theta - y           # prediction minus truth, for every point
    return np.mean(residual ** 2)      # mean squared error

def grad_cost(theta, X, y):
    n = len(y)
    residual = X @ theta - y
    return (2.0 / n) * (X.T @ residual)  # gradient [dJ/dw, dJ/db]

def gradient_descent(grad, theta0, X, y, lr=0.1, n_steps=100):
    \"\"\"Full-batch gradient descent; return final theta and the cost history.\"\"\"
    theta = np.array(theta0, dtype=float)
    history = []
    for _ in range(n_steps):
        history.append(cost(theta, X, y))
        theta = theta - lr * grad(theta, X, y)   # the same one rule
    return theta, np.array(history)

theta_gd, hist = gradient_descent(grad_cost, [0.0, 0.0], X, y, lr=0.1, n_steps=100)
print("gradient descent w, b =", theta_gd)
print("closed-form      w, b =", theta_star)
print("difference            =", np.abs(theta_gd - theta_star))
"""))

nb.append(md(r"""
The two agree closely — gradient descent **converged to the closed-form
solution** without ever using the formula. That is the point: for linear
regression we *could* use the normal equation, but for almost every other model
(logistic regression, neural networks) no formula exists, and gradient descent
is the only practical option.
"""))
nb.append(code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: data with both fitted lines (they should overlap)
xs = np.linspace(-3, 3, 50)
ax1.scatter(x, y, s=15, alpha=0.5, label="data")
ax1.plot(xs, theta_star[0] * xs + theta_star[1], "k-", lw=3, label="closed-form")
ax1.plot(xs, theta_gd[0] * xs + theta_gd[1], "r--", lw=2, label="gradient descent")
ax1.set_title("Fitted line"); ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend(); ax1.grid(True)

# Right: the cost coming down
ax2.plot(hist, color="navy")
ax2.set_title("Cost J(w, b) vs iteration")
ax2.set_xlabel("iteration"); ax2.set_ylabel("MSE")
ax2.grid(True)

plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 4. Feature scaling: why normalization speeds convergence

Here is a subtle but hugely practical point. The *shape* of the cost surface
controls how fast gradient descent moves. When features live on very different
scales, the contours of $J$ become long, thin **ellipses** (a stretched valley).
Gradient descent then zig-zags slowly across the narrow direction. When features
are **scaled** to similar ranges, the contours become nearly **circular**, and
descent heads almost straight to the bottom.

Let's build a regression with two features on wildly different scales — one in
the thousands, one near 1 — and look at the contour shapes.
"""))
nb.append(code("""
# Two features: x1 ~ thousands (e.g. house size), x2 ~ small (e.g. # bedrooms)
m = 200
x1 = rng.uniform(1000, 3000, size=m)     # large scale
x2 = rng.uniform(1, 5, size=m)           # small scale
y2 = 3.0 * x1 + 50.0 * x2 + rng.normal(0, 100, size=m)

# Raw design matrix (no intercept here, to focus on the two-feature shape)
A_raw = np.column_stack([x1, x2])

# Standardize each column: subtract mean, divide by standard deviation
mu  = A_raw.mean(axis=0)
sig = A_raw.std(axis=0)
A_scaled = (A_raw - mu) / sig

print("raw    feature means :", A_raw.mean(axis=0))
print("raw    feature stds  :", A_raw.std(axis=0))
print("scaled feature means :", A_scaled.mean(axis=0), " (~0)")
print("scaled feature stds  :", A_scaled.std(axis=0), " (~1)")
"""))

nb.append(md(r"""
Let's draw the cost contours for the two-parameter problem $J(w_1, w_2)$, once
with raw features and once with scaled features, around their respective optima.
"""))
nb.append(code("""
def mse_surface(A, target, w1_grid, w2_grid):
    \"\"\"Compute J(w1, w2) over a grid of weight values (no intercept).\"\"\"
    Z = np.zeros_like(w1_grid)
    for i in range(w1_grid.shape[0]):
        for j in range(w1_grid.shape[1]):
            w = np.array([w1_grid[i, j], w2_grid[i, j]])
            Z[i, j] = np.mean((A @ w - target) ** 2)
    return Z

# Optimal weights for each version (closed form) to center the plots
w_raw    = np.linalg.solve(A_raw.T @ A_raw,       A_raw.T @ y2)
w_scaled = np.linalg.solve(A_scaled.T @ A_scaled, A_scaled.T @ y2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# RAW: contours are extremely elongated (note the tiny w1 range)
g1 = np.linspace(w_raw[0] - 0.5,  w_raw[0] + 0.5,  60)
g2 = np.linspace(w_raw[1] - 4000, w_raw[1] + 4000, 60)
W1, W2 = np.meshgrid(g1, g2)
Zraw = mse_surface(A_raw, y2, W1, W2)
ax1.contour(W1, W2, Zraw, levels=20)
ax1.plot(w_raw[0], w_raw[1], "r*", markersize=14)
ax1.set_title("RAW features: stretched (elongated) contours")
ax1.set_xlabel("w1"); ax1.set_ylabel("w2")

# SCALED: contours are nearly circular
g1 = np.linspace(w_scaled[0] - 400, w_scaled[0] + 400, 60)
g2 = np.linspace(w_scaled[1] - 400, w_scaled[1] + 400, 60)
W1, W2 = np.meshgrid(g1, g2)
Zsc = mse_surface(A_scaled, y2, W1, W2)
ax2.contour(W1, W2, Zsc, levels=20)
ax2.plot(w_scaled[0], w_scaled[1], "r*", markersize=14)
ax2.set_title("SCALED features: round contours")
ax2.set_xlabel("w1"); ax2.set_ylabel("w2")

plt.tight_layout()
plt.show()
"""))

nb.append(md(r"""
The raw-feature contours are absurdly stretched (look at the axis ranges: $w_1$
spans $\pm 0.5$ while $w_2$ spans $\pm 4000$). Gradient descent on such a surface
must use a tiny learning rate or it diverges along the steep direction, so it
crawls along the shallow one. After **standardizing** (subtract mean, divide by
standard deviation) the contours are round, and descent converges in far fewer
steps. **Always scale your features** before running gradient descent.
"""))

# ---- Exercise 3 ----
nb.append(md(r"""
---
## ✍️ Exercise 3 (solution included)

Demonstrate the speed-up numerically. Run `gradient_descent` (reusing the cost
helpers from Section 3) on the **raw** features `A_raw` with a learning rate that
is just barely stable, and on the **scaled** features `A_scaled` with `lr = 0.3`.
Compare how low the cost gets after 60 iterations.

*Hint:* for the raw features you must use a minuscule learning rate (try
`lr = 2e-7`) because of the huge feature scale.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
# Raw features need a tiny lr just to stay stable, and still barely move.
_, hist_raw = gradient_descent(grad_cost, [0.0, 0.0], A_raw, y2,
                               lr=2e-7, n_steps=60)
# Scaled features tolerate a healthy lr and converge fast.
_, hist_sc  = gradient_descent(grad_cost, [0.0, 0.0], A_scaled, y2,
                               lr=0.3,  n_steps=60)

print("raw    cost after 60 steps :", hist_raw[-1])
print("scaled cost after 60 steps :", hist_sc[-1])

plt.figure(figsize=(7, 4))
plt.plot(hist_raw, label="raw features (lr=2e-7)")
plt.plot(hist_sc,  label="scaled features (lr=0.3)")
plt.yscale("log")
plt.title("Feature scaling drastically speeds convergence")
plt.xlabel("iteration"); plt.ylabel("cost (log scale)")
plt.legend(); plt.grid(True)
plt.show()
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 5. Batch vs stochastic vs mini-batch gradient descent

So far each update used **all** the data to compute the gradient. That is
**batch** gradient descent: accurate but expensive when $n$ is large (millions of
points). Two cheaper variants trade accuracy for speed:

- **Batch GD** — gradient over the *whole* dataset each step. Smooth but slow per
  step.
- **Stochastic GD (SGD)** — gradient from **one random point** per step. Very
  cheap and fast, but noisy: the cost curve jitters.
- **Mini-batch GD** — gradient from a small random **batch** (e.g. 16 points).
  The practical sweet spot, and what real deep-learning training uses.

We implement all three on the linear-regression data from Section 3 (the design
matrix `X` and targets `y`) and compare their loss curves.
"""))
nb.append(code("""
def sgd(grad, theta0, X, y, lr=0.1, n_epochs=30, batch_size=1, rng=rng):
    \"\"\"Mini-batch gradient descent.

    batch_size = 1            -> stochastic GD (one point per step)
    batch_size = len(y)       -> batch GD (whole dataset per step)
    anything in between       -> mini-batch GD
    One 'epoch' = one full sweep through the shuffled data.
    \"\"\"
    theta = np.array(theta0, dtype=float)
    n = len(y)
    history = []                                   # cost on FULL data per epoch
    for _ in range(n_epochs):
        order = rng.permutation(n)                 # reshuffle each epoch
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]  # the current mini-batch
            theta = theta - lr * grad(theta, X[idx], y[idx])
        history.append(cost(theta, X, y))
    return theta, np.array(history)

# Run all three (fresh rng each time so the comparison is fair)
t_batch, h_batch = sgd(grad_cost, [0., 0.], X, y, lr=0.1, n_epochs=30,
                       batch_size=len(y), rng=np.random.default_rng(1))
t_mini,  h_mini  = sgd(grad_cost, [0., 0.], X, y, lr=0.1, n_epochs=30,
                       batch_size=16,      rng=np.random.default_rng(1))
t_sgd,   h_sgd   = sgd(grad_cost, [0., 0.], X, y, lr=0.1, n_epochs=30,
                       batch_size=1,       rng=np.random.default_rng(1))

print("closed-form :", theta_star)
print("batch       :", t_batch)
print("mini-batch  :", t_mini)
print("stochastic  :", t_sgd)
"""))

nb.append(code("""
plt.figure(figsize=(7.5, 4.5))
plt.plot(h_batch, "o-", markersize=3, label="batch (n=80)")
plt.plot(h_mini,  "o-", markersize=3, label="mini-batch (16)")
plt.plot(h_sgd,   "o-", markersize=3, label="stochastic (1)")
plt.title("Batch vs mini-batch vs stochastic GD")
plt.xlabel("epoch"); plt.ylabel("cost on full data")
plt.yscale("log"); plt.legend(); plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
All three reach essentially the same answer, but their *paths* differ:

- **batch** descends smoothly (every step uses the exact gradient);
- **stochastic** drops fast at first but stays jittery — each step uses a noisy
  one-point estimate that wanders around the true minimum;
- **mini-batch** combines the best of both: fast early progress and a steadier
  curve. This is why virtually all modern training uses mini-batches.

The noise in SGD is not purely a nuisance: in complicated (non-convex) loss
landscapes it can actually help the optimizer escape bad spots.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 6. Momentum (optional): adding inertia

Plain gradient descent can crawl through long, shallow valleys, zig-zagging
across the narrow direction. **Momentum** fixes this by letting the update build
up speed, like a heavy ball rolling downhill that keeps some of its previous
velocity:

$$v \leftarrow \beta\,v + \nabla J(\theta), \qquad
  \theta \leftarrow \theta - \eta\, v.$$

The velocity $v$ accumulates consistent gradient directions (accelerating along
the valley) while averaging out the oscillating ones. $\beta \in [0, 1)$ (often
$0.9$) controls how much past velocity is retained.
"""))
nb.append(code("""
def gd_momentum(grad, theta0, X, y, lr=0.1, beta=0.9, n_steps=100):
    theta = np.array(theta0, dtype=float)
    v = np.zeros_like(theta)            # velocity starts at rest
    history = []
    for _ in range(n_steps):
        history.append(cost(theta, X, y))
        v = beta * v + grad(theta, X, y)   # accumulate velocity
        theta = theta - lr * v             # step along the velocity
    return theta, np.array(history)

# Compare plain GD with momentum on the same regression problem
_, hist_plain = gradient_descent(grad_cost, [0., 0.], X, y, lr=0.05, n_steps=60)
_, hist_mom   = gd_momentum(grad_cost, [0., 0.], X, y, lr=0.05, beta=0.9, n_steps=60)

plt.figure(figsize=(7, 4))
plt.plot(hist_plain, label="plain GD (lr=0.05)")
plt.plot(hist_mom,   label="momentum (lr=0.05, beta=0.9)")
plt.yscale("log")
plt.title("Momentum accelerates convergence")
plt.xlabel("iteration"); plt.ylabel("cost (log scale)")
plt.legend(); plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
With the **same** learning rate, momentum reaches a low cost in far fewer steps.
Modern optimizers like *Adam* combine momentum with per-parameter learning-rate
scaling, but the core idea is exactly the heavy-ball intuition above.
"""))

# ---- Homework ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **A quartic landscape.** Minimize $J(\theta) = \theta^4 - 3\theta^2 + \theta$
   with 1D gradient descent (derive $J'$ by hand). This function has **two**
   local minima — run descent from several starting points and report which
   minimum each one finds. What does this say about non-convex optimization?

2. **Find the stability limit.** For the bowl $J(\theta) = (\theta - 3)^2$, the
   theory says descent diverges once $\eta \ge 1$. Confirm experimentally:
   sweep $\eta$ over `[0.2, 0.5, 0.9, 0.99, 1.0, 1.1]`, run 30 steps from
   $\theta = 0$, and plot the final loss against $\eta$.

3. **Regression with an intercept, scaled.** Take the two-feature data from
   Section 4, standardize the features, add a column of ones for the intercept,
   and fit all three parameters with mini-batch gradient descent. Confirm the
   result matches the normal equation on the scaled design matrix.

4. **Tune momentum.** On the Section 3 regression problem, fix `lr = 0.05` and
   try $\beta \in \{0, 0.5, 0.9, 0.99\}$ with `gd_momentum`. Plot all four loss
   curves on one axis. Which $\beta$ is fastest? What goes wrong when $\beta$ is
   too close to 1?
"""))

nb.append(md(r"""
## What you learned

- Gradient descent is one rule repeated: $\theta \leftarrow \theta - \eta\,\nabla J(\theta)$
  — *take a small step downhill on the loss surface*.
- The **learning rate** $\eta$ is critical: too small crawls, too large diverges.
- Built from scratch, gradient descent on the **MSE cost** converges to the
  **closed-form** (normal-equation) solution of linear regression.
- **Feature scaling** turns stretched contours into round ones, hugely speeding
  convergence — always standardize first.
- **Batch / stochastic / mini-batch** trade accuracy per step for speed;
  mini-batch is the practical default.
- **Momentum** adds inertia to accelerate descent through shallow valleys.

This is the engine under essentially all of machine learning. Every later model
you train — logistic regression, neural networks — is this same loop with a
different cost $J$ and gradient.
"""))

save(os.path.join(CH, "12_gradient_descent.ipynb"), nb)
