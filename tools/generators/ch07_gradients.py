"""Generator for Chapter 07 — Multivariate Calculus & Gradients.

Run from anywhere:  python tools/generators/ch07_gradients.py
Produces one notebook in 07-multivariate-calculus-gradients/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "07-multivariate-calculus-gradients")


# ---------------------------------------------------------------------------
# Notebook 07 — Multivariate Calculus & Gradients
# ---------------------------------------------------------------------------
nb = []

nb.append(md(r"""
# Chapter 07 — Multivariate Calculus & Gradients

> **The engine of learning.** Almost every machine-learning model is trained by
> the same recipe: write down a function that measures *how wrong* the model is,
> then repeatedly nudge the parameters *downhill* on that function. The compass
> that tells us which way is downhill is the **gradient**.

In Chapter 06 you differentiated functions of one variable, $f(x)$. Real models
have *many* knobs, so their loss is a function of many variables,
$f(x_1, x_2, \dots, x_n)$. This chapter builds the multivariable toolkit:

- functions of two variables $f(x,y)$, and how to *see* them;
- **partial derivatives** $\partial f/\partial x$ and $\partial f/\partial y$;
- the **gradient** $\nabla f$, a vector that points in the steepest-uphill direction;
- a tiny `numerical_gradient` helper built from finite differences;
- **directional derivatives** and the **chain rule**;
- the **Jacobian** of a vector-valued function;
- and finally **gradient descent** in 2D — exactly how models learn.

Run every code cell with **Shift + Enter**. Edit and re-run freely.
"""))

nb.append(md(r"""
## 0. Setup

We use only **NumPy** (fast array math) and **Matplotlib** (plots). We fix a
random seed with `np.random.default_rng(0)` so every run looks the same.
"""))
nb.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible randomness
np.set_printoptions(precision=4, suppress=True)  # tidy array printing
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 1. Functions of two variables

A function of two variables takes a point $(x, y)$ in the plane and returns a
single number — a *height*. Picture a landscape: at each ground position
$(x, y)$ there is an altitude $f(x, y)$.

Our running example is a simple bowl (a paraboloid), tilted a little:

$$f(x, y) = x^2 + 2y^2$$

In Python a multivariable function is still just a `def`. Thanks to NumPy it
works on single numbers **and** on whole arrays of numbers at once.
"""))
nb.append(code("""
def f(x, y):
    return x**2 + 2 * y**2     # a bowl: minimum at the origin (0, 0)

print(f(0.0, 0.0))   # 0.0  -> bottom of the bowl
print(f(1.0, 1.0))   # 3.0
print(f(2.0, 0.0))   # 4.0
"""))

nb.append(md(r"""
### Building a grid with `np.meshgrid`

To draw the landscape we need to evaluate $f$ at many $(x, y)$ points laid out
on a grid. `np.meshgrid` takes the $x$-axis and $y$-axis tick values and returns
two 2D arrays, `X` and `Y`, holding the coordinates of *every* grid point.
"""))
nb.append(code("""
xs = np.linspace(-3, 3, 5)      # 5 sample points on the x-axis
ys = np.linspace(-3, 3, 5)      # 5 sample points on the y-axis
X, Y = np.meshgrid(xs, ys)      # X[i,j], Y[i,j] = coordinates of grid point (i,j)

print("X =\\n", X)
print("Y =\\n", Y)

Z = f(X, Y)                     # f evaluated at EVERY grid point, all at once
print("Z = f(X, Y) =\\n", Z)
"""))

nb.append(md(r"""
### Seeing the landscape: contour plot + 3D surface

A **contour plot** is a topographic map: each curve joins points of equal
height (like elevation lines on a hiking map). A **3D surface** shows the same
thing as an actual landscape. We use a finer grid now (more points = smoother
picture).
"""))
nb.append(code("""
# A fine grid for smooth pictures
xs = np.linspace(-3, 3, 200)
ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

fig = plt.figure(figsize=(11, 4.5))

# Left: contour map (top-down view)
ax1 = fig.add_subplot(1, 2, 1)
cs = ax1.contour(X, Y, Z, levels=15)        # 15 elevation lines
ax1.clabel(cs, inline=True, fontsize=8)     # label the heights
ax1.set_title("Contour map of f(x, y) = x^2 + 2y^2")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_aspect("equal")

# Right: 3D surface (the actual landscape)
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9)
ax2.set_title("3D surface")
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("f")

plt.tight_layout()
plt.show()
"""))

nb.append(md(r"""
The contours are ellipses (the bowl is steeper in the $y$ direction because of
the factor $2$). The single lowest point — the bottom of the bowl — sits at the
origin. Training a model means *finding* that lowest point.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 2. Partial derivatives

A **partial derivative** asks: *if I wiggle only one variable and freeze the
others, how fast does $f$ change?*

- $\dfrac{\partial f}{\partial x}$ — slope as we move in the $x$ direction,
  holding $y$ fixed.
- $\dfrac{\partial f}{\partial y}$ — slope in the $y$ direction, holding $x$
  fixed.

You compute it with ordinary one-variable rules, treating the *other* variable
as a constant. For $f(x,y) = x^2 + 2y^2$:

$$\frac{\partial f}{\partial x} = 2x, \qquad \frac{\partial f}{\partial y} = 4y.$$
"""))
nb.append(code("""
def dfdx(x, y):
    return 2 * x       # partial derivative with respect to x

def dfdy(x, y):
    return 4 * y       # partial derivative with respect to y

# At the point (1, 1):
print("df/dx at (1,1) =", dfdx(1.0, 1.0))   # 2
print("df/dy at (1,1) =", dfdy(1.0, 1.0))   # 4
"""))

nb.append(md(r"""
### Checking with finite differences

Even if we did not know the formula, we could *estimate* a partial derivative
the same way as in one variable: nudge the input by a tiny $h$ and measure the
change. The symmetric (central) difference is accurate:

$$\frac{\partial f}{\partial x} \approx \frac{f(x+h,\, y) - f(x-h,\, y)}{2h}.$$
"""))
nb.append(code("""
h = 1e-5
x0, y0 = 1.0, 1.0

approx_dfdx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
approx_dfdy = (f(x0, y0 + h) - f(x0, y0 - h)) / (2 * h)

print("numeric df/dx =", approx_dfdx, " vs exact", dfdx(x0, y0))
print("numeric df/dy =", approx_dfdy, " vs exact", dfdy(x0, y0))
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 3. The gradient vector $\nabla f$

Collect the partial derivatives into a single vector. That vector is the
**gradient**:

$$\nabla f(x, y) = \left( \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y} \right).$$

It is the central object of this whole course, because (as we will see):

> **The gradient points in the direction of steepest ascent**, and its length
> is how steep that climb is. To go *downhill* — toward a smaller loss — we step
> in the direction $-\nabla f$.
"""))
nb.append(code("""
def grad_f(x, y):
    # return the gradient as a NumPy array [df/dx, df/dy]
    return np.array([2 * x, 4 * y])

print("grad f at (1, 1) =", grad_f(1.0, 1.0))   # [2, 4]
print("grad f at (0, 0) =", grad_f(0.0, 0.0))   # [0, 0]  -> flat: the minimum!
"""))

nb.append(md(r"""
Notice $\nabla f = (0, 0)$ at the bottom of the bowl. **A zero gradient means a
flat spot** — a minimum, maximum, or saddle. Optimization is, at heart, the
search for a point where the gradient vanishes.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 4. A reusable `numerical_gradient` helper

For complicated functions, deriving partials by hand is tedious and
error-prone. Let's package the finite-difference idea into one helper that works
for a function of **any number** of variables. We pass the point as a NumPy
array and nudge each coordinate in turn.
"""))
nb.append(code("""
def numerical_gradient(func, point, h=1e-5):
    \"\"\"Estimate the gradient of `func` at `point` using central differences.

    func  : a function taking a 1D array and returning one number
    point : a 1D NumPy array (the location where we want the gradient)
    returns a 1D NumPy array, the estimated gradient.
    \"\"\"
    point = np.asarray(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(point.size):        # nudge each coordinate, one at a time
        step = np.zeros_like(point)
        step[i] = h
        grad[i] = (func(point + step) - func(point - step)) / (2 * h)
    return grad

# To use it, write f as a function of a single vector v = [x, y]:
def f_vec(v):
    return v[0]**2 + 2 * v[1]**2

print("numeric :", numerical_gradient(f_vec, [1.0, 1.0]))
print("exact   :", grad_f(1.0, 1.0))
"""))

nb.append(md(r"""
The numeric and exact gradients agree to many digits. This helper is a great
*debugging* tool: whenever you derive a gradient by hand, check it against
`numerical_gradient` — this is called **gradient checking**, and real ML
libraries do it too.
"""))

# ---- Exercise 1 ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Let $g(x, y) = \sin(x)\,\cos(y)$.

1. Work out $\partial g/\partial x$ and $\partial g/\partial y$ by hand.
2. Write `g_vec(v)` and confirm your formulas against `numerical_gradient` at
   the point $(x, y) = (1, 2)$.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
# By hand:
#   dg/dx =  cos(x) cos(y)
#   dg/dy = -sin(x) sin(y)
def g_vec(v):
    return np.sin(v[0]) * np.cos(v[1])

def grad_g_exact(x, y):
    return np.array([np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y)])

point = [1.0, 2.0]
print("numeric :", numerical_gradient(g_vec, point))
print("exact   :", grad_g_exact(1.0, 2.0))
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 5. The gradient points uphill: the gradient field

Time for the key picture. We draw the contour map again and, at a scatter of
points, attach a little arrow showing the gradient $\nabla f$ at that point
(this is a **quiver** plot — a field of arrows).

Watch what happens: **every arrow points "outward and uphill", straight across
the contour lines toward higher ground.** The arrows are longest where the
landscape is steepest.
"""))
nb.append(code("""
# fine grid for the contour background
xs = np.linspace(-3, 3, 200)
ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

# coarse grid for the arrows (we don't want 40000 arrows!)
xa = np.linspace(-3, 3, 13)
ya = np.linspace(-3, 3, 13)
XA, YA = np.meshgrid(xa, ya)
U = 2 * XA          # df/dx at every arrow location
V = 4 * YA          # df/dy at every arrow location

plt.figure(figsize=(6, 5.5))
cs = plt.contour(X, Y, Z, levels=15, alpha=0.6)
plt.clabel(cs, inline=True, fontsize=7)
plt.quiver(XA, YA, U, V, color="crimson")   # the gradient arrows
plt.title("Gradient field of f points UPHILL (steepest ascent)")
plt.xlabel("x"); plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.show()
"""))

nb.append(md(r"""
Two facts to lock in:

1. **Direction:** $\nabla f$ is *perpendicular* to the contour line through that
   point and aims toward larger $f$.
2. **Descent:** therefore $-\nabla f$ aims toward *smaller* $f$. That is the
   direction we will step when we minimize a loss.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 6. The directional derivative

The partials measure slope along the $x$- and $y$-axes. What about slope in some
*arbitrary* direction given by a unit vector $\mathbf{u} = (u_1, u_2)$ (with
$\|\mathbf{u}\| = 1$)? That is the **directional derivative**, and it is just a
dot product with the gradient:

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}
   = \frac{\partial f}{\partial x} u_1 + \frac{\partial f}{\partial y} u_2.$$

Because $\nabla f \cdot \mathbf{u} = \|\nabla f\|\cos\theta$, the slope is
**largest when $\mathbf{u}$ aligns with $\nabla f$** ($\theta = 0$) — confirming
once more that the gradient is the steepest-ascent direction.
"""))
nb.append(code("""
def directional_derivative(grad, u):
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)        # make sure u is a UNIT vector
    return grad @ u                  # @ is the dot product

g_here = grad_f(1.0, 1.0)            # gradient at (1, 1) = [2, 4]

print("slope toward +x      :", directional_derivative(g_here, [1, 0]))   # = df/dx = 2
print("slope toward +y      :", directional_derivative(g_here, [0, 1]))   # = df/dy = 4
print("slope along gradient :", directional_derivative(g_here, g_here))   # steepest = |grad|
print("|grad| (max slope)   :", np.linalg.norm(g_here))
print("slope perpendicular  :", directional_derivative(g_here, [-4, 2])) # ~0: along a contour
"""))

nb.append(md(r"""
The step "along the gradient" gives the biggest slope, exactly equal to
$\|\nabla f\|$. The step *perpendicular* to the gradient gives slope $\approx 0$
— you are walking along a contour line, staying at the same height.
"""))

# ---- Exercise 2 ----
nb.append(md(r"""
## ✍️ Exercise 2 (solution included)

At the point $(2, -1)$ on our bowl $f(x,y)=x^2+2y^2$:

1. Compute $\nabla f$.
2. Find the directional derivative in the direction $(3, 4)$.
3. Which direction gives the **most negative** slope (steepest *descent*), and
   what is that slope?
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
g = grad_f(2.0, -1.0)
print("gradient at (2,-1)        :", g)                              # [4, -4]
print("D_u f in direction (3,4)  :", directional_derivative(g, [3, 4]))

# Steepest DESCENT is the direction -grad / |grad|; its slope is -|grad|.
steepest_descent_dir = -g / np.linalg.norm(g)
print("steepest-descent direction:", steepest_descent_dir)
print("most negative slope       :", -np.linalg.norm(g))             # = -|grad|
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 7. The chain rule (multivariable)

Suppose $x$ and $y$ themselves depend on a single parameter $t$ — say a particle
moving along a path $\big(x(t), y(t)\big)$ — and we care about how the height
$f$ changes as the particle moves. The **chain rule** says:

$$\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt}
                 + \frac{\partial f}{\partial y}\frac{dy}{dt}
               = \nabla f \cdot \left(\frac{dx}{dt}, \frac{dy}{dt}\right).$$

It is again a dot product of the gradient with a velocity vector. This rule —
generalized to many layers — is precisely **backpropagation** in neural
networks. Let's verify it numerically with the path
$x(t) = \cos t,\ y(t) = \sin t$ (the particle circles the bowl).
"""))
nb.append(code("""
def x_of_t(t):  return np.cos(t)
def y_of_t(t):  return np.sin(t)

def F(t):                       # the composed function f(x(t), y(t))
    return f(x_of_t(t), y_of_t(t))

t0 = 0.7

# (a) direct derivative of the composition, via finite difference:
h = 1e-6
dFdt_direct = (F(t0 + h) - F(t0 - h)) / (2 * h)

# (b) chain rule: grad f . velocity
grad_here = grad_f(x_of_t(t0), y_of_t(t0))
velocity  = np.array([-np.sin(t0), np.cos(t0)])   # [dx/dt, dy/dt]
dFdt_chain = grad_here @ velocity

print("dF/dt direct     :", dFdt_direct)
print("dF/dt chain rule :", dFdt_chain)
"""))

nb.append(md(r"""
The two numbers match — the chain rule works. Backprop is just this idea applied
mechanically through every layer of a network.
"""))

# ---- Exercise 3 ----
nb.append(md(r"""
## ✍️ Exercise 3 (solution included)

Let $f(x,y) = x^2 + 2y^2$ as before, with the **straight-line** path
$x(t) = 1 + 2t,\ y(t) = 3 - t$. Use the chain rule to compute $df/dt$ at $t=0$,
and check it against a finite-difference derivative of the composition.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
def x2(t): return 1 + 2 * t
def y2(t): return 3 - t
def F2(t): return f(x2(t), y2(t))

t0 = 0.0
grad_here = grad_f(x2(t0), y2(t0))      # gradient at (1, 3) = [2, 12]
velocity  = np.array([2.0, -1.0])       # [dx/dt, dy/dt]
print("chain rule  :", grad_here @ velocity)            # 2*2 + 12*(-1) = -8

h = 1e-6
print("finite diff :", (F2(t0 + h) - F2(t0 - h)) / (2 * h))
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 8. The Jacobian of a vector-valued function

So far $f$ returned **one** number. What if a function returns **several**
numbers — a vector? For example
$$\mathbf{F}(x, y) = \big(\,x^2 y,\ \ x + \sin y\,\big),$$
which maps a 2D point to another 2D point. Each output has its own gradient. We
stack those gradients as the **rows** of a matrix — the **Jacobian**:

$$J = \begin{pmatrix}
\dfrac{\partial F_1}{\partial x} & \dfrac{\partial F_1}{\partial y} \\[2mm]
\dfrac{\partial F_2}{\partial x} & \dfrac{\partial F_2}{\partial y}
\end{pmatrix}
= \begin{pmatrix} 2xy & x^2 \\ 1 & \cos y \end{pmatrix}.$$

The gradient is just the Jacobian of a *scalar* (single-output) function.
"""))
nb.append(code("""
def F_vec(v):
    x, y = v
    return np.array([x**2 * y, x + np.sin(y)])

def jacobian_exact(x, y):
    return np.array([[2 * x * y, x**2],
                     [1.0,       np.cos(y)]])

def numerical_jacobian(func, point, h=1e-5):
    point = np.asarray(point, dtype=float)
    f0 = func(point)
    J = np.zeros((f0.size, point.size))     # rows = outputs, cols = inputs
    for j in range(point.size):             # nudge each input coordinate
        step = np.zeros_like(point)
        step[j] = h
        J[:, j] = (func(point + step) - func(point - step)) / (2 * h)
    return J

p = [1.5, 0.5]
print("numeric Jacobian:\\n", numerical_jacobian(F_vec, p))
print("exact Jacobian:\\n",   jacobian_exact(1.5, 0.5))
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 9. Gradient descent in 2D — how models learn

Now the payoff. To find the minimum of $f$ we do not need a formula for it. We
just:

1. start somewhere, $\mathbf{p}_0$;
2. compute the gradient $\nabla f(\mathbf{p})$ (which way is uphill);
3. step the **opposite** way by a small amount $\eta$ (the *learning rate*):
   $$\mathbf{p}_{k+1} = \mathbf{p}_k - \eta\,\nabla f(\mathbf{p}_k);$$
4. repeat until the gradient is tiny.

That single update rule — *"take a small step downhill"* — is how essentially
every neural network is trained. The only differences in real ML are: $f$ is the
loss, $\mathbf{p}$ are millions of weights, and the gradient comes from
backprop. The idea is identical.
"""))
nb.append(code("""
def gradient_descent(grad, start, lr=0.1, n_steps=40):
    \"\"\"Return the list of points visited while stepping downhill.\"\"\"
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(n_steps):
        p = p - lr * grad(p)        # the one rule that matters
        path.append(p.copy())
    return np.array(path)

# gradient of our bowl as a function of the vector v = [x, y]
def grad_f_vec(v):
    return np.array([2 * v[0], 4 * v[1]])

path = gradient_descent(grad_f_vec, start=[-2.5, 2.5], lr=0.15, n_steps=40)
print("start :", path[0])
print("end   :", path[-1], " (should be near the minimum (0, 0))")
"""))

nb.append(md(r"""
### Watch it descend

Left: the path of points marching down the contour map, straight toward the
bottom of the bowl. Right: the **loss curve** — the value of $f$ at each
iteration — dropping toward zero. In real training you stare at exactly this
curve to see whether your model is learning.
"""))
nb.append(code("""
xs = np.linspace(-3, 3, 200)
ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

losses = [f(p[0], p[1]) for p in path]   # f-value at each visited point

fig = plt.figure(figsize=(11, 4.5))

# Left: descent path on the contour map
ax1 = fig.add_subplot(1, 2, 1)
ax1.contour(X, Y, Z, levels=20, alpha=0.6)
ax1.plot(path[:, 0], path[:, 1], "o-", color="crimson", markersize=4)
ax1.plot(0, 0, "k*", markersize=14, label="true minimum")
ax1.set_title("Gradient-descent path")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_aspect("equal"); ax1.legend()

# Right: loss vs iteration
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(losses, "o-", color="navy", markersize=4)
ax2.set_title("Loss vs iteration")
ax2.set_xlabel("iteration"); ax2.set_ylabel("f(x, y)")
ax2.grid(True)

plt.tight_layout()
plt.show()
"""))

nb.append(md(r"""
### The learning rate matters

The step size $\eta$ is the single most important knob:

- **too small** → painfully slow, the loss barely moves;
- **just right** → smooth, fast descent;
- **too large** → the steps overshoot and the loss can *blow up*.

Let's compare three learning rates on the same bowl.
"""))
nb.append(code("""
fig, ax = plt.subplots(figsize=(7, 4.5))
for lr in [0.02, 0.15, 0.55]:
    p = gradient_descent(grad_f_vec, start=[-2.5, 2.5], lr=lr, n_steps=40)
    L = [f(pt[0], pt[1]) for pt in p]
    ax.plot(L, "o-", markersize=3, label=f"lr = {lr}")
ax.set_title("Effect of the learning rate")
ax.set_xlabel("iteration"); ax.set_ylabel("loss f(x, y)")
ax.set_yscale("log")           # log scale: easier to compare decay rates
ax.legend(); ax.grid(True)
plt.show()
"""))

nb.append(md(r"""
With `lr = 0.02` the loss creeps down; with `lr = 0.15` it plunges; with
`lr = 0.55` it bounces and decays slowly (it is near the edge of instability).
Choosing a good learning rate is a daily concern in machine learning.
"""))

# ---- Homework ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **A new landscape.** Let $f(x, y) = (x - 1)^2 + (y + 2)^2$. By hand, find
   $\nabla f$ and the location of the minimum. Confirm with
   `numerical_gradient`, then run `gradient_descent` from $(4, 4)$ and check the
   path ends near your minimum.

2. **A saddle.** Plot the contours and 3D surface of $f(x, y) = x^2 - y^2$.
   Add the gradient field with `quiver`. Where is $\nabla f = (0,0)$? Start
   gradient descent at a few points near that spot and describe what happens —
   why is a saddle point tricky for optimizers?

3. **Directional derivative.** For $f(x,y) = e^{-(x^2 + y^2)}$ at the point
   $(0.5, 0.5)$, use your `numerical_gradient` to find the gradient, then report
   the directional derivative toward the origin. Is $f$ increasing or decreasing
   in that direction? Does that make sense from the shape of the surface?

4. **Two learning rates that fail.** For the bowl $f(x,y)=x^2+2y^2$, find (by
   experiment) a learning rate small enough that after 40 steps the loss is
   still above $1$, and a learning rate large enough that the loss *increases*
   (diverges). Plot both loss curves on the same axes.
"""))

nb.append(md(r"""
## What you learned

- A function $f(x,y)$ is a landscape; contours and 3D surfaces let you see it.
- **Partial derivatives** measure slope along each axis; the **gradient**
  $\nabla f$ stacks them into a vector.
- The gradient points in the direction of **steepest ascent** and is
  perpendicular to contour lines; $-\nabla f$ points downhill.
- The **directional derivative** $\nabla f \cdot \mathbf{u}$ gives the slope in
  any direction; it is maximal along the gradient.
- The **chain rule** $\frac{df}{dt} = \nabla f \cdot \mathbf{v}$ is the seed of
  backpropagation; the **Jacobian** generalizes the gradient to vector outputs.
- **Gradient descent** — step against the gradient — is how machine-learning
  models actually learn.

Next: we put gradients to work to *fit* models to data.
"""))

save(os.path.join(CH, "07_gradients.ipynb"), nb)
