"""Generator for Chapter 06 — Calculus & Derivatives (single variable).

Run from anywhere:  python tools/generators/ch06_calculus.py
Produces two notebooks in 06-calculus-and-derivatives/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "06-calculus-and-derivatives")


# ---------------------------------------------------------------------------
# Notebook 06a — Limits & Derivatives
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 06a — Limits & Derivatives

This notebook is about **change**. How fast is a function rising or falling at a
single instant? That number is the **derivative**, and almost every machine
learning algorithm finds its answer by following derivatives downhill.

We will build the idea from scratch:

1. A **limit** — sneaking up on a value we cannot reach directly.
2. The **derivative** as the slope of the tangent line / instantaneous rate of
   change.
3. A **numerical derivative** we can compute for *any* function with NumPy.
4. The basic differentiation **rules**, checked numerically.
5. The **linear (Taylor) approximation** — the workhorse of optimization.

Run each cell with **Shift + Enter**. Edit and re-run freely.
"""))

a.append(md(r"""
## 1. Setup

Pure NumPy for the math, Matplotlib for the pictures. We fix a random seed so
results are reproducible.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible randomness (used later)
"""))

a.append(md(r"""
## 2. The idea of a limit

The slope of a straight line through two points $(x_1,y_1)$ and $(x_2,y_2)$ is

$$\text{slope} = \frac{y_2 - y_1}{x_2 - x_1} = \frac{\Delta y}{\Delta x}.$$

For a *curve* there is no single slope — it bends. But if we pick two points
that are **very close together**, the line through them (a *secant*) looks almost
like the slope "at" that spot. A **limit** asks: what value does the secant slope
*approach* as the two points squeeze together?

Take $f(x)=x^2$ at $x=2$. The secant slope between $x=2$ and $x=2+h$ is

$$\frac{f(2+h)-f(2)}{h} = \frac{(2+h)^2 - 4}{h} = 4 + h.$$

As $h \to 0$ this clearly approaches $4$. Let's watch it happen numerically.
"""))
a.append(code("""
def f(x):
    return x**2

x0 = 2.0
for h in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    secant_slope = (f(x0 + h) - f(x0)) / h   # rise over run
    print(f"h = {h:8.4f}   secant slope = {secant_slope:.6f}")
"""))

a.append(md(r"""
The slope marches toward **4**. We say *"the limit of the secant slope as
$h\to 0$ is 4"*. That limiting slope is the **derivative** of $f$ at $x=2$.

We cannot just plug in $h=0$ (it would be $0/0$), but the limit dodges that
problem: we only ever get *close* to 0.
"""))

a.append(md(r"""
## 3. The derivative = slope of the tangent line

The **derivative** $f'(x)$ is the limit of the secant slope:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.$$

It is the **instantaneous rate of change** of $f$, and geometrically it is the
slope of the **tangent line** — the straight line that just grazes the curve at
$x$, matching both its height and its direction.

Below we draw $f(x)=x^2$ and several secant lines from the point $(2,4)$, each
using a smaller $h$. Watch them rotate toward the tangent (slope 4).
"""))
a.append(code("""
xs = np.linspace(0, 4, 200)        # 200 points from 0 to 4
plt.plot(xs, f(xs), label="f(x) = x^2", color="black")

x0 = 2.0
for h in [1.5, 1.0, 0.5]:
    slope = (f(x0 + h) - f(x0)) / h           # secant slope
    # line through (x0, f(x0)) with this slope, drawn across the window:
    secant = f(x0) + slope * (xs - x0)
    plt.plot(xs, secant, "--", label=f"secant h={h} (slope {slope:.1f})")

# the true tangent at x0 has slope 4:
tangent = f(x0) + 4.0 * (xs - x0)
plt.plot(xs, tangent, color="red", linewidth=2, label="tangent (slope 4)")

plt.scatter([x0], [f(x0)], color="red", zorder=5)
plt.title("Secant lines approaching the tangent")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
## 4. A numerical derivative for *any* function

On paper we use rules to get $f'$. But on a computer we can estimate the
derivative of **any** function we can evaluate, using the **central difference**:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}.$$

It looks left *and* right and averages, which is far more accurate than the
one-sided $\frac{f(x+h)-f(x)}{h}$. A good all-purpose step is $h \approx 10^{-5}$.
"""))
a.append(code("""
def deriv(f, x, h=1e-5):
    # central difference: (f(x+h) - f(x-h)) / (2h)
    # works elementwise if x is a NumPy array, because f is vectorized
    return (f(x + h) - f(x - h)) / (2 * h)

# check against the known answer f'(x) = 2x for f(x) = x^2
for x in [1.0, 2.0, 3.0]:
    print(f"deriv at x={x}: {deriv(f, x):.6f}   (exact 2x = {2*x})")
"""))

a.append(md(r"""
## 5. Plot a function and its derivative together

Because `deriv` accepts a whole array, we can evaluate $f'$ on a **grid** and
plot it. Let's use $f(x)=\sin(x)$, whose derivative is famously $\cos(x)$.
"""))
a.append(code("""
g = np.sin                       # the function
xs = np.linspace(-2*np.pi, 2*np.pi, 400)

y  = g(xs)                       # f
dy = deriv(g, xs)                # numerical f', computed on the whole grid

plt.plot(xs, y,  label="f(x) = sin(x)")
plt.plot(xs, dy, label="f'(x) (numerical)")
plt.plot(xs, np.cos(xs), "k--", label="cos(x) (exact)")
plt.axhline(0, color="gray", linewidth=0.8)
plt.title("A function and its derivative")
plt.xlabel("x"); plt.ylabel("value")
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
Notice the relationship: where $f$ has a **peak or valley** (slope zero), $f'$
**crosses zero**. Where $f$ rises, $f'$ is positive; where $f$ falls, $f'$ is
negative. The derivative is a complete "rate-of-change report" on $f$.
"""))

a.append(md(r"""
## 6. Drawing the tangent line at a point

The tangent line at $x_0$ has slope $f'(x_0)$ and passes through
$(x_0, f(x_0))$:

$$T(x) = f(x_0) + f'(x_0)\,(x - x_0).$$

Let's draw it for $f(x)=\sin(x)$ at $x_0 = 1$.
"""))
a.append(code("""
x0 = 1.0
slope = deriv(g, x0)             # f'(x0), numerically

xs = np.linspace(-1, 4, 300)
tangent = g(x0) + slope * (xs - x0)   # the tangent-line formula

plt.plot(xs, g(xs), label="f(x) = sin(x)")
plt.plot(xs, tangent, "r", label=f"tangent at x0=1 (slope {slope:.3f})")
plt.scatter([x0], [g(x0)], color="red", zorder=5)
plt.title("Tangent line touches the curve at x0")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
## 7. The differentiation rules, verified numerically

You will learn these rules in a calculus course; here we *confirm* them by
comparing our numerical `deriv` to the formula each rule predicts.

| Rule | Statement |
|------|-----------|
| Power | $\frac{d}{dx}x^n = n\,x^{n-1}$ |
| Sum | $(u+v)' = u' + v'$ |
| Product | $(uv)' = u'v + uv'$ |
| Chain | $\big(u(v(x))\big)' = u'(v(x))\cdot v'(x)$ |
"""))
a.append(code("""
x = 1.3   # test point

# --- Power rule: d/dx x^4 = 4 x^3 ---
p = lambda x: x**4
print("power  :", deriv(p, x), " vs ", 4 * x**3)

# --- Sum rule: (x^2 + sin x)' = 2x + cos x ---
s = lambda x: x**2 + np.sin(x)
print("sum    :", deriv(s, x), " vs ", 2*x + np.cos(x))

# --- Product rule: (x^2 * sin x)' = 2x sin x + x^2 cos x ---
pr = lambda x: x**2 * np.sin(x)
print("product:", deriv(pr, x), " vs ", 2*x*np.sin(x) + x**2*np.cos(x))

# --- Chain rule: (sin(x^2))' = cos(x^2) * 2x ---
ch = lambda x: np.sin(x**2)
print("chain  :", deriv(ch, x), " vs ", np.cos(x**2) * 2*x)
"""))

a.append(md(r"""
Every numerical value matches the rule's prediction to many decimals. The rules
are not magic — they are just bookkeeping for the same limit we computed in
Section 2.
"""))

a.append(md(r"""
## 8. Linear (Taylor) approximation

Near a point $x$, a smooth function looks almost like its tangent line. This is
the **first-order Taylor / linear approximation**:

$$f(x + \Delta) \approx f(x) + f'(x)\,\Delta.$$

In words: *"to move a little, take the current value plus slope times step."*
This single formula is the seed of gradient-based optimization. Let's see how
good it is for $f(x)=e^x$ around $x=0$ (where $f(0)=1$, $f'(0)=1$).
"""))
a.append(code("""
h = lambda x: np.exp(x)
x0 = 0.0
slope = deriv(h, x0)

xs = np.linspace(-1.5, 1.5, 300)
approx = h(x0) + slope * (xs - x0)     # f(x0) + f'(x0)*(x - x0)

plt.plot(xs, h(xs), label="f(x) = e^x (exact)")
plt.plot(xs, approx, "r--", label="linear approx at x0=0")
plt.scatter([x0], [h(x0)], color="red", zorder=5)
plt.title("Linear approximation is great near x0, drifts far away")
plt.xlabel("x"); plt.ylabel("y")
plt.legend(); plt.grid(True)
plt.show()
"""))
a.append(code("""
# Numerically: how close is the approximation for small steps?
x0 = 0.0
for delta in [0.5, 0.1, 0.01]:
    exact  = h(x0 + delta)
    approx = h(x0) + deriv(h, x0) * delta
    print(f"delta={delta:5.2f}   exact={exact:.6f}  approx={approx:.6f}"
          f"  error={abs(exact-approx):.6f}")
"""))

a.append(md(r"""
The error shrinks fast as the step gets smaller — roughly like $\Delta^2$.
That's why taking *small steps* along the slope is a trustworthy way to move.
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Use the central-difference `deriv` to estimate $f'(x)$ for
$f(x) = x^3 - 2x$ at $x = 2$. Compare with the exact derivative
$f'(x) = 3x^2 - 2$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
f = lambda x: x**3 - 2*x
x = 2.0
print("numerical:", deriv(f, x))
print("exact    :", 3*x**2 - 2)   # 3*4 - 2 = 10
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

For $f(x) = \cos(x)$, plot $f$ and its numerical derivative on $[0, 2\pi]$, and
overlay the exact derivative $-\sin(x)$ as a dashed line to confirm they agree.
"""))
a.append(md("**Solution:**"))
a.append(code("""
f = np.cos
xs = np.linspace(0, 2*np.pi, 300)

plt.plot(xs, f(xs),         label="cos(x)")
plt.plot(xs, deriv(f, xs),  label="numerical f'")
plt.plot(xs, -np.sin(xs), "k--", label="-sin(x) (exact)")
plt.title("cos and its derivative")
plt.xlabel("x"); plt.ylabel("value")
plt.legend(); plt.grid(True)
plt.show()
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Verify the **quotient rule** numerically: for $f(x)=\dfrac{\sin x}{x}$ at
   $x=1.5$, compare `deriv(f, 1.5)` against
   $\dfrac{x\cos x - \sin x}{x^2}$.
2. Make a table of the central-difference error for $f(x)=e^x$ at $x=0$ using
   $h = 10^{-1}, 10^{-3}, 10^{-5}, 10^{-8}, 10^{-12}$. Why does the error stop
   improving (and even get worse) for very tiny $h$?
3. Plot $f(x)=x^3 - 3x$ together with its numerical derivative on $[-2,2]$, and
   mark with dots the $x$-values where $f'(x)=0$.
4. Using the linear approximation $\sqrt{x+\Delta}\approx\sqrt{x}+
   \frac{1}{2\sqrt{x}}\Delta$ around $x=100$, estimate $\sqrt{101}$ and compare
   with `np.sqrt(101)`.
"""))

save(os.path.join(CH, "06a_limits_derivatives.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 06b — Optimization in 1D
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 06b — Optimization in 1D

Training a machine learning model means **minimizing** something — usually an
error. In one dimension we can already see every essential idea:

1. Minima and maxima happen where the slope is zero: $f'(x) = 0$.
2. We can **find** them by scanning a grid, or
3. by **gradient descent**: repeatedly step *downhill* opposite the slope.
4. The **learning rate** controls the step size — too small is slow, too large
   blows up.
5. As a bonus we meet **numerical integration** (the inverse idea: adding up
   slopes/areas).

This is the direct ancestor of how neural networks learn.
"""))

b.append(md(r"""
## 1. Setup

We bring back the central-difference derivative from notebook 06a.
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible

def deriv(f, x, h=1e-5):
    # central difference estimate of f'(x); vectorized over arrays
    return (f(x + h) - f(x - h)) / (2 * h)
"""))

b.append(md(r"""
## 2. Minima and maxima sit where $f'(x) = 0$

At the bottom of a valley (a **minimum**) or the top of a hill (a **maximum**)
the curve is momentarily flat — its tangent is horizontal, so $f'(x) = 0$. These
flat spots are called **critical points**.

Consider

$$f(x) = x^4 - 3x^2 + x.$$

Let's plot $f$ and $f'$ and see where the derivative crosses zero.
"""))
b.append(code("""
def f(x):
    return x**4 - 3*x**2 + x

xs = np.linspace(-2, 2, 400)

plt.plot(xs, f(xs),        label="f(x)")
plt.plot(xs, deriv(f, xs), label="f'(x)")
plt.axhline(0, color="gray", linewidth=0.8)
plt.title("Critical points: where f'(x) crosses zero")
plt.xlabel("x"); plt.ylabel("value")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
Where $f'$ crosses zero going **up** (negative to positive), $f$ has a local
**minimum**; crossing **down** gives a local **maximum**.
"""))

b.append(md(r"""
## 3. Finding the minimum by scanning a grid

The bluntest method: evaluate $f$ at many points and keep the smallest. NumPy's
`argmin` returns the *index* of the minimum value, which we use to look up the
winning $x$.
"""))
b.append(code("""
xs = np.linspace(-2, 2, 100001)   # a fine grid
ys = f(xs)

i = np.argmin(ys)                 # index of the smallest y
print("grid minimum near x =", round(xs[i], 4))
print("minimum value f(x)  =", round(ys[i], 4))
"""))

b.append(md(r"""
Scanning works but is wasteful: in real ML, $x$ may have *millions* of
components and we cannot grid them. We need a method that *follows the slope*.
"""))

b.append(md(r"""
## 4. Gradient descent

The derivative points in the direction of **steepest increase**. To go *down*,
step in the **opposite** direction. Repeat:

$$x \leftarrow x - \eta\, f'(x),$$

where $\eta$ (eta) is the **learning rate** — how big a step to take. Start
somewhere, take many small downhill steps, and you slide into a minimum.
"""))
b.append(code("""
def gradient_descent(f, x_start, lr, n_steps):
    # returns the list of x-values visited (the "path")
    x = x_start
    path = [x]
    for _ in range(n_steps):
        slope = deriv(f, x)      # which way is uphill?
        x = x - lr * slope       # step the opposite way
        path.append(x)
    return np.array(path)

path = gradient_descent(f, x_start=1.5, lr=0.1, n_steps=40)
print("started at x =", path[0])
print("ended   at x =", round(path[-1], 4))
print("f at end      =", round(f(path[-1]), 4))
"""))

b.append(md(r"""
## 5. Plotting the descent path on the curve

Let's watch the steps. Each dot is one update; they march downhill into the
nearest valley.
"""))
b.append(code("""
xs = np.linspace(-2, 2, 400)
plt.plot(xs, f(xs), label="f(x)", color="black")

# the path of points the optimizer visited:
plt.plot(path, f(path), "o-", color="red", markersize=4,
         label="gradient descent path")
plt.scatter([path[0]],  [f(path[0])],  color="green", zorder=5, label="start")
plt.scatter([path[-1]], [f(path[-1])], color="blue",  zorder=5, label="end")

plt.title("Gradient descent slides into a minimum")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
Notice the dots are **far apart** when the slope is steep and **bunch up** as the
curve flattens near the bottom — the step size $\eta f'(x)$ naturally shrinks
because $f'(x)\to 0$ there.
"""))

b.append(md(r"""
## 6. The effect of the learning rate

The learning rate $\eta$ is the single most important knob:

- **Too small** — tiny steps, painfully slow, may not arrive in time.
- **Just right** — steady, quick convergence.
- **Too large** — steps overshoot the valley and can *diverge* (fly off).

Let's run the same descent from the same start with three learning rates and
track $f(x)$ over the steps.
"""))
b.append(code("""
plt.figure()
for lr in [0.01, 0.1, 0.3]:
    path = gradient_descent(f, x_start=1.5, lr=lr, n_steps=40)
    plt.plot(f(path), "o-", markersize=3, label=f"lr = {lr}")

plt.title("Learning rate controls how fast (or whether) we converge")
plt.xlabel("step"); plt.ylabel("f(x) at current point")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(code("""
# A learning rate that is too big can DIVERGE. Watch the values explode:
path = gradient_descent(f, x_start=1.5, lr=0.6, n_steps=15)
print("x values with lr=0.6 (too large):")
print(np.round(path, 3))
"""))

b.append(md(r"""
With the careful rates the error falls smoothly; with `lr=0.6` the iterate
overshoots and the numbers blow up. This balance — fast but stable — is exactly
what ML practitioners tune every day.
"""))

b.append(md(r"""
## 7. Bonus: numerical integration (the inverse idea)

Differentiation measures slope; **integration** adds things up — the area under
a curve. They are inverses (the *Fundamental Theorem of Calculus*). The simplest
numerical method is the **trapezoid rule**: approximate the area by lots of thin
trapezoids. NumPy has it built in as `np.trapz`.

Let's check $\displaystyle\int_0^{\pi} \sin(x)\,dx = 2$.
"""))
b.append(code("""
xs = np.linspace(0, np.pi, 1000)
ys = np.sin(xs)

area = np.trapz(ys, xs)          # trapezoid-rule area under sin from 0 to pi
print("numerical integral:", area)
print("exact value       :", 2.0)
"""))
b.append(code("""
# Visualize the area we just measured
xs = np.linspace(0, np.pi, 200)
plt.plot(xs, np.sin(xs), color="black", label="sin(x)")
plt.fill_between(xs, np.sin(xs), alpha=0.3, label="area = integral")
plt.title("Integration adds up area under the curve")
plt.xlabel("x"); plt.ylabel("sin(x)")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
So derivatives tell us *which way to step* (optimization) and integrals tell us
*how much accumulates* (areas, probabilities, totals). Both are everywhere in
machine learning.
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Minimize the simple parabola $f(x) = (x - 3)^2$ with gradient descent, starting
at $x = 0$ with learning rate $0.1$ for 50 steps. The minimum is obviously at
$x = 3$ — does the optimizer find it?
"""))
b.append(md("**Solution:**"))
b.append(code("""
f = lambda x: (x - 3)**2
path = gradient_descent(f, x_start=0.0, lr=0.1, n_steps=50)
print("ended at x =", round(path[-1], 4), "(target 3)")
print("f at end   =", round(f(path[-1]), 6))
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Use `np.trapz` to estimate $\displaystyle\int_0^1 x^2\,dx$, whose exact value is
$\tfrac{1}{3}$. Try it with 5 grid points and again with 1000, and see how the
accuracy improves.
"""))
b.append(md("**Solution:**"))
b.append(code("""
g = lambda x: x**2
for n in [5, 1000]:
    xs = np.linspace(0, 1, n)
    print(f"n={n:5d}  trapz = {np.trapz(g(xs), xs):.6f}   (exact 0.333333)")
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Run gradient descent on $f(x) = x^4 - 3x^2 + x$ from two different starts,
   $x_0 = -1.5$ and $x_0 = 1.5$. Do they reach the **same** minimum? What does
   this tell you about *local* minima?
2. For the parabola $f(x)=(x-3)^2$, find by experiment the largest learning rate
   that still converges. What happens just above it?
3. Write `minimize(f, x0, lr, tol)` that **stops automatically** once
   $|f'(x)| < \text{tol}$ instead of running a fixed number of steps. Report how
   many steps it took.
4. Estimate $\displaystyle\int_{-3}^{3} e^{-x^2}\,dx$ with `np.trapz`. (The exact
   value is close to $\sqrt{\pi}\approx 1.7725$.) How many grid points do you
   need for 4-decimal accuracy?
"""))

save(os.path.join(CH, "06b_optimization_1d.ipynb"), b)
