"""Generator for Chapter 13c — Activation Functions.

Run from anywhere:  python tools/generators/ch13c_activations.py
Produces 13-neural-networks-from-scratch/13c_activation_functions.ipynb
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "13-neural-networks-from-scratch")


# ===========================================================================
# Notebook 13c — Activation Functions
# ===========================================================================
c = []

c.append(md(r"""
# Chapter 13c — Activation Functions

In 13a we met three activations in passing; in 13b we trained a network with
sigmoid. This notebook zooms in on the **activation function** itself — the
small nonlinear squash $\sigma$ that sits at the heart of every neuron.

By the end you will understand:

1. **why** a nonlinear activation is *required* (stacking linear layers
   collapses to a single linear map — we'll prove it numerically),
2. the five workhorses — **sigmoid, tanh, ReLU, Leaky ReLU**, and **softmax**
   (an *output* activation) — their formulas, plots, and **derivatives**,
3. the **vanishing-gradient** problem and why ReLU helps,
4. **"dying ReLU"** and why Leaky ReLU was invented,
5. a practical **"which activation to use when"** guide.

Everything is pure NumPy + Matplotlib, with `np.random.default_rng(0)` for
reproducibility.
"""))

c.append(md(r"""
## 1. Setup
"""))
c.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible random generator

z = np.linspace(-6, 6, 400)      # a grid of pre-activation values for plotting
"""))

# ---------------------------------------------------------------------------
# Why nonlinearity
# ---------------------------------------------------------------------------
c.append(md(r"""
## 2. Why do we need a nonlinear activation at all?

A neuron computes $z = w\cdot x + b$ — a **linear** (affine) map. If we then
*also* used a linear "activation" (or none at all), what would stacking layers
buy us? Nothing. Composing linear maps gives another linear map:

$$
\underbrace{W_2\,(W_1 x + b_1) + b_2}_{\text{two linear layers}}
   \;=\; \underbrace{(W_2 W_1)}_{W'}\,x + \underbrace{(W_2 b_1 + b_2)}_{b'}
   \;=\; W' x + b'.
$$

So a deep stack of *purely linear* layers is mathematically identical to a
**single** linear layer — it can only ever draw a straight decision boundary,
exactly the limitation that sank the perceptron on XOR (13a). The **nonlinear
activation between layers is what lets the network bend space** and represent
curved relationships. Let's verify the collapse numerically.
"""))
c.append(code("""
# Two random linear layers applied WITHOUT any activation in between.
W1 = rng.normal(size=(3, 4))   # layer 1: 3 -> 4
b1 = rng.normal(size=(4,))
W2 = rng.normal(size=(4, 2))   # layer 2: 4 -> 2
b2 = rng.normal(size=(2,))

x = rng.normal(size=(3,))      # one input vector

# Path A: push x through both linear layers, one after the other.
two_layers = (x @ W1 + b1) @ W2 + b2

# Path B: collapse them ANALYTICALLY into a single linear layer W', b'.
W_eq = W1 @ W2                 # (3,4)@(4,2) -> (3,2)
b_eq = b1 @ W2 + b2
one_layer = x @ W_eq + b_eq

print("two linear layers :", two_layers)
print("one equivalent    :", one_layer)
print("identical?         ", np.allclose(two_layers, one_layer))  # True
"""))
c.append(md(r"""
They match exactly: **two linear layers = one linear layer.** Depth only adds
power once we insert a *nonlinearity* between the layers. That nonlinearity is
the activation function — the rest of this notebook.
"""))

# ---------------------------------------------------------------------------
# The functions and derivatives
# ---------------------------------------------------------------------------
c.append(md(r"""
## 3. The five workhorse activations

We define each function **and its derivative** (we need derivatives for
backprop). For a neuron $a = \sigma(z)$, backprop multiplies the incoming error
by $\sigma'(z)$ — so the *shape of the derivative* decides how well gradients
flow. Keep an eye on where each derivative is large (gradient flows) versus near
zero (gradient is choked off).
"""))
c.append(md(r"""
### Definitions

$$
\begin{aligned}
\text{sigmoid}(z) &= \frac{1}{1+e^{-z}} \in (0,1),
   & \sigma'(z) &= \sigma(z)\,(1-\sigma(z)),\\[4pt]
\tanh(z) &= \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \in (-1,1),
   & \tanh'(z) &= 1 - \tanh(z)^2,\\[4pt]
\text{ReLU}(z) &= \max(0, z),
   & \text{ReLU}'(z) &= \begin{cases}1 & z>0\\ 0 & z<0\end{cases},\\[4pt]
\text{LeakyReLU}(z) &= \begin{cases}z & z>0\\ \alpha z & z\le 0\end{cases},
   & \text{LeakyReLU}'(z) &= \begin{cases}1 & z>0\\ \alpha & z<0\end{cases}
\end{aligned}
$$

with a small slope $\alpha$ (commonly $0.01$) for Leaky ReLU.
"""))
c.append(code("""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)             # sigma'(z) = sigma(z)(1 - sigma(z))

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z)**2       # tanh'(z) = 1 - tanh(z)^2

def relu(z):
    return np.maximum(0.0, z)

def relu_deriv(z):
    return (z > 0).astype(float)   # 1 where z>0, else 0

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
"""))

c.append(md(r"""
### Plotting each function next to its derivative

A row per activation: the function on the left, its derivative on the right.
"""))
c.append(code("""
funcs = [
    ("sigmoid",     sigmoid,     sigmoid_deriv),
    ("tanh",        tanh,        tanh_deriv),
    ("ReLU",        relu,        relu_deriv),
    ("Leaky ReLU",  leaky_relu,  leaky_relu_deriv),
]

fig, axes = plt.subplots(len(funcs), 2, figsize=(10, 12))
for row, (name, f, df) in enumerate(funcs):
    axes[row, 0].plot(z, f(z), color="tab:blue")
    axes[row, 0].set_title(name)
    axes[row, 1].plot(z, df(z), color="tab:red")
    axes[row, 1].set_title(name + "  derivative")
    for col in (0, 1):
        axes[row, col].axhline(0, color="grey", lw=0.7)
        axes[row, col].axvline(0, color="grey", lw=0.7)
        axes[row, col].grid(True)
fig.tight_layout()
plt.show()
"""))
c.append(md(r"""
**What to notice in the right column (the derivatives).**

- **sigmoid / tanh:** the derivative is a *bump* centred at $z=0$ that decays to
  almost $0$ for $|z|\gtrsim 4$. Sigmoid's slope peaks at only $0.25$; tanh's
  peaks at $1$ but still collapses when saturated.
- **ReLU:** the derivative is a *step* — exactly $1$ for $z>0$ and $0$ for
  $z<0$. No decay on the positive side: gradients pass through undamped.
- **Leaky ReLU:** same step, but the negative side is a small $\alpha$ instead
  of $0$ — so even "off" units still leak a little gradient.
"""))

# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------
c.append(md(r"""
## 4. Softmax — an *output* activation for several classes

Sigmoid turns one number into one probability (2-class problems). For **$K$
mutually exclusive classes** we use **softmax**, which turns a whole vector of
scores $z = (z_1,\dots,z_K)$ into a probability distribution:

$$
\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.
$$

Every output is in $(0,1)$ and they **sum to $1$**, so the vector reads as
"probability of each class". Softmax is an **output-layer** activation (it
generalizes sigmoid to many classes); we do *not* put it on hidden layers.

A numerical detail: $e^{z_k}$ overflows for large $z_k$, so we subtract the max
first — this is a standard trick and does not change the result, because the
constant cancels in the ratio.
"""))
c.append(code("""
def softmax(z):
    z = z - np.max(z)             # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum()

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print("scores       :", scores)
print("softmax probs:", np.round(probs, 4))
print("sum of probs :", probs.sum())     # 1.0  -> a valid distribution
"""))
c.append(code("""
# Visualize how softmax converts scores into a probability bar chart.
labels = ["class 0", "class 1", "class 2"]
plt.figure(figsize=(6, 4))
plt.bar(labels, probs, color="tab:purple")
plt.title("softmax([2.0, 1.0, 0.1]) -> a probability distribution")
plt.ylabel("probability")
plt.ylim(0, 1)
for i, p in enumerate(probs):
    plt.text(i, p + 0.02, f"{p:.2f}", ha="center")
plt.show()
"""))
c.append(md(r"""
The largest score (class 0) gets the largest probability, but softmax is
*soft*: the other classes still receive nonzero probability. That softness is
exactly what makes it differentiable and trainable.
"""))

# ---------------------------------------------------------------------------
# Vanishing gradient
# ---------------------------------------------------------------------------
c.append(md(r"""
## 5. The vanishing-gradient problem

Here is the deepest practical reason ReLU took over from sigmoid/tanh. Recall
from 13b that backprop **multiplies** the error by the activation derivative
$\sigma'(z)$ at *every layer* it passes through. Now look at the numbers:

- sigmoid's derivative is **at most $0.25$**, and far smaller once a unit
  saturates ($|z|$ large),
- tanh's derivative is at most $1$ but also collapses toward $0$ when saturated.

If you chain $L$ such layers, the error signal gets multiplied by roughly
$(\text{small number})^L$. For sigmoid that is at best $0.25^L$:
"""))
c.append(code("""
print("sigmoid'(0)  =", sigmoid_deriv(np.array(0.0)))   # 0.25  (its MAXIMUM)
print("sigmoid'(6)  =", sigmoid_deriv(np.array(6.0)))   # ~0.0025 (saturated)

# How an error signal shrinks through a deep stack, BEST case (slope 0.25 each):
for L in [1, 5, 10, 20]:
    shrink = 0.25 ** L
    print(f"after {L:2d} sigmoid layers, gradient scaled by <= {shrink:.2e}")
"""))
c.append(md(r"""
After only 10 layers the gradient is scaled by $\le 10^{-6}$ — and that is the
*best* case, before saturation makes it worse. The early layers receive almost
**no** learning signal: the gradient has **vanished**. This is why deep
sigmoid/tanh networks were historically so hard to train.

**ReLU's fix.** Its derivative is exactly $1$ on the positive side (and $0$ on
the negative side) — no shrinking factor at all for active units. The error
signal passes through **undamped**, so gradients survive across many layers.
Let's contrast the multiplicative shrink directly.
"""))
c.append(code("""
# Compare the per-layer multiplier for a typical positive pre-activation z=2.
z_typical = np.array(2.0)
print("sigmoid'(2) =", float(sigmoid_deriv(z_typical)))   # ~0.105  (chokes)
print("tanh'(2)    =", float(tanh_deriv(z_typical)))      # ~0.071  (chokes)
print("ReLU'(2)    =", float(relu_deriv(z_typical)))      # 1.0     (no decay)

for L in [1, 5, 10, 20]:
    print(f"{L:2d} layers:  sigmoid -> {0.105**L:.2e}   ReLU -> {1.0**L:.2e}")
"""))
c.append(md(r"""
ReLU keeps the multiplier at $1$ no matter how deep, while sigmoid drives it to
zero. This single property is the main reason **ReLU is the default hidden
activation in modern deep networks.**
"""))

# ---------------------------------------------------------------------------
# Dying ReLU
# ---------------------------------------------------------------------------
c.append(md(r"""
## 6. "Dying ReLU" — and why Leaky ReLU helps

ReLU is not perfect. Its derivative is **exactly $0$ for $z<0$**. If a unit's
pre-activation $z$ becomes negative for *every* training example — e.g. after a
large gradient step pushes its bias very negative — then:

- its output is always $0$ (it never fires), and
- its gradient is always $0$, so **it never updates again**.

The unit is stuck, permanently dead. With an unlucky initialization or too-large
a learning rate, a chunk of your ReLU units can die, wasting capacity. This is
the **dying-ReLU** problem.

**Leaky ReLU** fixes it by giving the negative side a small nonzero slope
$\alpha$ (e.g. $0.01$). Now the derivative there is $\alpha$, not $0$ — a dead
unit still receives a trickle of gradient and can **come back to life**.
"""))
c.append(code("""
# Count "dead" units: those whose derivative is 0 across ALL inputs.
z_units = rng.normal(loc=-1.0, scale=1.0, size=(1000,))   # mostly-negative pre-acts

relu_alive  = (relu_deriv(z_units) > 0).mean()
leaky_alive = (leaky_relu_deriv(z_units) > 0).mean()
print("fraction of inputs where ReLU passes gradient      :", relu_alive)
print("fraction of inputs where Leaky ReLU passes gradient:", leaky_alive)
# Leaky ReLU passes *some* gradient everywhere -> units can recover.
"""))

# ---------------------------------------------------------------------------
# Which to use guide
# ---------------------------------------------------------------------------
c.append(md(r"""
## 7. Which activation should I use? (a practical guide)

| Where | Recommended | Why |
|-------|-------------|-----|
| **Hidden layers (default)** | **ReLU** | cheap, no positive-side saturation, gradients flow |
| Hidden, if many units die | **Leaky ReLU** | nonzero negative slope keeps units alive |
| Hidden, small/shallow nets | tanh | zero-centred, smooth; fine when depth is small |
| **Output — binary (2-class)** | **sigmoid** | one probability in $(0,1)$ |
| **Output — multi-class** | **softmax** | a probability distribution over $K$ classes |
| Output — regression (real value) | **none (linear)** | the target is unbounded, don't squash it |

**Rules of thumb (Andrew Ng style):**

- Start hidden layers with **ReLU**; only reach for alternatives if you observe
  a problem (e.g. many dead units → Leaky ReLU).
- Pick the **output** activation from the *task*, not habit: sigmoid for
  yes/no, softmax for "pick one of $K$", linear for predicting a number.
- Avoid sigmoid/tanh in **deep** hidden stacks — the vanishing gradient of
  Section 5 will choke learning.
"""))

# ---------------------------------------------------------------------------
# Exercises with solutions
# ---------------------------------------------------------------------------
c.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

The handy identity $\sigma'(z) = \sigma(z)(1-\sigma(z))$ implies the sigmoid's
derivative is **maximized at $z=0$**, where it equals $0.25$. Verify this
numerically: evaluate `sigmoid_deriv` on the grid `z` and print its maximum and
the $z$ at which it occurs.
"""))
c.append(md("**Solution:**"))
c.append(code("""
d = sigmoid_deriv(z)
i = np.argmax(d)
print("max sigmoid'(z) =", round(float(d[i]), 4))   # 0.25
print("attained at z   =", round(float(z[i]), 4))    # ~0.0
"""))

c.append(md(r"""
## ✍️ Exercise 2 (solution included)

Show numerically that **softmax generalizes sigmoid**. For two classes, softmax
on scores $[z, 0]$ should give the same first probability as $\text{sigmoid}(z)$.
Confirm this for $z = -2, 0, 1.5$.
"""))
c.append(md("**Solution:**"))
c.append(code("""
for zv in [-2.0, 0.0, 1.5]:
    two_class = softmax(np.array([zv, 0.0]))[0]   # P(class 0)
    print(f"z={zv:+.1f}:  softmax->{two_class:.4f}   sigmoid->{sigmoid(zv):.4f}")
# The two columns match: sigmoid is the 2-class special case of softmax.
"""))

c.append(md(r"""
## ✍️ Exercise 3 (solution included)

Confirm the analytic derivatives against a **numerical** derivative
$\frac{f(z+h)-f(z-h)}{2h}$ for tanh and ReLU at $z = 0.7$ (a point where ReLU is
smooth). They should agree closely.
"""))
c.append(md("**Solution:**"))
c.append(code("""
z0, h = 0.7, 1e-5
for name, f, df in [("tanh", tanh, tanh_deriv), ("ReLU", relu, relu_deriv)]:
    numeric = (f(np.array(z0 + h)) - f(np.array(z0 - h))) / (2 * h)
    print(f"{name}: analytic={float(df(np.array(z0))):.6f}  numeric={float(numeric):.6f}")
"""))

# ---------------------------------------------------------------------------
# Homework
# ---------------------------------------------------------------------------
c.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **ELU and Softplus.** Look up and implement two more activations: the
   *softplus* $\,\text{softplus}(z)=\ln(1+e^z)$ (a smooth ReLU) and the *ELU*.
   Plot each with its derivative and compare to ReLU.
2. **Vanishing gradient, empirically.** Stack 20 sigmoid layers with random
   weights, push a vector forward, then multiply the per-layer derivatives to
   estimate the gradient magnitude reaching layer 1. Repeat with ReLU and
   compare.
3. **Dying ReLU in action.** In the 13b network, swap the hidden activation to
   ReLU and use a deliberately large learning rate. Count how many hidden units
   output $0$ for *all* inputs after training. Then switch to Leaky ReLU and
   recount.
4. **Softmax derivative.** Derive the Jacobian
   $\frac{\partial\,\text{softmax}(z)_i}{\partial z_j}$ and verify one entry
   numerically. (Hint: the answer is $s_i(\delta_{ij} - s_j)$.)
5. **Temperature.** Replace $z$ by $z/T$ inside softmax. Plot the output
   distribution for $T = 0.5, 1, 5$ on fixed scores and describe how $T$
   controls how "peaked" the distribution is.
"""))

save(os.path.join(CH, "13c_activation_functions.ipynb"), c)
