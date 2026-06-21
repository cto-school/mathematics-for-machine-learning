"""Generator for Chapter 13 — Neural Networks from Scratch (the capstone).

Run from anywhere:  python tools/generators/ch13_neural_networks.py
Produces two notebooks in 13-neural-networks-from-scratch/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "13-neural-networks-from-scratch")


# ===========================================================================
# Notebook 13a — The Perceptron: a single artificial neuron
# ===========================================================================
a = []

a.append(md(r"""
# Chapter 13a — The Perceptron: a Single Artificial Neuron

Welcome to the **capstone** of the course. Everything you have learned —
vectors and dot products (Ch. 03), matrices (Ch. 04), derivatives and the
chain rule (Ch. 06–07), and gradient descent (Ch. 12) — comes together to
build a **neural network from scratch**, using nothing but NumPy.

We start small: a *single neuron*. A neuron takes inputs $x$, forms a
**weighted sum**

$$z = w \cdot x + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b,$$

then passes $z$ through a nonlinear **activation function** $a = \sigma(z)$.
That is the entire object. A neural network is just *many* of these wired
together — which is the subject of notebook 13b.

In this notebook we will:

1. build one neuron and the three classic activation functions,
2. train a neuron with the **perceptron learning rule** on a separable
   dataset, and
3. watch it **fail on XOR** — the failure that forces us to invent
   *hidden layers*.
"""))

a.append(md(r"""
## 1. Setup

Pure NumPy and Matplotlib. We fix the random seed with
`np.random.default_rng(0)` so every run is identical and reproducible.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # one reproducible random generator
"""))

a.append(md(r"""
## 2. The weighted sum $z = w \cdot x + b$

A neuron's input is a vector $x \in \mathbb{R}^n$. It carries a weight vector
$w \in \mathbb{R}^n$ and a single number $b$ (the **bias**). The "pre-activation"
is the dot product plus the bias — exactly the dot product from Chapter 03.
"""))
a.append(code("""
def pre_activation(w, x, b):
    return np.dot(w, x) + b      # w . x + b  (a single number)

w = np.array([2.0, -1.0])        # weights
b = 0.5                          # bias
x = np.array([1.0, 3.0])         # an input point

print("z =", pre_activation(w, x, b))   # 2*1 + (-1)*3 + 0.5 = -0.5
"""))

a.append(md(r"""
## 3. Activation functions

The weighted sum $z$ is linear. To let neurons represent *curved* relationships
we squash $z$ through a nonlinear **activation** $\sigma$. The three classics:

$$
\text{sigmoid}(z) = \frac{1}{1+e^{-z}}, \qquad
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}, \qquad
\text{ReLU}(z) = \max(0, z).
$$

- **sigmoid** squashes into $(0,1)$ — natural for probabilities (Ch. 11).
- **tanh** squashes into $(-1,1)$ — like sigmoid but centred at $0$.
- **ReLU** ("rectified linear unit") is $0$ for negatives, identity for
  positives — cheap and the modern default for hidden layers.
"""))
a.append(code("""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)            # NumPy already has tanh

def relu(z):
    return np.maximum(0.0, z)    # elementwise max with 0
"""))

a.append(md(r"""
Let's plot all three on the same axes so you can see their shapes.
"""))
a.append(code("""
z = np.linspace(-6, 6, 200)      # 200 points from -6 to 6

plt.figure(figsize=(7, 4))
plt.plot(z, sigmoid(z), label="sigmoid")
plt.plot(z, tanh(z),    label="tanh")
plt.plot(z, relu(z),    label="ReLU")
plt.axhline(0, color="grey", lw=0.7)
plt.axvline(0, color="grey", lw=0.7)
plt.title("Three activation functions")
plt.xlabel("z"); plt.ylabel("activation(z)")
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
**Discussion.** Notice their ranges differ: sigmoid lives in $(0,1)$, tanh in
$(-1,1)$, ReLU in $[0,\infty)$. Both sigmoid and tanh **saturate** — for large
$|z|$ their slope is nearly $0$, which (we'll see in 13b) makes gradients tiny
and learning slow. ReLU has slope exactly $1$ for $z>0$, so it does not
saturate on the positive side; that is the main reason it dominates modern
deep networks. For our small 2-layer network in 13b, smooth sigmoid/tanh work
fine and make the calculus cleaner.

> **A whole notebook on this.** Activation functions are important enough that
> notebook **13c** is devoted to them: why nonlinearity is *required*, the
> derivatives of each, the **vanishing-gradient** problem, "dying ReLU", and a
> practical "which one to use when" guide. Come back to it after 13b.
"""))

a.append(md(r"""
### A neuron as a tiny *computation graph*

It helps to picture a neuron not as one formula but as a short **chain of simple
operations** — a *computation graph*:

$$
x \;\xrightarrow{\;\times w,\ +b\;}\; z \;\xrightarrow{\;\sigma\;}\; a.
$$

Read left to right, this is the **forward pass**: multiply by $w$, add $b$ to
get $z$, then squash to get $a$. Each arrow is one elementary step whose
derivative we know. In 13b we will run the *same* graph backwards — reusing the
values we computed on the way forward — to get gradients. That backward trip is
**backpropagation**. Keep this picture in mind: *a neural network is just a long
chain of simple operations, and calculus lets us differentiate the whole chain
one link at a time.*
"""))

a.append(md(r"""
## 4. A linearly separable dataset

Let's make a 2-class dataset in the plane that a single line *can* separate:
two Gaussian blobs, one labelled $0$ and one labelled $1$.
"""))
a.append(code("""
n = 40                                   # points per class
# class 0: centred near (-2, -2);  class 1: centred near (2, 2)
X0 = rng.normal(loc=[-2, -2], scale=0.8, size=(n, 2))
X1 = rng.normal(loc=[ 2,  2], scale=0.8, size=(n, 2))

X = np.vstack([X0, X1])                   # shape (80, 2): all points
y = np.array([0]*n + [1]*n)              # shape (80,):  labels 0/1
print("X shape:", X.shape, " y shape:", y.shape)
"""))
a.append(code("""
plt.figure(figsize=(5, 5))
plt.scatter(X0[:, 0], X0[:, 1], color="tab:red",  label="class 0")
plt.scatter(X1[:, 0], X1[:, 1], color="tab:blue", label="class 1")
plt.title("A linearly separable dataset")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.grid(True); plt.axis("equal")
plt.show()
"""))

a.append(md(r"""
## 5. The perceptron learning rule

The classic perceptron (Rosenblatt, 1958) uses a **step** activation: predict
$1$ if $z = w\cdot x + b \ge 0$, else predict $0$. Training is a beautifully
simple loop. For each point $(x, y)$:

1. predict $\hat{y} = \text{step}(w\cdot x + b)$,
2. compute the error $e = y - \hat{y}$ (which is $-1$, $0$, or $+1$),
3. nudge the weights toward fixing that error:

$$
w \leftarrow w + \eta\, e\, x, \qquad b \leftarrow b + \eta\, e.
$$

Here $\eta$ is the **learning rate**. If a point is already correct, $e=0$ and
nothing changes. The famous *perceptron convergence theorem* guarantees this
finds a separating line in finite steps **if one exists**.
"""))
a.append(code("""
def step(z):
    return 1 if z >= 0 else 0          # hard threshold

def train_perceptron(X, y, eta=0.1, epochs=20):
    w = np.zeros(2)                    # start from zero weights
    b = 0.0
    for _ in range(epochs):           # one epoch = one sweep over all points
        for xi, yi in zip(X, y):
            y_hat = step(np.dot(w, xi) + b)
            error = yi - y_hat        # -1, 0, or +1
            w = w + eta * error * xi  # the update rule
            b = b + eta * error
    return w, b

w_p, b_p = train_perceptron(X, y)
print("learned weights:", w_p, " bias:", b_p)
"""))

a.append(md(r"""
### Plotting the decision boundary

The boundary is the line where $z = 0$, i.e. $w_1 x_1 + w_2 x_2 + b = 0$.
Solving for $x_2$ gives $x_2 = -(w_1 x_1 + b)/w_2$.
"""))
a.append(code("""
def plot_boundary_line(w, b, X0, X1, title):
    plt.figure(figsize=(5, 5))
    plt.scatter(X0[:, 0], X0[:, 1], color="tab:red",  label="class 0")
    plt.scatter(X1[:, 0], X1[:, 1], color="tab:blue", label="class 1")
    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    ys = -(w[0] * xs + b) / w[1]      # the line w.x + b = 0
    plt.plot(xs, ys, "k--", label="decision boundary")
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.show()

plot_boundary_line(w_p, b_p, X0, X1, "Perceptron decision boundary")
"""))
a.append(code("""
# Accuracy: fraction of points predicted correctly
preds = np.array([step(np.dot(w_p, xi) + b_p) for xi in X])
print("training accuracy:", (preds == y).mean())   # should be 1.0
"""))

a.append(md(r"""
The dashed line cleanly separates the two blobs and the accuracy is $1.0$. For
**linearly separable** data, a single neuron is enough.
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

The **derivative** of the sigmoid has a famously tidy form,
$\sigma'(z) = \sigma(z)\,(1-\sigma(z))$. We'll need it for backpropagation in
13b. Write `sigmoid_deriv(z)` and verify it against a numerical derivative
$\frac{\sigma(z+h)-\sigma(z-h)}{2h}$ at $z = 0.7$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

z0 = 0.7
h = 1e-5
numeric = (sigmoid(z0 + h) - sigmoid(z0 - h)) / (2 * h)
print("formula:", sigmoid_deriv(z0))
print("numeric:", numeric)
print("difference:", abs(sigmoid_deriv(z0) - numeric))   # ~1e-11
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

Run the perceptron with a **larger** learning rate `eta=1.0` and only
`epochs=5`. Does it still reach perfect accuracy on our separable blobs?
Plot the boundary.
"""))
a.append(md("**Solution:**"))
a.append(code("""
w2, b2 = train_perceptron(X, y, eta=1.0, epochs=5)
preds2 = np.array([step(np.dot(w2, xi) + b2) for xi in X])
print("accuracy:", (preds2 == y).mean())
plot_boundary_line(w2, b2, X0, X1, "Perceptron (eta=1.0, 5 epochs)")
# Because the data is linearly separable, the perceptron converges quickly;
# a different eta just rescales w/b and gives a (possibly tilted) valid line.
"""))

# ---- The XOR failure ----
a.append(md(r"""
---
## 6. Why a single neuron cannot solve XOR

Now the punchline that motivates the whole field. **XOR** ("exclusive or") is
the 4-point dataset

| $x_1$ | $x_2$ | label |
|------|------|-------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The two "1" points sit on one diagonal, the two "0" points on the other. **No
single straight line can separate them.** A single neuron only ever draws a
straight line ($w\cdot x + b = 0$), so it is *mathematically incapable* of
solving XOR. Let's watch it fail.
"""))
a.append(code("""
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

w_x, b_x = train_perceptron(X_xor, y_xor, eta=0.1, epochs=100)
preds_x = np.array([step(np.dot(w_x, xi) + b_x) for xi in X_xor])
print("predictions:", preds_x)
print("targets    :", y_xor)
print("accuracy   :", (preds_x == y_xor).mean())   # stuck around 0.5
"""))
a.append(code("""
# Visualize: there is NO straight line separating o from x
plt.figure(figsize=(5, 5))
for xi, yi in zip(X_xor, y_xor):
    marker = "x" if yi == 1 else "o"
    color  = "tab:blue" if yi == 1 else "tab:red"
    plt.scatter(xi[0], xi[1], marker=marker, color=color, s=200)
xs = np.linspace(-0.5, 1.5, 50)
ys = -(w_x[0] * xs + b_x) / (w_x[1] if w_x[1] != 0 else 1e-6)
plt.plot(xs, ys, "k--", label="best line the neuron found")
plt.title("XOR: no single line works")
plt.xlabel("x1"); plt.ylabel("x2")
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
The accuracy is stuck near $0.5$ (no better than guessing) and the line cannot
possibly cut the two classes apart, no matter how long we train.

**The fix:** stack neurons in layers. A *hidden layer* first bends the space
(applying several lines + nonlinearities), and then the output neuron *can*
separate the transformed points. That curved decision boundary is exactly what
we build in **notebook 13b** — and the tool that trains it is **backpropagation**,
the chain rule from Chapter 07 applied layer by layer.
"""))

a.append(md(r"""
### A note before 13b: from a loop to *matrices* (vectorization)

Look again at `train_perceptron`: it has a Python `for` loop that handles **one
point at a time** (`for xi, yi in zip(X, y)`). That is easy to read, but slow,
and it hides the linear-algebra structure. In 13b we switch to **vectorization**:
instead of looping over the $N$ data points, we stack them as the rows of a
matrix $X$ (shape $N\times n$) and let a **single matrix multiply** $XW$ process
the whole batch at once.

This is not just a speed trick — it is how you should *think*. One matrix
product = "apply this layer to every example simultaneously." Andrew Ng's advice:
**whenever you see yourself writing a loop over training examples, ask whether a
matrix operation can replace it.** NumPy then runs that operation in fast
compiled code, and the math reads as clean linear algebra.
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Leaky ReLU.** Implement $\text{leaky}(z) = z$ for $z>0$ and $0.01 z$
   otherwise, and add it to the activation plot. Why might a small negative
   slope help compared to plain ReLU?
2. **Track convergence.** Modify `train_perceptron` to record the number of
   misclassified points after each epoch, and plot that count versus epoch for
   the separable blobs. How many epochs until it hits zero?
3. **AND and OR are linearly separable.** Build the truth-table datasets for
   logical AND and OR and confirm a single perceptron learns each one
   perfectly (plot the boundaries). Contrast with XOR.
4. **Move the blobs together.** Reduce the gap between the two Gaussian
   centres until they overlap. At what point does the perceptron stop reaching
   100% accuracy, and why?
"""))

save(os.path.join(CH, "13a_perceptron.ipynb"), a)


# ===========================================================================
# Notebook 13b — A 2-layer neural network with backpropagation
# ===========================================================================
b = []

b.append(md(r"""
# Chapter 13b — A Neural Network from Scratch (Backpropagation)

In 13a a single neuron failed on XOR because it can only draw a straight line.
Here we build a **2-layer multilayer perceptron (MLP)** — one *hidden* layer
plus an output neuron — entirely in NumPy, and train it with **gradient
descent** powered by **backpropagation**.

This is the capstone of the course. Every ingredient is something you have
already met:

- the **forward pass** is matrix multiplication (Ch. 04) + activations (13a),
- the **loss** measures how wrong we are (Ch. 09, 11),
- **backpropagation** is the **chain rule** (Ch. 07) applied layer by layer,
- and we update the weights with **gradient descent** (Ch. 12).
"""))

b.append(md(r"""
## 1. Setup
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_deriv(a):
    # IMPORTANT: argument is the OUTPUT a = sigmoid(z), not z itself.
    return a * (1 - a)            # sigma'(z) = a(1-a)
"""))

b.append(md(r"""
## 2. A dataset that needs a hidden layer: two moons

We build the classic **two-moons** dataset: two interleaving crescents. Like
XOR, no straight line separates them — a curved boundary is required, so a
single neuron cannot do it but a hidden layer can.
"""))
b.append(code("""
def make_moons(n_per, noise=0.15):
    # Upper moon (label 0) and lower shifted moon (label 1).
    t = np.linspace(0, np.pi, n_per)
    # moon 0
    x0 = np.stack([np.cos(t),           np.sin(t)], axis=1)
    # moon 1: flip and shift so the two crescents interlock
    x1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    X = np.vstack([x0, x1])
    X = X + rng.normal(0, noise, size=X.shape)   # jitter
    y = np.array([0]*n_per + [1]*n_per)
    return X, y

X, y = make_moons(100)
y = y.reshape(-1, 1)                  # shape (200, 1) — a column vector
print("X shape:", X.shape, " y shape:", y.shape)
"""))
b.append(code("""
plt.figure(figsize=(5, 5))
plt.scatter(X[y[:, 0]==0, 0], X[y[:, 0]==0, 1], color="tab:red",  label="class 0")
plt.scatter(X[y[:, 0]==1, 0], X[y[:, 0]==1, 1], color="tab:blue", label="class 1")
plt.title("Two moons (needs a curved boundary)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.grid(True); plt.axis("equal")
plt.show()
"""))

b.append(md(r"""
## 3. Network architecture and shapes

Our network has:

- an **input layer** of size $2$ (the two coordinates),
- a **hidden layer** of size $H$ with sigmoid activation,
- an **output layer** of size $1$ with sigmoid (a probability of class 1).

We process all $N$ points at once by stacking them as rows of $X$ (shape
$N\times 2$). The parameters are two weight matrices and two bias vectors:

$$
W_1:\ (2 \times H), \quad b_1:\ (1 \times H), \qquad
W_2:\ (H \times 1), \quad b_2:\ (1 \times 1).
$$

We initialize weights with **small random numbers** (so neurons start
different) and biases at $0$.

> **Tip — always check your matrix dimensions.** Andrew Ng's single most useful
> debugging habit: after writing any layer, *write down the shape of every
> array and verify the products line up.* For a multiply $A\,B$ to be legal the
> inner dimensions must match: $(N\times 2)(2\times H) \to (N\times H)$. The
> overwhelming majority of neural-net bugs are shape mismatches, and almost all
> of them are caught by sketching $(\text{rows}\times\text{cols})$ next to each
> line before you run it. We print the shapes below — get into this habit.
"""))
b.append(code("""
N, n_in = X.shape       # N = 200 points, n_in = 2 features
H = 8                   # hidden layer size (try changing this!)
n_out = 1

# Small random init scaled by 0.5 keeps starting z values in a good range.
W1 = rng.normal(0, 0.5, size=(n_in, H))    # (2, 8)
b1 = np.zeros((1, H))                       # (1, 8)
W2 = rng.normal(0, 0.5, size=(H, n_out))   # (8, 1)
b2 = np.zeros((1, n_out))                   # (1, 1)

print("W1", W1.shape, " b1", b1.shape, " W2", W2.shape, " b2", b2.shape)
"""))

b.append(md(r"""
### Why *random* weights? (symmetry breaking)

Why not just start every weight at $0$, the way we did for the perceptron? Here
is Andrew Ng's classic warning: **if all the hidden weights are identical, all
the hidden units compute the same thing forever.**

Think it through. If every row of $W_1$ is the same, then every hidden unit sees
the same $z$, produces the same activation, and — crucially — receives the
*same* gradient during backprop. So they all update identically and stay
identical. Your $H$ hidden units behave like **one** unit; the extra capacity is
wasted. This is the **symmetry problem**, and the cure is to **break the symmetry
by initializing the weights randomly** (biases can safely start at $0$). The
tiny demo below makes the failure concrete.
"""))
b.append(code("""
# DEMO: all-zero init keeps every hidden unit identical.
W1_zero = np.zeros((n_in, H))      # every hidden unit has the same (zero) weights
b1_zero = np.zeros((1, H))
Z1_zero = X @ W1_zero + b1_zero    # all H columns are identical
A1_zero = sigmoid(Z1_zero)
# Are all hidden columns the same? Compare each column to the first one.
print("all hidden units identical with zero init? ",
      np.allclose(A1_zero, A1_zero[:, [0]]))   # True -> symmetry NOT broken

# Now random init: the hidden units differ, so they can learn different features.
A1_rand = sigmoid(X @ W1 + b1)
print("all hidden units identical with random init?",
      np.allclose(A1_rand, A1_rand[:, [0]]))   # False -> symmetry broken
"""))

b.append(md(r"""
## 4. The forward pass (matrix form)

Feeding the data forward, layer by layer:

$$
\begin{aligned}
Z_1 &= X W_1 + b_1 & &(N\times H)\\
A_1 &= \sigma(Z_1) & &(N\times H)\quad\text{hidden activations}\\
Z_2 &= A_1 W_2 + b_2 & &(N\times 1)\\
A_2 &= \sigma(Z_2) & &(N\times 1)\quad\text{predicted probabilities}
\end{aligned}
$$

The bias rows broadcast across all $N$ data rows automatically (Ch. 01).

**The forward pass as a computation graph — and why we *cache*.** Lay the four
lines out as a chain of simple operations:

$$
X \;\xrightarrow{\;W_1,b_1\;}\; Z_1 \;\xrightarrow{\;\sigma\;}\; A_1
  \;\xrightarrow{\;W_2,b_2\;}\; Z_2 \;\xrightarrow{\;\sigma\;}\; A_2 \;\to\; L.
$$

Going forward we don't just want the final answer $A_2$ — we also **save the
intermediate values** $Z_1, A_1, Z_2, A_2$ in a `cache`. Why keep them? Because
backpropagation will need them. The derivative of each link depends on the value
that flowed through it (e.g. $\sigma'$ is expressed via $A_1$). Rather than
recompute those values on the way back, we **store them once on the forward pass
and reuse them on the backward pass**. This forward-cache / backward-reuse split
is the heart of how backprop is organized — and exactly what `forward` returns
below as `cache`.
"""))
b.append(code("""
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1        # (N,2)@(2,H) -> (N,H)
    A1 = sigmoid(Z1)        # (N,H)
    Z2 = A1 @ W2 + b2       # (N,H)@(H,1) -> (N,1)
    A2 = sigmoid(Z2)        # (N,1) predicted probabilities
    cache = (Z1, A1, Z2, A2)
    return A2, cache

A2, cache = forward(X, W1, b1, W2, b2)
print("prediction shape:", A2.shape)        # (200, 1)
print("first 3 raw predictions:\\n", A2[:3])  # near 0.5 before training
"""))

b.append(md(r"""
## 5. The loss: binary cross-entropy

To measure how wrong the predictions $A_2 = \hat{y}$ are against the true
labels $y \in \{0,1\}$, we use **binary cross-entropy** (the natural loss for
the sigmoid, from Ch. 11), averaged over all $N$ points:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}
\Big[\, y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\,\Big].
$$

It is $0$ for a perfect confident prediction and grows as predictions move
away from the truth.
"""))
b.append(code("""
def bce_loss(y, y_hat):
    eps = 1e-9                          # avoid log(0)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

print("loss before training:", bce_loss(y, A2))   # around 0.69 = ln 2
"""))

b.append(md(r"""
## 6. Backpropagation — the chain rule, layer by layer

We want the gradients of the loss $L$ with respect to every parameter:
$\frac{\partial L}{\partial W_2}, \frac{\partial L}{\partial b_2},
\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial b_1}$.

**The big idea: push an *error signal* backwards.** Backpropagation runs the
computation graph in reverse. We compute, for each layer, a quantity

$$\delta = \frac{\partial L}{\partial Z}$$

— the gradient of the loss with respect to that layer's **pre-activation** $Z$.
Read $\delta$ as an *error signal*: **it measures how much each pre-activation is
responsible for the final loss.** A unit with large $|\delta|$ is "to blame" for
the error and will be adjusted a lot; a unit with $\delta\approx 0$ is already
fine and barely moves.

The whole algorithm is just two moves, repeated:

1. **turn an error signal into parameter gradients** for the current layer
   (e.g. $\frac{\partial L}{\partial W} = (\text{input to layer})^\top \delta$), and
2. **propagate the error signal one layer back** by sending it through the
   weights ($\delta\,W^\top$) and then through the activation's derivative
   ($\odot\,\sigma'$).

That is *all* backprop is — the **chain rule applied repeatedly**, right to
left, carrying $\delta$ from the loss back toward the inputs and reusing the
cached forward values at every step.

**The key simplification.** For sigmoid output *with* cross-entropy loss, the
two derivatives combine into a strikingly clean result (a standard calculus
exercise — multiply $\frac{\partial L}{\partial A_2}$ by
$\frac{\partial A_2}{\partial Z_2}=A_2(1-A_2)$ and watch terms cancel):

$$
\frac{\partial L}{\partial Z_2} \;=\; \frac{1}{N}\,(A_2 - y).
$$

From there we walk **backwards** using the chain rule. With
$\delta_2 = \frac{\partial L}{\partial Z_2}$ and
$\delta_1 = \frac{\partial L}{\partial Z_1}$:

$$
\begin{aligned}
\delta_2 &= \tfrac{1}{N}(A_2 - y)            & &(N\times 1)\\
\frac{\partial L}{\partial W_2} &= A_1^{\top}\,\delta_2 & &(H\times 1)\\
\frac{\partial L}{\partial b_2} &= \textstyle\sum_i (\delta_2)_i & &(1\times 1)\\[4pt]
\delta_1 &= (\delta_2\, W_2^{\top}) \odot \sigma'(Z_1)
          = (\delta_2\, W_2^{\top}) \odot A_1(1-A_1) & &(N\times H)\\
\frac{\partial L}{\partial W_1} &= X^{\top}\,\delta_1 & &(2\times H)\\
\frac{\partial L}{\partial b_1} &= \textstyle\sum_i (\delta_1)_i & &(1\times H)
\end{aligned}
$$

Here $\odot$ is elementwise multiplication. **Each line is one chain-rule
step**: how the loss changes through that layer's linear map, then through its
activation. Notice the pattern from the intuition above — $\delta_1$ is built by
taking the *next* layer's error $\delta_2$, sending it back through the weights
($\delta_2 W_2^\top$), and gating it by the local activation slope
($\odot\,\sigma'(Z_1)$). The error literally flows backward through the same
graph it came forward through.

> **Dimension-check tip (again!).** Every gradient has the **same shape as the
> thing it differentiates**: $\frac{\partial L}{\partial W_2}$ is $(H\times 1)$,
> exactly like $W_2$; $\frac{\partial L}{\partial W_1}$ is $(2\times H)$, like
> $W_1$. If a gradient comes out the wrong shape, you have a bug. Read the shape
> comments in the code below and confirm each line for yourself — this single
> habit catches almost every backprop mistake.
"""))
b.append(code("""
def backward(X, y, cache, W2):
    Z1, A1, Z2, A2 = cache
    N = X.shape[0]

    # ----- output layer -----
    dZ2 = (A2 - y) / N           # (N,1)  combined sigmoid+BCE derivative
    dW2 = A1.T @ dZ2             # (H,N)@(N,1) -> (H,1)
    db2 = dZ2.sum(axis=0, keepdims=True)   # (1,1)

    # ----- hidden layer (chain rule back through W2 and sigmoid) -----
    dA1 = dZ2 @ W2.T            # (N,1)@(1,H) -> (N,H)
    dZ1 = dA1 * sigmoid_deriv(A1)          # (N,H) elementwise * A1(1-A1)
    dW1 = X.T @ dZ1            # (2,N)@(N,H) -> (2,H)
    db1 = dZ1.sum(axis=0, keepdims=True)   # (1,H)

    return dW1, db1, dW2, db2
"""))

b.append(md(r"""
### Gradient check (does our calculus match the numbers?)

Before trusting backprop, compare one analytic gradient entry against a tiny
numerical perturbation $\frac{L(W+h)-L(W-h)}{2h}$. They should agree to many
decimals.
"""))
b.append(code("""
dW1, db1, dW2, db2 = backward(X, y, cache, W2)

# numerically perturb a single entry of W2
i, j = 0, 0
h = 1e-5
W2p = W2.copy(); W2p[i, j] += h
W2m = W2.copy(); W2m[i, j] -= h
Lp, _ = forward(X, W1, b1, W2p, b2)
Lm, _ = forward(X, W1, b1, W2m, b2)
numeric = (bce_loss(y, Lp) - bce_loss(y, Lm)) / (2 * h)

print("analytic dW2[0,0]:", dW2[i, j])
print("numeric  dW2[0,0]:", numeric)
print("difference       :", abs(dW2[i, j] - numeric))   # should be tiny
"""))

b.append(md(r"""
## 7. Training with gradient descent

Now the full loop (Ch. 12). Repeat many times:

1. **forward** to get predictions and the loss,
2. **backward** to get all four gradients,
3. **descend**: subtract `learning_rate * gradient` from each parameter.

$$W \leftarrow W - \eta\,\frac{\partial L}{\partial W}.$$

We record the loss each step to plot the **loss curve**.
"""))
b.append(code("""
# fresh initialization so this cell is self-contained
W1 = rng.normal(0, 0.5, size=(n_in, H))
b1 = np.zeros((1, H))
W2 = rng.normal(0, 0.5, size=(H, n_out))
b2 = np.zeros((1, n_out))

eta = 1.0           # learning rate
epochs = 4000       # gradient-descent steps
losses = []

for step_i in range(epochs):
    A2, cache = forward(X, W1, b1, W2, b2)          # 1. forward
    losses.append(bce_loss(y, A2))
    dW1, db1, dW2, db2 = backward(X, y, cache, W2)  # 2. backward
    W1 -= eta * dW1                                  # 3. descend
    b1 -= eta * db1
    W2 -= eta * dW2
    b2 -= eta * db2

print("loss start:", losses[0])
print("loss end  :", losses[-1])    # should be much smaller
"""))
b.append(code("""
plt.figure(figsize=(7, 4))
plt.plot(losses)
plt.title("Training loss (binary cross-entropy)")
plt.xlabel("gradient-descent step"); plt.ylabel("loss")
plt.grid(True)
plt.show()
"""))

b.append(md(r"""
The loss falls steadily — gradient descent is working. Let's measure accuracy:
we threshold the predicted probability at $0.5$.
"""))
b.append(code("""
A2, _ = forward(X, W1, b1, W2, b2)
pred_labels = (A2 > 0.5).astype(int)
accuracy = (pred_labels == y).mean()
print("training accuracy:", accuracy)
"""))

b.append(md(r"""
## 8. The learned (curved) decision boundary

Finally, the payoff. We evaluate the network on a dense grid covering the
plane and colour each region by its predicted class. The boundary is **curved**
— something the single neuron of 13a could never produce.
"""))
b.append(code("""
# build a grid of points covering the data
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]      # (90000, 2)

probs, _ = forward(grid, W1, b1, W2, b2)
zz = probs.reshape(xx.shape)              # predicted prob over the grid

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.7)
plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)  # boundary
plt.scatter(X[y[:, 0]==0, 0], X[y[:, 0]==0, 1], color="tab:red",  edgecolor="k", label="class 0")
plt.scatter(X[y[:, 0]==1, 0], X[y[:, 0]==1, 1], color="tab:blue", edgecolor="k", label="class 1")
plt.title("Learned decision boundary (2-layer network)")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(); plt.show()
"""))

b.append(md(r"""
The black curve weaves between the two crescents. A network of simple neurons,
trained by gradient descent on gradients computed with the chain rule, has
learned a **nonlinear** classifier from scratch. That is the whole idea of deep
learning, and you built it yourself.
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Wrap the whole pipeline into a single function `train(H, eta, epochs)` that
returns the final accuracy, so we can experiment with the **hidden-layer
size** $H$. Try $H = 1, 2, 4, 16$ and print the accuracy for each. What is the
smallest $H$ that classifies the moons well?
"""))
b.append(md("**Solution:**"))
b.append(code("""
def train(H, eta=1.0, epochs=4000):
    g = np.random.default_rng(0)               # fresh, reproducible
    W1 = g.normal(0, 0.5, size=(2, H)); b1 = np.zeros((1, H))
    W2 = g.normal(0, 0.5, size=(H, 1)); b2 = np.zeros((1, 1))
    for _ in range(epochs):
        A2, cache = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, cache, W2)
        W1 -= eta * dW1; b1 -= eta * db1
        W2 -= eta * dW2; b2 -= eta * db2
    A2, _ = forward(X, W1, b1, W2, b2)
    return ((A2 > 0.5).astype(int) == y).mean()

for H_try in [1, 2, 4, 16]:
    print(f"H = {H_try:2d}  ->  accuracy = {train(H_try):.3f}")
# H=1 behaves like a single neuron (poor); a few hidden units already suffice.
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Replace the **hidden** activation with $\tanh$ (keep sigmoid on the output).
The derivative is $\tanh'(z) = 1 - \tanh(z)^2$. Retrain and report the final
loss and accuracy. *Hint:* you only need to change the activation and its
derivative in a copy of `forward`/`backward`.
"""))
b.append(md("**Solution:**"))
b.append(code("""
def forward_tanh(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)            # tanh hidden activation
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)           # sigmoid output (still a probability)
    return A2, (Z1, A1, Z2, A2)

def backward_tanh(X, y, cache, W2):
    Z1, A1, Z2, A2 = cache
    N = X.shape[0]
    dZ2 = (A2 - y) / N
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (1 - A1**2)    # tanh'(z) = 1 - tanh(z)^2 = 1 - A1^2
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2

g = np.random.default_rng(0)
W1 = g.normal(0, 0.5, size=(2, H)); b1 = np.zeros((1, H))
W2 = g.normal(0, 0.5, size=(H, 1)); b2 = np.zeros((1, 1))
for _ in range(4000):
    A2, cache = forward_tanh(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward_tanh(X, y, cache, W2)
    W1 -= 1.0 * dW1; b1 -= 1.0 * db1
    W2 -= 1.0 * dW2; b2 -= 1.0 * db2
A2, _ = forward_tanh(X, W1, b1, W2, b2)
print("final loss    :", bce_loss(y, A2))
print("final accuracy:", ((A2 > 0.5).astype(int) == y).mean())
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Solve XOR end-to-end.** Train this 2-layer network on the XOR dataset
   from 13a (`X = [[0,0],[0,1],[1,0],[1,1]]`, `y = [0,1,1,0]`). Use a small
   hidden layer ($H=2$ or $H=4$) and confirm it reaches 100% accuracy —
   succeeding exactly where the single neuron failed.
2. **Learning-rate sweep.** Plot the loss curves for `eta` in
   $\{0.1, 0.3, 1.0, 3.0\}$ on the same axes. Which converges fastest? Does any
   value diverge or oscillate?
3. **Add a second hidden layer.** Extend the network to sizes
   $2 \to H_1 \to H_2 \to 1$. Write the new forward pass and derive/implement
   the extra backprop step. Does the deeper network fit the moons better?
4. **MSE vs cross-entropy.** Swap the loss for mean-squared error
   $L=\frac1N\sum(\hat y-y)^2$. Re-derive $\partial L/\partial Z_2$ (now you
   must *not* drop the $\sigma'(Z_2)$ factor) and compare training speed
   against cross-entropy.
"""))

save(os.path.join(CH, "13b_neural_network.ipynb"), b)
