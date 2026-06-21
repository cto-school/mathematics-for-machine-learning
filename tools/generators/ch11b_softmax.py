"""Generator for Chapter 11b — Softmax Regression (Multi-class Classification).

Run from anywhere:  python tools/generators/ch11b_softmax.py
Produces one notebook in 11-logistic-regression-classification/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "11-logistic-regression-classification")


# ---------------------------------------------------------------------------
# Notebook 11b — Softmax Regression (Multi-class Classification)
# ---------------------------------------------------------------------------
nb = []

nb.append(md(r"""
# Chapter 11b — Softmax Regression (Multi-class Classification)

In **11a** we built logistic regression: the sigmoid squashes a single score
into a probability $p = P(y=1\mid x)$, perfect when there are exactly **two**
classes. But the world rarely comes in twos — handwritten digits have 10
classes, an image classifier might have 1000.

> Given an input $x$, which of $K$ **classes** does it belong to?
> (digit 0–9, species A/B/C, the type of an iris flower)

The clean generalization of the sigmoid to $K$ classes is the **softmax**
function. This notebook builds **softmax regression** (also called *multinomial
logistic regression*) entirely by hand with NumPy: the softmax, the model, the
cross-entropy cost, its surprisingly clean gradient (verified with a numerical
check), a from-scratch training loop, and a picture of the multi-class decision
regions.

Run each cell with **Shift + Enter**.
"""))

nb.append(md(r"""
## 1. From sigmoid (2 classes) to softmax ($K$ classes)

Recall the sigmoid from 11a: one score $z$, one probability $p = \sigma(z)$, and
the "other" class gets $1 - p$. With two classes a *single* number is enough.

With $K$ classes we instead produce a **vector of $K$ scores**
$z = (z_1, z_2, \dots, z_K)$, one per class, and we want to turn it into a
**probability distribution**: $K$ non-negative numbers that sum to 1. The
softmax does exactly that:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

- The exponential $e^{z_k}$ makes every entry **positive**.
- Dividing by the sum $\sum_j e^{z_j}$ makes the entries **add up to 1**.
- A bigger score $z_k$ gives a bigger probability — the name "soft-max" is
  because it is a smooth, differentiable stand-in for "pick the maximum".

Let's compute one by hand.
"""))
nb.append(code("""
import numpy as np
import matplotlib.pyplot as plt

def softmax_naive(z):
    # The textbook formula, applied to a 1D score vector.
    e = np.exp(z)               # exponentiate every score
    return e / np.sum(e)        # normalize so the entries sum to 1

scores = np.array([2.0, 1.0, -1.0])     # three class scores
probs = softmax_naive(scores)
print("scores        :", scores)
print("probabilities :", np.round(probs, 4))
print("sum of probs  :", round(float(np.sum(probs)), 6), " (should be 1.0)")
print("argmax (chosen class):", int(np.argmax(probs)))
"""))

nb.append(md(r"""
The largest score (here $z_1 = 2$) gets the largest probability, and the three
probabilities sum to 1 — a genuine distribution over the classes.
"""))

nb.append(md(r"""
## 2. A numerically *stable* softmax

The naive version has a hidden danger: $e^{z_k}$ **overflows** for large scores.
`np.exp(1000)` is `inf`, and `inf / inf` is `nan`. Training routinely produces
big scores, so this would crash.

The fix uses a simple algebraic fact. Subtract the same constant $c$ from every
score and the softmax is **unchanged**:

$$\frac{e^{z_k - c}}{\sum_j e^{z_j - c}}
 = \frac{e^{z_k}\,e^{-c}}{\sum_j e^{z_j}\,e^{-c}}
 = \frac{e^{z_k}}{\sum_j e^{z_j}}.$$

The $e^{-c}$ factors cancel top and bottom. Choosing $c = \max_k z_k$ makes the
largest exponent exactly $0$, so $e^{0}=1$ is the biggest value we ever
exponentiate — no overflow. This is the version everyone actually uses.
"""))
nb.append(code("""
def softmax(Z):
    # Numerically stable, row-wise softmax for a matrix Z of shape (N, K):
    # each ROW is one example's K class scores.
    Z = np.atleast_2d(Z)                          # treat a 1D input as one row
    Z_shift = Z - Z.max(axis=1, keepdims=True)    # subtract each row's max
    e = np.exp(Z_shift)                           # now the biggest exp is e^0 = 1
    return e / e.sum(axis=1, keepdims=True)       # normalize each row to sum to 1

# Same answer as the naive version on safe inputs...
print("stable :", np.round(softmax(scores), 4))

# ...but the naive one breaks on large scores, while the stable one survives:
big = np.array([1000.0, 1001.0, 1002.0])
print("naive on big scores :", softmax_naive(big))   # nan nan nan
print("stable on big scores:", np.round(softmax(big), 4))
"""))

nb.append(md(r"""
The naive softmax returns `nan` on the big scores; the stable one returns a
sensible distribution. From here on we always use the stable `softmax`.
"""))

nb.append(md(r"""
## 3. The model

Each class $k$ gets its own weight vector and bias. Stacking those columns gives
a weight **matrix** $W$ of shape `(n_features, K)` and a bias **row** $b$ of
length $K$. For a whole data matrix $X$ of shape `(N, n_features)`:

$$Z = X W + b \quad (\text{shape } N\times K), \qquad P = \text{softmax}(Z)\ \text{row-wise}.$$

Row $i$ of $P$ is the predicted probability distribution over the $K$ classes
for example $i$. We **predict** the class with the highest probability,
$\hat y_i = \arg\max_k P_{ik}$.

Compare with 11a: there $W$ was a single vector and softmax was the sigmoid.
Softmax regression is the *same idea*, widened from one column to $K$.
"""))
nb.append(code("""
def scores_matrix(X, W, b):
    # Z = X W + b. NumPy "broadcasts" the length-K bias across all N rows.
    return X @ W + b

def predict_proba(X, W, b):
    return softmax(scores_matrix(X, W, b))     # (N, K) probabilities

def predict(X, W, b):
    return np.argmax(predict_proba(X, W, b), axis=1)   # (N,) class labels

# Tiny hand example: 2 examples, 2 features, K = 3 classes.
X_demo = np.array([[1.0, 2.0],
                   [-1.0, 0.5]])
W_demo = np.array([[0.5, -0.2, 0.1],     # shape (n_features=2, K=3)
                   [0.3,  0.8, -0.5]])
b_demo = np.array([0.0, 0.1, -0.1])      # one bias per class

P_demo = predict_proba(X_demo, W_demo, b_demo)
print("probabilities (each row sums to 1):")
print(np.round(P_demo, 4))
print("row sums   :", np.round(P_demo.sum(axis=1), 6))
print("predictions:", predict(X_demo, W_demo, b_demo))
"""))

nb.append(md(r"""
## 4. One-hot encoding of the labels

The labels arrive as integers $y_i \in \{0, 1, \dots, K-1\}$. For the cost and
gradient it is far tidier to represent each label as a **one-hot** row: a vector
of $K$ zeros with a single $1$ in the position of the true class.

$$y_i = 2 \;\;\longrightarrow\;\; (0,\,0,\,1,\,0,\dots) \quad(\text{the } 1 \text{ is in slot } 2).$$

The result $Y$ has shape `(N, K)`, matching $P$ — so we can compare predictions
and truth entry-by-entry.
"""))
nb.append(code("""
def one_hot(y, K):
    # y: integer labels, shape (N,).  Returns Y of shape (N, K).
    y = np.asarray(y, dtype=int)
    Y = np.zeros((y.shape[0], K))
    Y[np.arange(y.shape[0]), y] = 1.0      # set a 1 at column = the label
    return Y

y_demo = np.array([0, 2, 1, 2])
print("labels  :", y_demo)
print("one-hot :")
print(one_hot(y_demo, K=3))
"""))

nb.append(md(r"""
## 5. The cost: categorical cross-entropy

For two classes we used the log-loss. Its multi-class cousin is **categorical
cross-entropy**. For one example, only the probability of the *true* class
matters: if the truth is class $y_i$, the loss is $-\log p_{i,\,y_i}$. Averaged
over $N$ examples:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,\,y_i}.$$

Using the one-hot matrix $Y$ (which is $1$ only at the true class) we can write
the same thing as a clean sum over all entries:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} Y_{ik}\,\log P_{ik}.$$

As in 11a we **clip** the probabilities away from exactly $0$ so `np.log` never
sees a zero (which is $-\infty$).
"""))
nb.append(code("""
def cross_entropy(P, Y, eps=1e-12):
    # P: predicted probs (N, K).  Y: one-hot truth (N, K).
    P = np.clip(P, eps, 1.0)                  # keep log away from 0
    return -np.mean(np.sum(Y * np.log(P), axis=1))

# Sanity check: with K classes and all-zero weights every prediction is the
# uniform distribution (1/K for each class), so the loss should be -log(1/K).
K_demo = 3
P_uniform = np.full((4, K_demo), 1.0 / K_demo)
Y_demo = one_hot(y_demo, K_demo)
print("loss at uniform probs:", round(cross_entropy(P_uniform, Y_demo), 4))
print("-log(1/K)            :", round(float(-np.log(1.0 / K_demo)), 4))
"""))

nb.append(md(r"""
## 6. The gradient (the clean result)

Just as in 11a the sigmoid + log-loss gave the tidy gradient
$\frac1n X^\top(p-y)$, the softmax + cross-entropy pairing collapses to an
equally clean form. With $P = \text{softmax}(XW+b)$ and one-hot truth $Y$,
define the **error matrix**

$$\mathrm{d}Z = \frac{1}{N}\,(P - Y) \quad (\text{shape } N\times K).$$

Then the gradients are

$$\boxed{\;\frac{\partial L}{\partial W} = X^{\top}\,\mathrm{d}Z, \qquad
\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \mathrm{d}Z_{i}\;}$$

In words: the gradient is again driven by the **prediction error** $P - Y$
(predicted distribution minus the truth), spread back onto the weights by
$X^\top$. The bias gradient is just the column-sum of the error. We won't grind
through the algebra, but it is the exact multi-class analogue of 11a.

The gradient-descent updates (learning rate $\alpha$) are then

$$W \leftarrow W - \alpha\,\frac{\partial L}{\partial W}, \qquad
b \leftarrow b - \alpha\,\frac{\partial L}{\partial b}.$$
"""))
nb.append(code("""
def gradients(X, Y, W, b):
    N = X.shape[0]
    P = predict_proba(X, W, b)        # (N, K)
    dZ = (P - Y) / N                  # the prediction error, scaled by 1/N
    dW = X.T @ dZ                     # (n_features, K)
    db = dZ.sum(axis=0)              # (K,)
    return dW, db
"""))

nb.append(md(r"""
### Verifying the gradient numerically

Before trusting the formula, we run a **gradient check**: nudge each parameter
by a tiny $h$ and compare the analytic gradient against the central-difference
estimate $\dfrac{L(\theta+h) - L(\theta-h)}{2h}$. They should match to many
decimals.
"""))
nb.append(code("""
rng_check = np.random.default_rng(1)
n_feat, Kc = 4, 3
Xc = rng_check.normal(size=(6, n_feat))           # 6 random examples
yc = rng_check.integers(0, Kc, size=6)            # random labels 0..K-1
Yc = one_hot(yc, Kc)
Wc = rng_check.normal(size=(n_feat, Kc))          # random weights
bc = rng_check.normal(size=Kc)                     # random biases

def loss_at(W, b):
    return cross_entropy(predict_proba(Xc, W, b), Yc)

dW_analytic, db_analytic = gradients(Xc, Yc, Wc, bc)

# numerical gradient w.r.t. W
h = 1e-6
dW_numeric = np.zeros_like(Wc)
for i in range(Wc.shape[0]):
    for j in range(Wc.shape[1]):
        step = np.zeros_like(Wc); step[i, j] = h
        dW_numeric[i, j] = (loss_at(Wc + step, bc) - loss_at(Wc - step, bc)) / (2 * h)

print("max |dW analytic - dW numeric|:", np.max(np.abs(dW_analytic - dW_numeric)))

# numerical gradient w.r.t. b
db_numeric = np.zeros_like(bc)
for j in range(bc.shape[0]):
    step = np.zeros_like(bc); step[j] = h
    db_numeric[j] = (loss_at(Wc, bc + step) - loss_at(Wc, bc - step)) / (2 * h)

print("max |db analytic - db numeric|:", np.max(np.abs(db_analytic - db_numeric)))
"""))

nb.append(md(r"""
Both differences are tiny (around $10^{-9}$), so our hand-coded gradient is
correct. Now we can train with confidence.
"""))

nb.append(md(r"""
## 7. A 2D, three-class dataset

Let's make data we can actually see: **three Gaussian blobs**, one per class,
placed at the corners of a triangle.
"""))
nb.append(code("""
rng = np.random.default_rng(0)        # fixed seed -> reproducible data

K = 3                                  # number of classes
n_per = 100                            # points per class
centers = np.array([[-2.0, -2.0],      # class 0
                    [ 2.0, -2.0],      # class 1
                    [ 0.0,  2.5]])     # class 2

X_list, y_list = [], []
for k in range(K):
    X_list.append(rng.normal(loc=centers[k], scale=0.8, size=(n_per, 2)))
    y_list.append(np.full(n_per, k))

X = np.vstack(X_list)                   # shape (300, 2)
y = np.concatenate(y_list)             # labels 0,0,...,1,1,...,2,2,...
Y = one_hot(y, K)                       # (300, 3) for the cost/gradient

print("X shape:", X.shape, " y shape:", y.shape, " Y shape:", Y.shape)

markers = ["o", "^", "s"]
for k in range(K):
    pts = X[y == k]
    plt.scatter(pts[:, 0], pts[:, 1], marker=markers[k], label=f"class {k}")
plt.title("Three Gaussian blobs (the data)")
plt.xlabel("feature x1"); plt.ylabel("feature x2")
plt.legend(); plt.grid(True); plt.show()
"""))

nb.append(md(r"""
Note we keep the bias $b$ **separate** this time (rather than the bias-column
trick from 11a) — with a weight *matrix* it reads more clearly to carry $b$ as
its own length-$K$ vector.
"""))

nb.append(md(r"""
## 8. Training from scratch with gradient descent

Start from zero weights, step downhill repeatedly, and record the loss so we can
watch it fall.
"""))
nb.append(code("""
n_features = X.shape[1]                 # 2
W = np.zeros((n_features, K))           # start all weights at 0
b = np.zeros(K)                         # start all biases at 0

alpha = 0.5                             # learning rate (step size)
n_epochs = 2000                         # number of gradient-descent steps

history = []                            # store the loss each step
for t in range(n_epochs):
    dW, db = gradients(X, Y, W, b)
    W = W - alpha * dW                  # the update rule
    b = b - alpha * db
    history.append(cross_entropy(predict_proba(X, W, b), Y))

print("final loss:", round(history[-1], 4))
print("final W:\\n", np.round(W, 3))
print("final b:", np.round(b, 3))
"""))

nb.append(md(r"""
### The loss curve

A healthy run shows the cross-entropy dropping quickly, then leveling off.
"""))
nb.append(code("""
plt.plot(history)
plt.title("Categorical cross-entropy during gradient descent")
plt.xlabel("epoch")
plt.ylabel("loss L")
plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
## 9. The multi-class decision regions

With two classes the boundary was a single line. With $K$ classes the plane is
carved into $K$ **regions** — at every point we predict the class with the
highest probability. To draw them we evaluate the model on a fine grid and color
each grid point by its predicted class (`np.argmax`), then scatter the data on
top.
"""))
nb.append(code("""
# build a grid covering the data
pad = 1.0
x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                     np.linspace(x2_min, x2_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]    # (90000, 2): every grid point
grid_pred = predict(grid, W, b).reshape(xx.shape)   # class at each grid point

plt.contourf(xx, yy, grid_pred, alpha=0.3, levels=np.arange(K + 1) - 0.5)
for k in range(K):
    pts = X[y == k]
    plt.scatter(pts[:, 0], pts[:, 1], marker=markers[k],
                edgecolor="k", label=f"class {k}")
plt.title("Softmax decision regions (color = predicted class)")
plt.xlabel("feature x1"); plt.ylabel("feature x2")
plt.legend(); plt.grid(True); plt.show()
"""))

nb.append(md(r"""
The plane is split into three colored regions. Their borders are **straight
lines** — softmax regression, like logistic regression, is a *linear*
classifier; it just draws several lines at once.
"""))

nb.append(md(r"""
## 10. Accuracy from scratch

Accuracy is the fraction of examples whose predicted class matches the truth.
"""))
nb.append(code("""
y_pred = predict(X, W, b)
accuracy = np.mean(y_pred == y)
print("accuracy:", round(float(accuracy), 4),
      "  (", int(np.sum(y_pred == y)), "of", len(y), "correct )")
"""))

nb.append(md(r"""
## 11. With $K=2$, softmax *is* logistic regression

Softmax is a strict generalization: when there are only two classes it reduces
to the sigmoid from 11a. The reason is the same cancellation we used for
stability. For two scores $z_0, z_1$,

$$\text{softmax}(z)_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}}
 = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0).$$

So the probability of class 1 is just a sigmoid of the score *difference*. The
extra weight vector is redundant — only the difference matters — which is why
for two classes we get away with a single weight vector. Let's confirm it
numerically.
"""))
nb.append(code("""
z0, z1 = 0.4, 1.1
soft_p1 = softmax(np.array([z0, z1]))[0, 1]     # softmax prob of class 1
sig_p1 = 1.0 / (1.0 + np.exp(-(z1 - z0)))        # sigmoid of the difference
print("softmax class-1 prob :", round(float(soft_p1), 6))
print("sigmoid of (z1 - z0) :", round(float(sig_p1), 6))
"""))

# ---- Exercise 1 (with solution) ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Verify the two defining properties of the stable `softmax` on a random score
matrix of shape `(5, 4)`: every entry is in $(0, 1)$, and **every row sums to
1**. Also check it agrees with `softmax_naive` on the first row.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
rng_ex = np.random.default_rng(7)
Z = rng_ex.normal(size=(5, 4))
P = softmax(Z)

print("all entries in (0,1):", bool(np.all((P > 0) & (P < 1))))
print("row sums            :", np.round(P.sum(axis=1), 6))   # all 1.0
print("matches naive (row 0):",
      np.allclose(P[0], softmax_naive(Z[0])))
"""))

# ---- Exercise 2 (with solution) ----
nb.append(md(r"""
## ✍️ Exercise 2 (solution included)

The softmax is **shift-invariant**: adding the same constant to every score in a
row leaves the probabilities unchanged. Confirm this by adding $100$ to one row
of scores and checking the softmax output is identical (this is exactly *why*
the stable version is allowed to subtract the row-max).
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
z = np.array([2.0, 0.5, -1.0, 3.0])
p_before = softmax(z)
p_after = softmax(z + 100.0)          # add the same constant to every entry

print("softmax(z)       :", np.round(p_before, 6))
print("softmax(z + 100) :", np.round(p_after, 6))
print("identical?       :", np.allclose(p_before, p_after))
"""))

# ---- Exercise 3 (with solution) ----
nb.append(md(r"""
## ✍️ Exercise 3 (solution included)

Train/test split. Shuffle the 300 points, train softmax regression on the first
80% (240 points), then report accuracy on the held-out 20% it never saw. Reuse
the training loop from Section 8.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
rng_split = np.random.default_rng(123)
perm = rng_split.permutation(len(y))          # a random ordering of all indices
n_train = int(0.8 * len(y))
train_idx, test_idx = perm[:n_train], perm[n_train:]

X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx], y[test_idx]
Y_tr = one_hot(y_tr, K)

W2 = np.zeros((X.shape[1], K)); b2 = np.zeros(K)
for t in range(2000):
    dW, db = gradients(X_tr, Y_tr, W2, b2)
    W2 = W2 - 0.5 * dW
    b2 = b2 - 0.5 * db

acc_tr = np.mean(predict(X_tr, W2, b2) == y_tr)
acc_te = np.mean(predict(X_te, W2, b2) == y_te)
print("train accuracy:", round(float(acc_tr), 4))
print("test  accuracy:", round(float(acc_te), 4))
"""))

# ---- Homework (no solutions) ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **More classes.** Add a fourth blob (class 3) at a new center, set `K = 4`,
   regenerate the one-hot labels, retrain, and redraw the decision regions and
   accuracy. Does the recipe change at all?
2. **Overlap.** Move the three centers closer together (e.g. scale the `centers`
   array by $0.4$) so the blobs overlap. Retrain and report the new accuracy.
   Why does it drop?
3. **Confusion counts.** After training, build a $K\times K$ table whose entry
   $(a, b)$ counts how many true-class-$a$ points were predicted as class $b$
   (a *confusion matrix*). Which classes get confused most?
4. **Learning rate sweep.** Re-run training for
   $\alpha \in \{0.01, 0.1, 0.5, 2.0\}$ (each for 500 epochs from zero weights)
   and plot all four loss curves on one figure. Which converges fastest, and
   does any value misbehave?
"""))

save(os.path.join(CH, "11b_softmax_multiclass.ipynb"), nb)
