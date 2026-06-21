"""Generator for Chapter 11 — Logistic Regression & Classification.

Run from anywhere:  python tools/generators/ch11_logistic_regression.py
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
# Notebook 11 — Logistic Regression & Classification
# ---------------------------------------------------------------------------
nb = []

nb.append(md(r"""
# Chapter 11 — Logistic Regression & Classification

In Chapter 10 we fit a straight line to predict a **number** (a price, a
temperature). That is *regression*. Now we ask a different question:

> Given an input $x$, which of two **classes** does it belong to?
> (spam / not-spam, malignant / benign, pass / fail)

This is **classification**. The output is no longer a real number but a
**label** $y \in \{0, 1\}$. The workhorse model for the two-class case is
**logistic regression**, which — despite its name — is a *classification*
method. We will build it entirely by hand with NumPy: the model, the cost,
its gradient, and a from-scratch gradient-descent training loop.

Run each cell with **Shift + Enter**.
"""))

nb.append(md(r"""
## 1. Classification vs. regression

| | Regression | Classification |
|---|---|---|
| Output $y$ | a real number ($-\infty,\infty$) | a label, e.g. $\{0,1\}$ |
| Example | "house costs \$340k" | "email is spam (1) or not (0)" |
| Model output we want | the value itself | a **probability** $p=P(y=1\mid x)$ |

We would *like* our model to output a probability $p \in (0,1)$. But a linear
combination $w\cdot x + b$ can be any real number — large positive, large
negative. We need a function that **squashes** any real number into $(0,1)$.
That function is the **sigmoid**.
"""))

nb.append(md(r"""
## 2. The sigmoid function

The sigmoid (or *logistic*) function is

$$\sigma(z) = \frac{1}{1 + e^{-z}}.$$

Key facts:

- It maps **any** real number $z$ to a value in $(0, 1)$ — perfect for a
  probability.
- $\sigma(0) = 1/2$.
- As $z \to +\infty$, $\sigma(z) \to 1$; as $z \to -\infty$, $\sigma(z) \to 0$.
- It has a clean derivative: $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$.

Let us plot it.
"""))
nb.append(code("""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    # 1 / (1 + e^{-z}); works element-wise on a NumPy array
    return 1.0 / (1.0 + np.exp(-z))

z = np.linspace(-8, 8, 200)      # 200 points from -8 to 8
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color="gray", linestyle="--")   # the 0.5 line
plt.axvline(0.0, color="gray", linestyle="--")   # z = 0
plt.title("The sigmoid  sigma(z) = 1 / (1 + e^{-z})")
plt.xlabel("z")
plt.ylabel("sigma(z)  (a probability in (0,1))")
plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
Notice the **S-shape**: far to the left it flattens near 0, far to the right it
flattens near 1, and it passes smoothly through $1/2$ at $z=0$. Whatever real
number we feed in, we get back a probability.
"""))

nb.append(md(r"""
## 3. The model

Logistic regression first forms a linear score (sometimes called the *logit*)

$$z = w \cdot x + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b,$$

and then squashes it through the sigmoid to get a probability:

$$p = \sigma(w \cdot x + b) = P(y = 1 \mid x).$$

- $w$ is the **weight** vector (one number per feature).
- $b$ is the **bias** (a single number, like the intercept in a line).
- $p$ near 1 means "confident class 1"; $p$ near 0 means "confident class 0".

**Bias trick.** To avoid carrying $b$ separately, we glue a column of 1's onto
the data. If $X$ has the 1's column, then $z = X\theta$ where
$\theta = (b, w_1, \dots, w_d)$ holds the bias *and* the weights together. This
keeps the linear algebra tidy.
"""))
nb.append(code("""
# A tiny example by hand (one data point, two features).
x = np.array([2.0, -1.0])      # the features
w = np.array([0.5, 1.5])       # some weights
b = 0.3                        # the bias

z = w @ x + b                  # @ is the dot product:  0.5*2 + 1.5*(-1) + 0.3
p = sigmoid(z)
print("score z =", z)
print("probability p = sigma(z) =", round(float(p), 4))
print("predicted class =", 1 if p >= 0.5 else 0)
"""))

nb.append(md(r"""
## 4. The decision boundary

To turn a probability into a **decision**, we threshold at $1/2$:

$$\hat{y} = \begin{cases} 1 & \text{if } p \ge 0.5 \\ 0 & \text{otherwise.}\end{cases}$$

Because $\sigma(z) \ge 0.5$ exactly when $z \ge 0$, the decision rule is simply

$$\hat{y} = 1 \iff w\cdot x + b \ge 0.$$

The set where $w\cdot x + b = 0$ is a **straight line** (in 2D) — the
**decision boundary**. On one side the model predicts 1, on the other it
predicts 0. Logistic regression is a *linear* classifier: its boundary is flat.
"""))

nb.append(md(r"""
## 5. Why not least squares? The log-loss

For regression we minimized the mean squared error (MSE). Could we just do the
same with $p$? It turns out MSE composed with the sigmoid is **non-convex**
(lots of flat regions and local dips), so gradient descent gets stuck and
learns slowly.

Instead we use the **cross-entropy** / **log-loss**, which comes from maximum
likelihood. For a single example with true label $y$ and prediction $p$:

$$\text{loss} = -\big[\,y\log p + (1-y)\log(1-p)\,\big].$$

Read it as two cases:

- If $y = 1$: loss $= -\log p$. Predict $p\to 1$ and loss $\to 0$; predict
  $p\to 0$ and loss $\to \infty$ (heavily punished for confident wrong answer).
- If $y = 0$: loss $= -\log(1-p)$. Symmetric: confidently saying 1 is punished.

Averaged over $n$ examples, the cost is

$$J(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\Big[ y_i \log p_i + (1-y_i)\log(1-p_i)\Big],
\qquad p_i = \sigma(X_i\theta).$$

This cost **is** convex in $\theta$, so gradient descent reliably finds the
minimum.
"""))
nb.append(code("""
# Visualize WHY log-loss is a good teacher when the true label is y = 1.
p = np.linspace(0.001, 0.999, 200)     # avoid 0 and 1 exactly (log blows up)
loss_if_y1 = -np.log(p)                 # penalty when the truth is class 1

plt.plot(p, loss_if_y1)
plt.title("Log-loss when the true label is y = 1")
plt.xlabel("predicted probability p")
plt.ylabel("loss = -log(p)")
plt.grid(True)
plt.show()
# As p approaches the correct value 1, the loss approaches 0.
# As p approaches the wrong value 0, the loss explodes.
"""))

nb.append(md(r"""
## 6. The gradient (the clean result)

To run gradient descent we need $\partial J/\partial \theta$. The remarkable
fact — and the reason the sigmoid + log-loss pairing is so elegant — is that
the messy derivatives collapse into the **same simple form** as linear
regression:

$$\boxed{\;\frac{\partial J}{\partial \theta} = \frac{1}{n}\,X^{\top}\big(\sigma(X\theta) - y\big)\;}$$

In words: take the prediction error $(\,p - y\,)$ for every example, and
multiply by $X^{\top}$. That's it. We won't grind through the algebra here, but
it follows from $\sigma'(z) = \sigma(z)(1-\sigma(z))$ and the chain rule.

The gradient-descent update (learning rate $\alpha$) is then

$$\theta \leftarrow \theta - \alpha\,\frac{\partial J}{\partial \theta}.$$
"""))

nb.append(md(r"""
## 7. A 2D, two-class dataset

Let's make data we can actually see: two Gaussian "blobs", one per class.
"""))
nb.append(code("""
rng = np.random.default_rng(0)     # fixed seed -> reproducible data

n_per = 100                         # points per class

# Class 0: centered at (-2, -2).  Class 1: centered at (2, 2).
X0 = rng.normal(loc=[-2.0, -2.0], scale=1.2, size=(n_per, 2))
X1 = rng.normal(loc=[ 2.0,  2.0], scale=1.2, size=(n_per, 2))

X_raw = np.vstack([X0, X1])                       # shape (200, 2)
y = np.concatenate([np.zeros(n_per), np.ones(n_per)])   # labels 0 then 1

print("X_raw shape:", X_raw.shape)
print("y shape    :", y.shape, " (first few:", y[:3], "... last few:", y[-3:], ")")

plt.scatter(X0[:, 0], X0[:, 1], label="class 0", marker="o")
plt.scatter(X1[:, 0], X1[:, 1], label="class 1", marker="^")
plt.title("Two Gaussian blobs (the data)")
plt.xlabel("feature x1"); plt.ylabel("feature x2")
plt.legend(); plt.grid(True); plt.show()
"""))

nb.append(md(r"""
Now add the **bias column** of 1's so the bias rides along inside $\theta$.
"""))
nb.append(code("""
n = X_raw.shape[0]                       # number of examples = 200
ones = np.ones((n, 1))                   # a column of 1's
X = np.hstack([ones, X_raw])             # shape (200, 3): [1, x1, x2]
print("X shape:", X.shape, " (first row:", X[0], ")")
# theta will be (b, w1, w2): the first entry multiplies the 1's column = bias.
"""))

nb.append(md(r"""
## 8. Cost and gradient, in code

We use a **numerically-stable** log-loss: we `clip` the probabilities away from
exactly 0 and 1 so `np.log` never sees a 0 (which would give `-inf`).
"""))
nb.append(code("""
def predict_proba(X, theta):
    # p = sigma(X theta);  returns one probability per row of X
    return sigmoid(X @ theta)

def log_loss(X, y, theta, eps=1e-12):
    p = predict_proba(X, theta)
    p = np.clip(p, eps, 1 - eps)         # keep p inside (0,1) for safe log
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(X, y, theta):
    n = X.shape[0]
    p = predict_proba(X, theta)
    # the clean result:  (1/n) X^T (p - y)
    return (X.T @ (p - y)) / n

# sanity check at theta = 0: every p = sigma(0) = 0.5, so loss = -log(0.5).
theta0 = np.zeros(X.shape[1])
print("loss at theta=0:", round(log_loss(X, y, theta0), 4),
      " (should be -log(0.5) =", round(float(-np.log(0.5)), 4), ")")
"""))

nb.append(md(r"""
## 9. Training from scratch with gradient descent

Start at $\theta = 0$, repeatedly step downhill, and record the loss so we can
watch it fall.
"""))
nb.append(code("""
theta = np.zeros(X.shape[1])     # start at the origin: b=w1=w2=0
alpha = 0.1                      # learning rate (step size)
n_iters = 2000                   # number of gradient-descent steps

history = []                     # we'll store the loss at each step
for t in range(n_iters):
    g = gradient(X, y, theta)
    theta = theta - alpha * g    # the update rule
    history.append(log_loss(X, y, theta))

print("final theta (b, w1, w2):", np.round(theta, 3))
print("final loss             :", round(history[-1], 4))
"""))

nb.append(md(r"""
### The loss curve

A healthy training run shows the loss dropping fast, then leveling off.
"""))
nb.append(code("""
plt.plot(history)
plt.title("Log-loss during gradient descent")
plt.xlabel("iteration")
plt.ylabel("log-loss J(theta)")
plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
## 10. The learned decision boundary

The boundary is where $z = b + w_1 x_1 + w_2 x_2 = 0$. Solving for $x_2$:

$$x_2 = -\frac{b + w_1 x_1}{w_2}.$$

We draw that line on top of the data.
"""))
nb.append(code("""
b, w1, w2 = theta                            # unpack the learned parameters

# pick two x1 values spanning the plot, compute the boundary x2 for each
x1_line = np.array([X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1])
x2_line = -(b + w1 * x1_line) / w2

plt.scatter(X0[:, 0], X0[:, 1], label="class 0", marker="o")
plt.scatter(X1[:, 0], X1[:, 1], label="class 1", marker="^")
plt.plot(x1_line, x2_line, "k-", linewidth=2, label="decision boundary")
plt.title("Learned decision boundary")
plt.xlabel("feature x1"); plt.ylabel("feature x2")
plt.legend(); plt.grid(True); plt.show()
"""))

nb.append(md(r"""
## 11. Accuracy from scratch

Accuracy is just the fraction of examples we label correctly. We threshold the
probabilities at $0.5$ and compare to the truth.
"""))
nb.append(code("""
p = predict_proba(X, theta)
y_pred = (p >= 0.5).astype(int)        # True/False -> 1/0
accuracy = np.mean(y_pred == y)        # fraction of matches
print("accuracy:", round(float(accuracy), 4),
      "  (", int(np.sum(y_pred == y)), "of", len(y), "correct )")
"""))

nb.append(md(r"""
## 12. Beyond two classes: softmax (a peek)

What if there are $K > 2$ classes (digits 0–9, say)? We generalize the sigmoid
to the **softmax** function. Each class $k$ gets its own weight vector, giving a
score $z_k = w_k\cdot x + b_k$, and softmax turns the whole score vector into a
probability distribution that sums to 1:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$

The accompanying cost is the multi-class **cross-entropy**, and training again
uses gradient descent — the very same recipe, just with a matrix of weights
instead of a single vector. (With $K=2$, softmax reduces back to the sigmoid.)
We won't implement it here, but the ideas carry over directly.
"""))

# ---- Exercise 1 (with solution) ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Verify the derivative identity $\sigma'(z) = \sigma(z)\,(1-\sigma(z))$
**numerically** at $z = 0.7$. Compare the analytic value with the difference
quotient $\dfrac{\sigma(z+h) - \sigma(z-h)}{2h}$ for a small $h$.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
z0 = 0.7
h = 1e-5

analytic = sigmoid(z0) * (1 - sigmoid(z0))
numeric = (sigmoid(z0 + h) - sigmoid(z0 - h)) / (2 * h)

print("analytic sigma'(0.7):", round(float(analytic), 8))
print("numeric  estimate   :", round(float(numeric), 8))
print("difference          :", abs(analytic - numeric))
"""))

# ---- Exercise 2 (with solution) ----
nb.append(md(r"""
## ✍️ Exercise 2 (solution included)

Confirm that our hand-coded `gradient(X, y, theta)` agrees with a **numerical**
gradient of `log_loss` at a random $\theta$. (This is the standard "gradient
check" used to catch bugs.) Use the central difference for each component.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
rng_check = np.random.default_rng(1)
theta_test = rng_check.normal(size=X.shape[1])    # a random parameter vector
h = 1e-6

analytic_grad = gradient(X, y, theta_test)

numeric_grad = np.zeros_like(theta_test)
for j in range(len(theta_test)):
    step = np.zeros_like(theta_test)
    step[j] = h
    numeric_grad[j] = (log_loss(X, y, theta_test + step)
                       - log_loss(X, y, theta_test - step)) / (2 * h)

print("analytic gradient:", np.round(analytic_grad, 6))
print("numeric  gradient:", np.round(numeric_grad, 6))
print("max difference   :", np.max(np.abs(analytic_grad - numeric_grad)))
"""))

# ---- Exercise 3 (with solution) ----
nb.append(md(r"""
## ✍️ Exercise 3 (solution included)

The learning rate $\alpha$ matters. Re-run training for
$\alpha \in \{0.001, 0.01, 0.1, 1.0\}$ (each for 500 steps from $\theta = 0$)
and plot all four loss curves on one figure. Which converges fastest?
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
for alpha_try in [0.001, 0.01, 0.1, 1.0]:
    th = np.zeros(X.shape[1])
    hist = []
    for t in range(500):
        th = th - alpha_try * gradient(X, y, th)
        hist.append(log_loss(X, y, th))
    plt.plot(hist, label=f"alpha = {alpha_try}")

plt.title("Effect of the learning rate on convergence")
plt.xlabel("iteration"); plt.ylabel("log-loss")
plt.legend(); plt.grid(True); plt.show()
# Larger alpha converges faster here; too-large can overshoot or diverge.
"""))

# ---- Homework (no solutions) ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Harder data.** Move the two blob centers closer together (e.g. $(-1,-1)$
   and $(1,1)$) so the classes overlap. Retrain and report the new accuracy.
   Why is it lower?
2. **Test set.** Split the 200 points into a training half and a testing half.
   Train only on the training half, then measure accuracy on the *unseen*
   testing half. Does it differ from the training accuracy?
3. **Probability map.** Using `np.meshgrid`, evaluate the learned $p$ on a fine
   grid over the plane and draw it with `plt.contourf`. Overlay the data. You
   should see the probability rise smoothly from 0 to 1 across the boundary.
4. **A third feature that's useless.** Add a column of pure noise as a third
   feature, retrain, and inspect the learned weight on it. Is it close to 0,
   as you'd hope?
"""))

save(os.path.join(CH, "11_logistic_regression.ipynb"), nb)
