"""Generator for Chapter 09 — Introduction to Machine Learning.

Run from anywhere:  python tools/generators/ch09_intro_ml.py
Produces one notebook in 09-introduction-to-machine-learning/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "09-introduction-to-machine-learning")


# ---------------------------------------------------------------------------
# Notebook 09 — Introduction to Machine Learning
# ---------------------------------------------------------------------------
nb = []

nb.append(md(r"""
# Chapter 09 — Introduction to Machine Learning

> **The big idea.** A machine-learning model is just a **function with knobs**.
> We *learn* the knobs (the **parameters**) from data by making the function's
> mistakes as small as possible. Everything you have built so far — vectors,
> matrices, derivatives, gradients — exists to make this one idea work.

This chapter is mostly **conceptual**, with small concrete demos. By the end you
will be able to say precisely what these words mean:

- a **model** is a function $f_{\theta}$ with parameters $\theta$;
- **supervised** vs **unsupervised** learning;
- the **data layout**: a feature matrix $X$ and a label vector $y$;
- a **loss** (cost) function that scores how wrong the model is;
- the **train/test split** and why we must test on *held-out* data;
- **generalization**, and its enemies **overfitting** and **underfitting**;
- the **bias–variance** trade-off.

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
## 1. What *is* machine learning?

Classical programming: a human writes the rules.

> *"If the email contains the word LOTTERY, mark it as spam."*

Machine learning flips this around. Instead of writing the rules, we **show the
computer examples** (emails already labelled *spam* / *not spam*) and let it
*find* a rule that fits them. In one sentence:

> **Machine learning = learning a function from data.**

We assume there is some true but unknown function $g$ that maps an input to the
right output, $y = g(x)$. We never see $g$. We only see a finite sample of
input–output pairs. Our job is to build a function $f$ that **approximates** $g$
well — not just on the examples we have seen, but on *new* inputs.
"""))

nb.append(md(r"""
### Supervised vs unsupervised (the two big families)

| | You are given | Goal | Example |
|---|---|---|---|
| **Supervised** | inputs **and** correct outputs $(x, y)$ | predict $y$ for new $x$ | predict house price from size |
| **Unsupervised** | inputs **only** $(x)$ | find structure | group customers into segments |

Supervised learning splits further by what $y$ looks like:

- **Regression** — $y$ is a real number (price, temperature, age).
- **Classification** — $y$ is a category (spam / not-spam, digit 0–9).

This chapter focuses on the supervised picture, because it makes every concept
(model, loss, train/test) concrete. We mention unsupervised learning only
briefly here; clustering and dimensionality reduction get their own chapters.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 2. The data layout: feature matrix $X$ and label vector $y$

Almost all of ML stores data as a **table of numbers**:

- one **row** per example (a *sample*),
- one **column** per measured quantity (a *feature*).

That table is the **feature matrix** $X$ with shape $(n, d)$: $n$ samples,
$d$ features. The matching answers form the **label vector** $y$ of length $n$.

$$
X = \begin{bmatrix} x_{1,1} & \cdots & x_{1,d}\\ \vdots & & \vdots \\ x_{n,1} & \cdots & x_{n,d}\end{bmatrix}
\in \mathbb{R}^{n\times d},
\qquad
y = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^{n}.
$$

Row $i$ of $X$ is the feature vector of example $i$; $y_i$ is its label. Let's
build a tiny toy dataset: predict a person's (made-up) monthly spending from two
features, **age** and **income**.
"""))
nb.append(code("""
# 5 samples, 2 features  ->  X has shape (5, 2)
X = np.array([
    [25, 30.0],   # age 25, income 30k
    [32, 55.0],
    [47, 80.0],
    [51, 60.0],
    [62, 90.0],
])
y = np.array([12.0, 20.0, 30.0, 24.0, 33.0])   # monthly spending (one per row)

print("X shape:", X.shape)    # (n_samples, n_features) = (5, 2)
print("y shape:", y.shape)    # (n_samples,)            = (5,)
print("row 0 (first person):", X[0], "-> label", y[0])
print("column 0 (all ages) :", X[:, 0])
"""))

nb.append(md(r"""
**Convention to memorize:** rows are samples, columns are features. When a
library complains about shapes, 90% of the time it is because $X$ is transposed.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 3. A model is a function with parameters

A **model** is a family of candidate functions, indexed by **parameters**
$\theta$. Learning = choosing the $\theta$ that fits the data best.

The simplest useful model is **linear regression**. With a single feature $x$:

$$f_{\theta}(x) = w\,x + b, \qquad \theta = (w, b).$$

Here $w$ (the *weight* / slope) and $b$ (the *bias* / intercept) are the knobs.
Different $\theta$ give different lines. The same idea with $d$ features is just a
dot product, $f_{\theta}(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$.

Let's *make up* a line and see its predictions. (We are choosing $\theta$ by
hand for now; soon we'll let the data choose it.)
"""))
nb.append(code("""
def predict(x, w, b):
    return w * x + b          # the model f_theta(x), with theta = (w, b)

# one feature: spending vs age, using just column 0 of X
ages = X[:, 0]

w, b = 0.4, 3.0               # a guess for the parameters
y_pred = predict(ages, w, b)  # NumPy applies the formula to every age at once

print("ages      :", ages)
print("predicted :", y_pred)
print("actual    :", y)
"""))

nb.append(md(r"""
Some predictions are close, some are off. We need a single number that says
*how good* a choice of $\theta$ is. That number is the **loss**.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 4. Loss functions: scoring the mistakes

A **loss** (or **cost**) function takes the predictions and the true labels and
returns one number — **smaller is better**. Training a model literally means
*minimizing the loss* over $\theta$.

### Regression: Mean Squared Error (MSE)

The workhorse for regression is the **mean squared error**:

$$\mathrm{MSE}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\bigl(f_{\theta}(x_i) - y_i\bigr)^2.$$

We square the residuals so that positive and negative errors don't cancel, and
so that *big* mistakes are punished much more than small ones.
"""))
nb.append(code("""
def mse(y_true, y_hat):
    residuals = y_hat - y_true        # how far each prediction is off
    return np.mean(residuals**2)      # square, then average

print("MSE of our guess:", mse(y, y_pred))
"""))

nb.append(md(r"""
A lower MSE means a better fit. We can compare two parameter choices just by
comparing their losses:
"""))
nb.append(code("""
guess_A = predict(ages, w=0.4, b=3.0)
guess_B = predict(ages, w=0.5, b=0.0)

print("loss A:", mse(y, guess_A))
print("loss B:", mse(y, guess_B))
print("B is better" if mse(y, guess_B) < mse(y, guess_A) else "A is better")
"""))

nb.append(md(r"""
### Classification: accuracy and the 0–1 loss

For classification the labels are categories, so squaring an error makes no
sense. The simplest score is **accuracy**: the fraction of examples we get
right. Equivalently, the **0–1 loss** counts the fraction we get *wrong*:

$$\text{accuracy} = \frac{1}{n}\sum_{i=1}^n \mathbf{1}\{\hat{y}_i = y_i\},
\qquad \text{0–1 loss} = 1 - \text{accuracy}.$$
"""))
nb.append(code("""
def accuracy(y_true, y_hat):
    return np.mean(y_hat == y_true)   # mean of True/False -> fraction correct

# toy: true labels (0/1) and a classifier's predictions
y_true = np.array([0, 1, 1, 0, 1, 1])
y_hat  = np.array([0, 1, 0, 0, 1, 1])   # one mistake (3rd example)

print("accuracy :", accuracy(y_true, y_hat))   # 5/6 correct
print("0-1 loss :", 1 - accuracy(y_true, y_hat))
"""))

# ---- Exercise 1 ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

The MSE punishes big errors a lot. The **mean absolute error**
$\mathrm{MAE} = \frac{1}{n}\sum_i |f_\theta(x_i) - y_i|$ treats all errors more
evenly. Write a function `mae(y_true, y_hat)` and compare the MAE and MSE of our
earlier guess (`y_pred`). Which one is the bigger number here, and why?
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
def mae(y_true, y_hat):
    return np.mean(np.abs(y_hat - y_true))    # average absolute residual

print("MAE:", mae(y, y_pred))
print("MSE:", mse(y, y_pred))
# MSE is larger here: squaring residuals > 1 inflates them, while |.| does not.
# MSE therefore reacts much more strongly to any single large mistake (outlier).
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 5. The train/test split (from scratch) — and *why*

Here is the central honesty problem of machine learning:

> A model can score perfectly on the examples it was *trained* on simply by
> **memorizing** them. That tells us nothing about new data.

We care about **generalization**: performance on inputs the model has never
seen. To measure it honestly we hide some data from training and use it only to
*test*:

- **training set** — used to choose $\theta$ (to *fit* the model);
- **test set** — locked away, used *once* at the end to estimate real-world error.

Let's write the split **from scratch**, the way every library does internally:
shuffle the row indices with our `rng`, then slice them into two groups.
"""))
nb.append(code("""
def train_test_split(X, y, test_frac=0.25, rng=rng):
    n = X.shape[0]                       # number of samples
    idx = np.arange(n)                   # [0, 1, 2, ..., n-1]
    rng.shuffle(idx)                     # shuffle the order IN PLACE

    n_test = int(round(n * test_frac))   # how many go to the test set
    test_idx  = idx[:n_test]             # first chunk  -> test
    train_idx = idx[n_test:]             # the rest     -> train

    return (X[train_idx], X[test_idx],   # X_train, X_test
            y[train_idx], y[test_idx])   # y_train, y_test
"""))

nb.append(md(r"""
Why **shuffle** first? Real datasets are often sorted (all the "yes" rows, then
all the "no" rows, or sorted by date). Slicing without shuffling could put an
entire category into the test set and none into training. Shuffling makes the two
sets statistically similar. Let's try it on a slightly bigger toy dataset.
"""))
nb.append(code("""
# make 20 samples, 1 feature, with a linear trend plus noise
n = 20
x_all = np.linspace(0, 10, n)                         # feature values
y_all = 2.0 * x_all + 1.0 + rng.normal(0, 1.5, n)     # y = 2x + 1 + noise
X_all = x_all.reshape(-1, 1)                          # shape (20, 1): a column

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_frac=0.25)

print("total samples:", n)
print("train size   :", X_train.shape[0])   # 15
print("test  size   :", X_test.shape[0])    # 5
"""))

# ---- Exercise 2 ----
nb.append(md(r"""
## ✍️ Exercise 2 (solution included)

After a split, the train and test sets should **not overlap** and **together**
cover every original sample. Using the 20-sample dataset above, verify this by
checking that the train and test labels, combined, contain exactly the same
values as `y_all` (just in a different order).
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
combined = np.concatenate([y_train, y_test])   # glue the two pieces together

# sorting both lets us compare them ignoring order
same = np.allclose(np.sort(combined), np.sort(y_all))
print("no data lost or duplicated:", same)
print("sizes add up:", y_train.size + y_test.size == y_all.size)
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 6. Fitting a model and reading two errors

`np.polyfit(x, y, degree)` finds the polynomial of the requested degree that
**minimizes the MSE** on the data — it does the optimization for us.
`np.polyval(coeffs, x)` then evaluates that polynomial. A *line* is just a
degree-1 polynomial, so this is exactly our linear-regression model.

We always report **two** numbers: the **training error** (loss on data we fit)
and the **test error** (loss on the held-out data). The gap between them is the
whole story of generalization.
"""))
nb.append(code("""
xtr = X_train.ravel()   # polyfit wants 1-D arrays
xte = X_test.ravel()

coeffs = np.polyfit(xtr, y_train, deg=1)     # fit a line on the TRAIN set only
print("learned slope, intercept:", coeffs)   # ~ [2, 1], the true values

train_err = mse(y_train, np.polyval(coeffs, xtr))
test_err  = mse(y_test,  np.polyval(coeffs, xte))
print("train MSE:", train_err)
print("test  MSE:", test_err)
"""))
nb.append(code("""
# picture: the data and the fitted line
grid = np.linspace(0, 10, 100)
plt.scatter(xtr, y_train, label="train", color="C0")
plt.scatter(xte, y_test, label="test", color="C1", marker="s")
plt.plot(grid, np.polyval(coeffs, grid), color="k", label="fitted line")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Linear fit (degree 1)")
plt.legend(); plt.grid(True)
plt.show()
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 7. Overfitting vs underfitting

A degree-1 line fit the linear data well. What if we use a *more flexible* model
than the data needs? Let's create data from a gentle curve plus noise, then fit
polynomials of increasing degree.

- **Underfitting** (too simple): the model can't capture the real pattern. High
  error on *both* train and test.
- **Overfitting** (too flexible): the model bends to chase the random noise. Tiny
  train error, but **large test error** — it memorized instead of learning.
"""))
nb.append(code("""
# true pattern: a smooth curve;  data = pattern + noise
def true_fn(x):
    return np.sin(1.2 * x) + 0.5 * x

n = 30
x = np.sort(rng.uniform(0, 5, n))           # random x's in [0, 5]
y = true_fn(x) + rng.normal(0, 0.35, n)     # add noise

# split into train / test
Xc = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_frac=0.30)
xtr, xte = X_train.ravel(), X_test.ravel()
print("train:", xtr.size, " test:", xte.size)
"""))
nb.append(code("""
# fit three models: too simple, about right, way too flexible
grid = np.linspace(0, 5, 300)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, deg in zip(axes, [1, 3, 15]):
    c = np.polyfit(xtr, y_train, deg)                 # fit on train only
    ax.scatter(xtr, y_train, s=20, label="train", color="C0")
    ax.scatter(xte, y_test, s=25, label="test", color="C1", marker="s")
    ax.plot(grid, np.polyval(c, grid), "k", label=f"deg {deg}")
    ax.plot(grid, true_fn(grid), "g--", alpha=0.6, label="truth")
    ax.set_ylim(-2, 5)
    ax.set_title(f"degree {deg}")
    ax.legend(fontsize=8); ax.grid(True)

axes[0].set_ylabel("y")
plt.suptitle("Left: underfit   Middle: good fit   Right: overfit")
plt.tight_layout()
plt.show()
"""))

nb.append(md(r"""
Look at the right panel: the degree-15 curve wiggles wildly to pass near every
*training* point, but it strays far from the green truth and from the test
points. That is overfitting in one picture.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 8. The classic U-shaped test-error curve

Let's make the trade-off quantitative. We fit every degree from 1 to 15 and
record the **train** and **test** MSE. The pattern is one of the most important
in all of ML:

- **Train error** keeps **falling** as the model grows more flexible (it can
  always fit the training points better).
- **Test error** first **falls** (the model is learning the real pattern) then
  **rises** (the model starts memorizing noise). It traces a **U**.

The bottom of the U is the **sweet spot** — the model complexity that
generalizes best.
"""))
nb.append(code("""
degrees = range(1, 16)
train_errs, test_errs = [], []

for deg in degrees:
    c = np.polyfit(xtr, y_train, deg)               # fit on train
    train_errs.append(mse(y_train, np.polyval(c, xtr)))
    test_errs.append(mse(y_test,  np.polyval(c, xte)))

train_errs = np.array(train_errs)
test_errs  = np.array(test_errs)

best_deg = list(degrees)[int(np.argmin(test_errs))]   # degree with lowest test error
print("best degree (lowest test error):", best_deg)
"""))
nb.append(code("""
plt.plot(list(degrees), train_errs, "o-", label="train error")
plt.plot(list(degrees), test_errs,  "s-", label="test error")
plt.axvline(best_deg, color="gray", ls="--", label=f"best deg = {best_deg}")
plt.yscale("log")                       # errors span orders of magnitude
plt.xlabel("polynomial degree (model complexity)")
plt.ylabel("MSE  (log scale)")
plt.title("Train vs test error: the U-shaped curve")
plt.legend(); plt.grid(True, which="both")
plt.show()
"""))

nb.append(md(r"""
Notice the signature of overfitting on the right of the plot: train error near
zero while test error climbs. A huge gap between train and test error is the
classic alarm bell.
"""))

# ---- Exercise 3 ----
nb.append(md(r"""
## ✍️ Exercise 3 (solution included)

The **generalization gap** is `test_error - train_error`. Print the gap for
degree 1 (underfit), the best degree, and degree 15 (overfit). Confirm in words
that the gap is small for the underfit/good models and large for the overfit
one.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
deg_list = list(degrees)
for deg in [1, best_deg, 15]:
    i = deg_list.index(deg)
    gap = test_errs[i] - train_errs[i]
    print(f"degree {deg:2d}:  train={train_errs[i]:.4f}  "
          f"test={test_errs[i]:.4f}  gap={gap:.4f}")

# The degree-15 model has a tiny train error but a much larger test error,
# so its gap is by far the biggest -> it memorized noise (overfitting).
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 9. Bias–variance, in words

The U-curve has a beautiful explanation. The expected test error of a model can
be split into three parts:

$$\text{test error} \;=\; \underbrace{\text{bias}^2}_{\text{too rigid}}
\;+\; \underbrace{\text{variance}}_{\text{too jumpy}}
\;+\; \underbrace{\text{irreducible noise}}_{\text{unavoidable}}.$$

- **Bias** — error from the model being **too simple** to represent the truth.
  A straight line fit to a curve has high bias. *(left of the U: underfitting.)*
- **Variance** — how much the fitted model would **change if we resampled** the
  training data. A degree-15 polynomial swings wildly with each new noisy sample,
  so it has high variance. *(right of the U: overfitting.)*
- **Irreducible noise** — randomness in the data itself. No model can beat it.

> Simple models: **high bias, low variance**.
> Complex models: **low bias, high variance**.
> The best model **balances the two** — the bottom of the U.

This is the **bias–variance trade-off**, and it is *why* we always evaluate on
held-out data instead of trusting the training score.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## 10. A tiny classifier: the threshold rule

To see classification end-to-end, here is the simplest possible classifier. We
have one feature and two classes; we pick a **threshold** $t$ and predict class 1
when the feature exceeds it. "Learning" means choosing the $t$ with the best
training accuracy.
"""))
nb.append(code("""
# one feature; class 1 tends to have larger feature values
feat = np.concatenate([rng.normal(2, 1, 40), rng.normal(5, 1, 40)])
label = np.concatenate([np.zeros(40), np.ones(40)]).astype(int)

# try many thresholds and keep the one with the best ACCURACY
candidates = np.linspace(feat.min(), feat.max(), 200)
accs = [accuracy(label, (feat > t).astype(int)) for t in candidates]

best_t = candidates[int(np.argmax(accs))]
print("best threshold:", round(best_t, 3))
print("best accuracy :", round(max(accs), 3))
"""))
nb.append(code("""
plt.hist(feat[label == 0], bins=15, alpha=0.6, label="class 0")
plt.hist(feat[label == 1], bins=15, alpha=0.6, label="class 1")
plt.axvline(best_t, color="k", ls="--", label=f"threshold = {best_t:.2f}")
plt.xlabel("feature value"); plt.ylabel("count")
plt.title("A threshold classifier")
plt.legend(); plt.grid(True)
plt.show()
"""))

nb.append(md(r"""
That's the entire machine-learning loop in miniature: a model (the rule
`feat > t`), a parameter ($t$), a loss (1 − accuracy), and learning by picking
the parameter that minimizes the loss. Bigger models like neural networks add
*millions* of parameters and use gradient descent (Chapter 07) instead of a
brute-force search — but the skeleton is identical.
"""))

# ---------------------------------------------------------------------------
nb.append(md(r"""
## Recap

- ML = **learning a function from data**; we approximate an unknown $g$ by an
  $f_\theta$ chosen from examples.
- **Supervised** learning uses labelled pairs $(x, y)$; **regression** predicts
  numbers, **classification** predicts categories. **Unsupervised** learning has
  no labels.
- Data lives in a feature matrix $X$ of shape $(n, d)$ (rows = samples) and a
  label vector $y$.
- A **model** is a function $f_\theta$ with **parameters** $\theta$; a **loss**
  scores its mistakes (**MSE** for regression, **accuracy / 0–1** for
  classification).
- We **split** data into train and test (shuffle, then slice) and judge a model
  by its **test** error — this measures **generalization**.
- **Underfitting** = too simple (high bias); **overfitting** = too flexible
  (high variance). Plotting train vs test error gives the **U-shaped** curve, and
  the bottom is the sweet spot.
"""))

# ---- Homework ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Root mean squared error.** Write `rmse(y_true, y_hat)` returning
   $\sqrt{\mathrm{MSE}}$. Why is RMSE often preferred for *reporting* a model's
   error to a human, even though MSE is what we minimize? *(Hint: think about
   units.)*

2. **Repeated splits.** The test error depends on *which* points landed in the
   test set. Re-run `train_test_split` 50 times (with fresh shuffles), fit a
   degree-3 polynomial each time on the Section 7 data, and collect the 50 test
   MSEs. Print their mean and standard deviation. What does the spread tell you?

3. **Your own U-curve.** Generate a fresh dataset from a different true function
   (e.g. $g(x) = x^2 - 3x$) plus noise, then reproduce the train-vs-test-error
   plot of Section 8. Which degree wins, and does it match the shape of $g$?

4. **A 2-D threshold-ish classifier.** Build a toy dataset with **two** features
   where class 1 is centred at $(4, 4)$ and class 0 at $(1, 1)$. Classify a point
   as class 1 if $x_1 + x_2 > t$, sweep $t$ to maximize accuracy, and report the
   best accuracy. (You have just built a *linear* classifier.)
"""))

save(os.path.join(CH, "09_intro_ml.ipynb"), nb)
