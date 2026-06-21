"""Generator for Chapter 10 — Linear Regression (first real ML model).

Run from anywhere:  python tools/generators/ch10_linear_regression.py
Produces one notebook in 10-linear-regression/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "10-linear-regression")


# ---------------------------------------------------------------------------
# Notebook 10 — Linear Regression
# ---------------------------------------------------------------------------
c = []

c.append(md(r"""
# Chapter 10 — Linear Regression (your first ML model, by hand)

This is the moment all the previous chapters were building toward. We take real
ideas — **vectors**, **matrices**, **derivatives**, **minimization** — and snap
them together into the very first machine-learning model: **linear regression**.

The promise of supervised learning is simple to state:

> Given examples of inputs $x$ and matching outputs $y$, find a rule that
> predicts $y$ from $x$ — and that *generalizes* to inputs we have not seen.

Linear regression chooses the simplest possible rule: a **straight line**. Yet
the recipe we use — write down an error, then minimize it — is *exactly* the
recipe behind logistic regression, neural networks, and deep learning. Master
it here and the rest of the course is variations on this theme.

We build everything from scratch with **pure NumPy**, then check our answer
against scikit-learn at the very end.
"""))

c.append(md(r"""
## 1. The model: a line with a slope and an offset

For a single input feature $x$, linear regression predicts

$$\hat{y} = w\,x + b,$$

where

- $w$ is the **weight** (slope) — how much $\hat{y}$ changes per unit of $x$,
- $b$ is the **bias** (intercept) — the value of $\hat{y}$ when $x = 0$.

The hat on $\hat{y}$ means "prediction", to distinguish it from the true,
observed $y$. *Learning* means choosing the numbers $w$ and $b$ so the line
passes as close as possible to the data.
"""))
c.append(code("""
import numpy as np
import matplotlib.pyplot as plt

# A reproducible random-number generator (same numbers every run).
rng = np.random.default_rng(0)

# The model is literally one line of code:
def predict(x, w, b):
    return w * x + b          # y_hat = w*x + b

print(predict(2.0, w=3.0, b=1.0))   # 3*2 + 1 = 7.0
"""))

c.append(md(r"""
## 2. Synthetic data: a known truth plus noise

To test a learning algorithm it helps to *invent* data where we already know
the right answer. We pick a "true" line $y = w_\star x + b_\star$, sample some
$x$ values, and add random Gaussian **noise** to mimic messy real measurements.

If our algorithm is any good, it should recover $w_\star$ and $b_\star$ from the
noisy points — without ever being told them.
"""))
c.append(code("""
# The hidden truth we will try to recover.
w_true, b_true = 2.5, -1.0

n = 40                                  # number of data points
x = rng.uniform(0, 4, size=n)           # inputs spread over [0, 4]
noise = rng.normal(0, 1.0, size=n)      # measurement noise, std = 1.0
y = w_true * x + b_true + noise         # noisy observations

# A quick look at the data.
plt.scatter(x, y, color="steelblue", label="noisy data")
plt.plot(x, w_true * x + b_true, "k--", label="true line")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Synthetic noisy linear data")
plt.legend(); plt.grid(True)
plt.show()
"""))

c.append(md(r"""
## 3. Measuring how wrong we are: the cost function

For any guess $(w, b)$, each point has a **residual** — the gap between what we
predicted and what actually happened:

$$r_i = \hat{y}_i - y_i = (w x_i + b) - y_i.$$

We want all residuals small. Summing them directly is no good (positive and
negative gaps cancel), so we **square** them and average. This is the
**Mean Squared Error** (MSE), our *cost* (or *loss*) function:

$$J(w, b) = \frac{1}{n}\sum_{i=1}^{n}\bigl(\hat{y}_i - y_i\bigr)^2.$$

Squaring does two helpful things: it makes every term positive, and it punishes
large mistakes far more than small ones. Lower $J$ means a better fit; learning
is the search for the $(w, b)$ that makes $J$ as small as possible.
"""))
c.append(code("""
def mse(x, y, w, b):
    y_hat = w * x + b           # predictions for every point at once (NumPy!)
    residuals = y_hat - y       # the gaps
    return np.mean(residuals ** 2)   # average of the squared gaps

# The true line is good but not perfect — noise gives it nonzero cost.
print("cost at the TRUE line :", mse(x, y, w_true, b_true))
# A deliberately bad guess costs much more.
print("cost at a bad guess   :", mse(x, y, 0.0, 0.0))
"""))

c.append(md(r"""
## 4. Where does the minimum live? The least-squares idea

$J(w, b)$ is a smooth, bowl-shaped function of two variables. From Chapter 06 we
know the minimum of a smooth function sits where its slope is zero. Here we have
two knobs, so we set **both** partial derivatives to zero:

$$\frac{\partial J}{\partial w} = 0, \qquad \frac{\partial J}{\partial b} = 0.$$

Working these out (a short calculus exercise) gives two linear equations in the
two unknowns $w$ and $b$ — the **normal equations**. Because they are *linear*,
we can solve them exactly, in one shot, with no iteration. That exact answer is
the famous **least-squares solution**.

Rather than carry around two special-case formulas, we'll package everything
into clean matrix form, which then works for *any* number of features.
"""))

c.append(md(r"""
## 5. Matrix form and the normal equation

Collect the parameters into one vector $\theta = \begin{bmatrix} b \\ w
\end{bmatrix}$ and stack the inputs into a **design matrix** $X$ whose first
column is all ones (so the ones multiply $b$, the intercept):

$$
X = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix},
\qquad
X\theta = \begin{bmatrix} b + w x_1 \\ \vdots \\ b + w x_n \end{bmatrix} = \hat{y}.
$$

Now the cost is just a squared vector length:
$J(\theta) = \tfrac{1}{n}\lVert X\theta - y \rVert^2$. Setting its gradient to
zero gives the **normal equation**

$$X^{\!\top} X\,\theta = X^{\!\top} y \quad\Longrightarrow\quad
\theta = \bigl(X^{\!\top} X\bigr)^{-1} X^{\!\top} y.$$

**Practical note:** we never actually form the inverse $(X^\top X)^{-1}$.
Inverting a matrix is slower and numerically shakier than *solving* the linear
system directly, so we hand $X^\top X$ and $X^\top y$ to `np.linalg.solve`.
"""))
c.append(code("""
def fit_normal_equation(X, y):
    # Solve (X^T X) theta = (X^T y) for theta, WITHOUT forming an inverse.
    XtX = X.T @ X               # @ is matrix multiplication
    Xty = X.T @ y
    theta = np.linalg.solve(XtX, Xty)   # preferred over inv(XtX) @ Xty
    return theta

# Build the design matrix: a column of ones, then the x column.
X = np.column_stack([np.ones(n), x])     # shape (n, 2)
theta = fit_normal_equation(X, y)
b_hat, w_hat = theta                      # first entry is bias, second is weight

print("learned w =", w_hat, "  (true was", w_true, ")")
print("learned b =", b_hat, "  (true was", b_true, ")")
"""))

c.append(md(r"""
The recovered slope and intercept land close to the hidden truth — the gap is
only because of the noise we added. Let's see the fitted line.
"""))
c.append(code("""
xline = np.linspace(x.min(), x.max(), 100)
yline = w_hat * xline + b_hat

plt.scatter(x, y, color="steelblue", label="data")
plt.plot(xline, yline, "r-", lw=2, label="least-squares fit")
plt.plot(xline, w_true * xline + b_true, "k--", label="true line")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Fitted line via the normal equation")
plt.legend(); plt.grid(True)
plt.show()
"""))

# ---- Exercise 1 ----
c.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

The fitted line should have **lower** MSE than the true line on *this* noisy
sample (least squares fits the data we actually have, noise and all). Compute
both costs and confirm the fitted one is smaller.
"""))
c.append(md("**Solution:**"))
c.append(code("""
cost_fit  = mse(x, y, w_hat,  b_hat)
cost_true = mse(x, y, w_true, b_true)
print("MSE of fitted line :", cost_fit)
print("MSE of true line   :", cost_true)
print("fitted is lower?   :", cost_fit < cost_true)   # True
"""))

c.append(md(r"""
## 6. Residuals: looking at what is left over

A residual plot shows $r_i = \hat{y}_i - y_i$ against $x$. For a *good* linear
fit the residuals should look like **structureless noise** scattered around
zero — no trend, no curve. Any leftover pattern is a hint that a straight line
is missing something.
"""))
c.append(code("""
y_hat = w_hat * x + b_hat
residuals = y_hat - y

plt.scatter(x, residuals, color="darkorange")
plt.axhline(0, color="black", lw=1)        # the "perfect" reference line
plt.xlabel("x"); plt.ylabel("residual  (y_hat - y)")
plt.title("Residuals: should be patternless noise around 0")
plt.grid(True)
plt.show()
"""))

c.append(md(r"""
## 7. Seeing the cost surface $J(w, b)$

Because there are exactly two parameters, we can *draw* the entire cost
landscape: a grid of $(w, b)$ values colored by $J$. The least-squares solution
sits at the bottom of the bowl. This picture is the geometric heart of all of
optimization — every training algorithm is a way of walking downhill on a
surface like this.
"""))
c.append(code("""
# A grid of candidate (w, b) values around the solution.
w_grid = np.linspace(w_hat - 3, w_hat + 3, 100)
b_grid = np.linspace(b_hat - 3, b_hat + 3, 100)
WW, BB = np.meshgrid(w_grid, b_grid)

# Evaluate J at every grid point.
J = np.zeros_like(WW)
for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        J[i, j] = mse(x, y, WW[i, j], BB[i, j])

# --- Contour view ---
plt.contourf(WW, BB, J, levels=30, cmap="viridis")
plt.colorbar(label="cost J(w, b)")
plt.plot(w_hat, b_hat, "r*", markersize=16, label="optimum")
plt.xlabel("w (slope)"); plt.ylabel("b (intercept)")
plt.title("Cost surface (contour) with the minimum marked")
plt.legend()
plt.show()
"""))
c.append(code("""
# --- 3D view of the same bowl ---
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(WW, BB, J, cmap="viridis", alpha=0.85)
ax.scatter(w_hat, b_hat, mse(x, y, w_hat, b_hat),
           color="red", s=60, label="optimum")
ax.set_xlabel("w"); ax.set_ylabel("b"); ax.set_zlabel("J(w, b)")
ax.set_title("The cost is a convex bowl")
plt.show()
"""))

c.append(md(r"""
The surface is a **convex bowl**: it has a single lowest point and no other
local minima. That is *why* linear regression is so well behaved — there is one
right answer, and the normal equation jumps straight to it.
"""))

# ---- Exercise 2 ----
c.append(md(r"""
---
## ✍️ Exercise 2 (solution included)

Verify numerically that the normal-equation solution really is the bottom of the
bowl: find the grid point with the smallest $J$ and check that its $(w, b)$ is
close to $(\hat{w}, \hat{b})$.

*Hint:* `np.argmin(J)` gives a flat index; `np.unravel_index` turns it into row,
column.
"""))
c.append(md("**Solution:**"))
c.append(code("""
flat = np.argmin(J)                       # index into the flattened grid
i, j = np.unravel_index(flat, J.shape)    # back to (row, col)
print("grid-min  w =", WW[i, j], " b =", BB[i, j])
print("normal-eq w =", w_hat,     " b =", b_hat)
# They agree up to the resolution of the grid.
"""))

c.append(md(r"""
## 8. More than one feature: multivariate regression

The beauty of the matrix form is that **nothing changes** when there are many
features. With $d$ features per example, $\hat{y} = w_1 x_1 + \dots + w_d x_d +
b$, the design matrix $X$ simply gets one column per feature (plus the column of
ones), and the *same* normal equation solves it.

Let's invent a small dataset with **three** features and a known weight vector,
then recover it.
"""))
c.append(code("""
n2, d = 200, 3
X_feats = rng.normal(0, 1, size=(n2, d))   # 200 examples, 3 features each

w_vec_true = np.array([1.5, -2.0, 0.5])    # hidden weights, one per feature
b_scalar_true = 4.0                         # hidden intercept

y2 = X_feats @ w_vec_true + b_scalar_true + rng.normal(0, 0.5, size=n2)

# Same recipe: prepend a column of ones, then solve the normal equation.
X2 = np.column_stack([np.ones(n2), X_feats])   # shape (n2, d+1)
theta2 = fit_normal_equation(X2, y2)

print("learned b      :", theta2[0],  "  (true 4.0)")
print("learned weights:", theta2[1:], "  (true", w_vec_true, ")")
"""))

c.append(md(r"""
## 9. How good is the fit? The $R^2$ score

MSE is in squared units of $y$, which makes it hard to interpret on its own.
The **coefficient of determination** $R^2$ rescales it to a unitless number,
usually between 0 and 1:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
      = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}.$$

The denominator is the error of the dumbest model — always predicting the mean
$\bar{y}$. So $R^2$ answers *"how much of the variation in $y$ does our line
explain, beyond just guessing the average?"* $R^2 = 1$ is a perfect fit;
$R^2 = 0$ is no better than the mean.
"""))
c.append(code("""
def r2_score(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)          # error of our model
    ss_tot = np.sum((y - np.mean(y)) ** 2)     # error of "always the mean"
    return 1 - ss_res / ss_tot

# R^2 for the single-feature fit from earlier...
print("R^2 (1 feature) :", r2_score(y, w_hat * x + b_hat))
# ...and for the multivariate fit.
y2_hat = X2 @ theta2
print("R^2 (3 features):", r2_score(y2, y2_hat))
"""))

# ---- Exercise 3 ----
c.append(md(r"""
---
## ✍️ Exercise 3 (solution included)

The constant model "always predict $\bar{y}$" should score exactly $R^2 = 0$ by
construction. Confirm this by feeding `r2_score` a prediction array filled with
the mean of `y`.
"""))
c.append(md("**Solution:**"))
c.append(code("""
mean_prediction = np.full_like(y, np.mean(y))   # every entry = mean(y)
print("R^2 of the mean model:", r2_score(y, mean_prediction))   # 0.0
"""))

c.append(md(r"""
## 10. Sanity check against scikit-learn

`scikit-learn` is the standard Python ML library. Its `LinearRegression` solves
the *same* least-squares problem, so its coefficients should match ours almost
exactly. (We wrap the import in `try/except` so this notebook still runs if the
library is not installed.)
"""))
c.append(code("""
try:
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()       # fits an intercept by default
    model.fit(X_feats, y2)           # NOTE: give it the raw features, no ones column

    print("sklearn intercept:", model.intercept_)
    print("ours             :", theta2[0])
    print("sklearn coefs    :", model.coef_)
    print("ours             :", theta2[1:])
    print("max difference   :",
          np.max(np.abs(model.coef_ - theta2[1:])))   # ~ 1e-13
except ImportError:
    print("scikit-learn not installed - skipping the comparison.")
    print("Our from-scratch weights were:", theta2)
"""))

c.append(md(r"""
The numbers agree to roughly machine precision: our hand-built normal equation
*is* what the library does under the hood.

## 11. What about gradient descent?

The normal equation is exact and perfect for small problems. But forming
$X^\top X$ becomes expensive when there are millions of features, and for
*non-linear* models (logistic regression, neural networks) there is no
closed-form solution at all. The general-purpose alternative is to start
somewhere and repeatedly step **downhill** on the cost surface:

$$\theta \leftarrow \theta - \eta\,\nabla J(\theta),$$

with learning rate $\eta$. This is **gradient descent** — the workhorse of all
of deep learning — and we devote **Chapter 12** to building it from scratch and
watching it roll down the very bowl we plotted above.
"""))

c.append(md(r"""
## Summary

- The model is a line: $\hat{y} = wx + b$ (or $X\theta$ in matrix form).
- We measure error with **MSE**: $J = \frac{1}{n}\sum(\hat{y}-y)^2$.
- Setting the gradient to zero gives the **normal equation**
  $X^\top X\,\theta = X^\top y$, solved with `np.linalg.solve`.
- The cost surface is a **convex bowl** — one global minimum.
- The same matrix recipe handles **many features** at once.
- **$R^2$** reports the fraction of variance explained.
- **scikit-learn** confirms our from-scratch answer.
- **Gradient descent** (Chapter 12) is the iterative alternative for big or
  non-linear problems.
"""))

# ---- Homework ----
c.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Effect of noise.** Regenerate the single-feature data with noise standard
   deviation `0.1`, then `3.0`. For each, fit the line and report the recovered
   $(w, b)$ and the $R^2$. How does more noise affect accuracy and $R^2$?

2. **Outliers and the squared penalty.** Take the single-feature dataset and
   move one point far away (e.g. add `+30` to one `y` value). Refit and overlay
   the new line on the old one. Explain *why* squaring the residuals makes
   least squares so sensitive to a single outlier.

3. **Polynomial regression by feature engineering.** Generate data from a curve
   $y = 0.5x^2 - x + 2 + \text{noise}$. Build a design matrix with columns
   $[1, x, x^2]$ and fit it with your `fit_normal_equation`. Plot the fitted
   curve over the data. (Linear regression in the *parameters*, even though the
   curve bends!)

4. **Train/test split.** Split the multivariate dataset into the first 150 rows
   (train) and the last 50 (test). Fit on the training rows only, then report
   $R^2$ on *both* sets. Are they similar? What would it mean if the test $R^2$
   were much worse than the train $R^2$?
"""))

save(os.path.join(CH, "10_linear_regression.ipynb"), c)
