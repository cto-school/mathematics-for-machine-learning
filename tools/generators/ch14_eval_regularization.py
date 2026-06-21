"""Generator for Chapter 14 — Model Evaluation & Regularization.

Run from anywhere:  python tools/generators/ch14_eval_regularization.py
Produces two notebooks in 14-model-evaluation-and-regularization/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "14-model-evaluation-and-regularization")


# ===========================================================================
# Notebook 14a — Evaluation metrics
# ===========================================================================
a = []

a.append(md(r"""
# Chapter 14a — Model Evaluation Metrics (Beyond Accuracy)

> **The big idea.** Training a model is only half the job. The other half is
> **measuring honestly how good it is**. A single number like "accuracy" can be
> deeply misleading. This notebook builds the metrics that working machine
> learning actually relies on — the **confusion matrix**, **precision**,
> **recall**, **F1**, the **threshold trade-off**, regression metrics
> (**MSE / RMSE / MAE / R²**), and the gold standard for choosing settings:
> **cross-validation** — all from scratch in NumPy.

By the end you will be able to:

- explain *why* accuracy lies on **imbalanced** data;
- build a **confusion matrix** (TP / FP / FN / TN) and read it;
- compute **precision**, **recall**, and **F1**, and say when each one matters;
- move the **decision threshold** and watch precision trade against recall (and
  sketch an **ROC**-style curve);
- recap the regression metrics **MSE / RMSE / MAE / R²**;
- split data into **train / validation / test** and run **k-fold
  cross-validation** to pick a hyperparameter.

Run every code cell with **Shift + Enter**. Edit and re-run freely.
"""))

a.append(md(r"""
## 0. Setup

We use only **NumPy** and **Matplotlib**. We fix the random seed with
`np.random.default_rng(0)` so every run looks identical.
"""))
a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)                   # reproducible randomness
np.set_printoptions(precision=4, suppress=True)  # tidy array printing
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 1. Why accuracy misleads on imbalanced classes

Recall **accuracy** = fraction of predictions that are correct,

$$\text{accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\{\hat{y}_i = y_i\}.$$

It is a fine summary when the two classes are roughly **balanced**. But suppose
we screen for a rare disease that only **1 in 100** people have. Consider the
"model" that ignores the input entirely and always predicts **healthy**.
"""))
a.append(code("""
# 1000 patients; only 1% are actually sick (label 1)
n = 1000
y_true = np.zeros(n, dtype=int)
y_true[:10] = 1                      # 10 sick, 990 healthy
rng.shuffle(y_true)

# the lazy classifier: ALWAYS predict 0 ("healthy"), no matter what
y_pred_lazy = np.zeros(n, dtype=int)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("accuracy of 'always healthy':", accuracy(y_true, y_pred_lazy))
"""))

a.append(md(r"""
**99% accuracy — and the model is useless.** It never catches a single sick
patient, which is the whole point of the test. Accuracy hid the failure because
it lumps the rare, important class in with the common one. We need metrics that
look at the **types** of mistakes separately. That is the confusion matrix.
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 2. The confusion matrix (TP / FP / FN / TN)

For binary classification (positive = 1, negative = 0) every prediction falls
into one of **four** boxes:

|                       | actual **1** (positive) | actual **0** (negative) |
|-----------------------|-------------------------|-------------------------|
| **predicted 1**       | **TP** true positive    | **FP** false positive   |
| **predicted 0**       | **FN** false negative   | **TN** true negative    |

- **TP** — correctly flagged a positive.
- **FP** — false alarm: said positive, was actually negative.
- **FN** — a miss: said negative, was actually positive.
- **TN** — correctly left a negative alone.

Everything else (accuracy, precision, recall, ...) is just arithmetic on these
four counts. Let's count them **from scratch**.
"""))
a.append(code("""
def confusion_counts(y_true, y_pred):
    \"\"\"Return TP, FP, FN, TN for binary labels in {0, 1}.\"\"\"
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))   # said 1, was 1
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))   # said 1, was 0
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))   # said 0, was 1
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))   # said 0, was 0
    return tp, fp, fn, tn

tp, fp, fn, tn = confusion_counts(y_true, y_pred_lazy)
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
# The lazy model: TP=0 (caught nobody), FN=10 (missed every sick patient).
"""))

a.append(md(r"""
Let's also build a slightly smarter (but imperfect) classifier so the matrix has
interesting numbers, then **visualize** it with `imshow`.
"""))
a.append(code("""
# a noisy but real classifier: catches most sick people, with some false alarms
y_pred = y_true.copy()
flip = rng.random(n) < 0.05            # randomly flip ~5% of labels as "errors"
y_pred[flip] = 1 - y_pred[flip]

tp, fp, fn, tn = confusion_counts(y_true, y_pred)
print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
"""))
a.append(code("""
# arrange the four counts into a 2x2 matrix and draw it
# rows = predicted (1 then 0), columns = actual (1 then 0)
M = np.array([[tp, fp],
              [fn, tn]])

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(M, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["actual 1", "actual 0"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["pred 1", "pred 0"])
for i in range(2):                     # write the count inside each cell
    for j in range(2):
        ax.text(j, i, M[i, j], ha="center", va="center",
                color="black", fontsize=14)
ax.set_title("Confusion matrix")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 3. Precision, recall, and F1

From the four counts we build the three metrics you will use constantly.

**Precision** — of everything we *flagged positive*, what fraction really was?
It punishes **false alarms (FP)**.

$$\text{precision} = \frac{TP}{TP + FP}.$$

**Recall** (a.k.a. *sensitivity*) — of all the *actual positives*, what fraction
did we *catch*? It punishes **misses (FN)**.

$$\text{recall} = \frac{TP}{TP + FN}.$$

**F1 score** — the **harmonic mean** of the two, a single number that is high
only when *both* are high:

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.$$
"""))
a.append(code("""
def precision(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

print("precision:", round(precision(y_true, y_pred), 3))
print("recall   :", round(recall(y_true, y_pred), 3))
print("F1       :", round(f1_score(y_true, y_pred), 3))
"""))

a.append(md(r"""
Now compare against the lazy "always healthy" model. Its accuracy was 99%, but:
"""))
a.append(code("""
print("lazy model precision:", round(precision(y_true, y_pred_lazy), 3))
print("lazy model recall   :", round(recall(y_true, y_pred_lazy), 3))
print("lazy model F1       :", round(f1_score(y_true, y_pred_lazy), 3))
# recall = 0: it caught NONE of the sick. F1 = 0. The 99% accuracy was a mirage.
"""))

a.append(md(r"""
### When does each one matter?

- **Recall matters most** when a **miss is expensive**: disease screening, fraud
  detection, "is there a pedestrian in front of the car?" You'd rather raise a
  few false alarms than let a true positive slip through.
- **Precision matters most** when a **false alarm is expensive**: flagging an
  email as spam (don't bury a real email), recommending an aggressive treatment.
- **F1** is the go-to single number when you care about **both** and the classes
  are imbalanced (so accuracy is untrustworthy).
"""))

# ---- Exercise 1 ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

A model is evaluated and produces these counts: **TP = 40, FP = 10, FN = 20,
TN = 930**. By hand-feeding these into the formulas (no data needed), compute the
**accuracy, precision, recall, and F1**. Which is highest, and why is accuracy so
optimistic here?
"""))
a.append(md("**Solution:**"))
a.append(code("""
tp, fp, fn, tn = 40, 10, 20, 930
total = tp + fp + fn + tn

acc = (tp + tn) / total
prec = tp / (tp + fp)
rec  = tp / (tp + fn)
f1   = 2 * prec * rec / (prec + rec)

print(f"accuracy : {acc:.3f}")
print(f"precision: {prec:.3f}")
print(f"recall   : {rec:.3f}")
print(f"F1       : {f1:.3f}")
# Accuracy (0.97) is highest because the 930 easy true-negatives dominate the
# count. Precision/recall ignore the TN box, so they reveal the model is far
# from perfect at the task that matters (catching positives).
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 4. The classification threshold trade-off

Most classifiers don't output 0/1 directly — they output a **score** or
**probability** $\hat{p} \in [0, 1]$. We turn that into a label by comparing it
to a **threshold** $t$:

$$\hat{y} = 1 \quad\text{if}\quad \hat{p} \ge t, \qquad \hat{y} = 0 \text{ otherwise.}$$

The default $t = 0.5$ is just a convention. **Moving $t$ trades precision against
recall:**

- **Lower $t$** → we flag more things positive → catch more true positives
  (**recall up**) but also more false alarms (**precision down**).
- **Raise $t$** → we are stricter → fewer false alarms (**precision up**) but we
  miss more (**recall down**).

Let's make scored data and sweep the threshold.
"""))
a.append(code("""
# 400 samples; positives tend to get higher scores, but the two overlap
m = 400
labels = rng.integers(0, 2, size=m)                  # true 0/1 labels
scores = np.where(labels == 1,
                  rng.normal(0.65, 0.18, m),         # positives: higher scores
                  rng.normal(0.35, 0.18, m))         # negatives: lower scores
scores = np.clip(scores, 0, 1)

thresholds = np.linspace(0, 1, 101)
precs, recs = [], []
for t in thresholds:
    pred = (scores >= t).astype(int)
    precs.append(precision(labels, pred))
    recs.append(recall(labels, pred))
precs, recs = np.array(precs), np.array(recs)
"""))
a.append(code("""
plt.plot(thresholds, precs, label="precision")
plt.plot(thresholds, recs,  label="recall")
plt.axvline(0.5, color="gray", ls="--", label="default t = 0.5")
plt.xlabel("threshold t"); plt.ylabel("score")
plt.title("Precision and recall vs decision threshold")
plt.legend(); plt.grid(True)
plt.show()
# As t rises (left to right): recall falls, precision rises. The classic tug-of-war.
"""))

a.append(md(r"""
### An ROC-style curve (optional but illuminating)

The **ROC curve** plots the **true-positive rate** (= recall) against the
**false-positive rate** as the threshold sweeps:

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}.$$

A perfect classifier hugs the **top-left** corner; the diagonal is random
guessing. The **area under the curve (AUC)** summarizes it in one number (1.0 =
perfect, 0.5 = coin flip).
"""))
a.append(code("""
tprs, fprs = [], []
for t in thresholds:
    pred = (scores >= t).astype(int)
    tp, fp, fn, tn = confusion_counts(labels, pred)
    tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
tprs, fprs = np.array(tprs), np.array(fprs)

# AUC via the trapezoidal rule (fprs run high->low as t increases, so negate)
auc = np.trapz(tprs[::-1], fprs[::-1])

plt.plot(fprs, tprs, "-o", markersize=3, label=f"ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random guessing")
plt.xlabel("false-positive rate"); plt.ylabel("true-positive rate (recall)")
plt.title("ROC curve")
plt.legend(); plt.grid(True)
plt.show()
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 5. Regression metrics: MSE, RMSE, MAE, R²

For **regression** the prediction is a number, so we measure *how far off* it is.

- **MSE** — mean squared error (what we usually minimize):
  $$\mathrm{MSE} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2.$$
- **RMSE** — its square root, **back in the original units** (nicer to report):
  $$\mathrm{RMSE} = \sqrt{\mathrm{MSE}}.$$
- **MAE** — mean absolute error, more robust to outliers:
  $$\mathrm{MAE} = \frac{1}{n}\sum_i |\hat{y}_i - y_i|.$$
- **R²** (coefficient of determination) — the fraction of variance the model
  explains. **1.0 is perfect**; **0** means "no better than predicting the mean
  $\bar{y}$"; negative means *worse* than the mean.
  $$R^2 = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}{\sum_i (y_i - \bar{y})^2}.$$
"""))
a.append(code("""
def mse(y_true, y_hat):
    return np.mean((y_hat - y_true)**2)

def rmse(y_true, y_hat):
    return np.sqrt(mse(y_true, y_hat))

def mae(y_true, y_hat):
    return np.mean(np.abs(y_hat - y_true))

def r2_score(y_true, y_hat):
    ss_res = np.sum((y_hat - y_true)**2)          # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)  # total variance of y
    return 1 - ss_res / ss_tot
"""))
a.append(code("""
# toy regression: y = 3x + 2 + noise, fit a line, measure all four metrics
x = np.linspace(0, 10, 50)
y = 3 * x + 2 + rng.normal(0, 2.0, x.size)

coeffs = np.polyfit(x, y, deg=1)         # least-squares line
y_hat = np.polyval(coeffs, x)

print(f"MSE : {mse(y, y_hat):.4f}")
print(f"RMSE: {rmse(y, y_hat):.4f}   (same units as y)")
print(f"MAE : {mae(y, y_hat):.4f}")
print(f"R2  : {r2_score(y, y_hat):.4f}   (close to 1 = good fit)")
"""))

# ---- Exercise 2 ----
a.append(md(r"""
---
## ✍️ Exercise 2 (solution included)

Show that a model which **always predicts the mean** $\bar{y}$ gets $R^2 = 0$
exactly. Compute it on the data above to confirm, and explain in one line why
this is the natural "zero" of $R^2$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
y_mean_pred = np.full_like(y, np.mean(y))    # constant prediction = y-bar
print("R2 of mean-predictor:", round(r2_score(y, y_mean_pred), 6))
# It is 0 because then ss_res == ss_tot, so 1 - ss_res/ss_tot = 0. R2 measures
# improvement OVER just guessing the mean, so the mean-predictor scores exactly 0.
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 6. Validating properly: train / validation / test

In Chapter 9 we split data into **train** and **test**. But there is a subtle
trap: if we keep tweaking the model while watching the **test** error, we slowly
**leak** the test set into our decisions — the test score stops being honest.

The fix is a **three-way split**:

- **training set** — fit the model parameters $\theta$;
- **validation set** — choose **hyperparameters** (degree, $\lambda$, ...) and
  compare models;
- **test set** — locked in a vault, touched **once** at the very end.

Let's build the three-way split from scratch (shuffle, then slice).
"""))
a.append(code("""
def train_val_test_split(X, y, val_frac=0.2, test_frac=0.2, rng=rng):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)                          # shuffle so order can't poison it

    n_val  = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    val_idx   = idx[:n_val]
    test_idx  = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]          # whatever remains -> train

    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])

# demo on a curve-plus-noise dataset
N = 60
xx = np.sort(rng.uniform(0, 5, N))
yy = np.sin(1.2 * xx) + 0.5 * xx + rng.normal(0, 0.35, N)
XX = xx.reshape(-1, 1)

Xtr, Xva, Xte, ytr, yva, yte = train_val_test_split(XX, yy)
print("train / val / test sizes:", Xtr.shape[0], Xva.shape[0], Xte.shape[0])
"""))

a.append(md(r"""
The weakness of a single validation set: the score depends on *which* points
happened to land in it, and we "waste" those points by never training on them.
With small datasets that is costly. The standard cure is **cross-validation**.
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## 7. k-fold cross-validation (from scratch)

**k-fold cross-validation** uses *all* the data for both training and validation,
fairly:

1. Shuffle, then split the data into **k** equal **folds**.
2. For each fold $j$: train on the other $k-1$ folds, validate on fold $j$.
3. **Average** the $k$ validation scores.

Every point is used for validation exactly once and for training $k-1$ times. The
averaged score is a far more **stable** estimate than a single split. Let's
implement it and use it to **pick the polynomial degree**.
"""))
a.append(code("""
def k_fold_indices(n, k, rng=rng):
    \"\"\"Yield (train_idx, val_idx) pairs for k folds.\"\"\"
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)            # k roughly-equal chunks
    for j in range(k):
        val_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
        yield train_idx, val_idx

def cv_score_poly(x, y, degree, k=5, rng=rng):
    \"\"\"Average validation MSE of a degree-`degree` polynomial over k folds.\"\"\"
    scores = []
    for train_idx, val_idx in k_fold_indices(len(x), k, rng=rng):
        c = np.polyfit(x[train_idx], y[train_idx], degree)   # fit on k-1 folds
        pred = np.polyval(c, x[val_idx])                     # predict held-out fold
        scores.append(mse(y[val_idx], pred))
    return np.mean(scores)                                   # average the folds
"""))
a.append(code("""
# use cross-validation to choose the best polynomial degree
degrees = range(1, 11)
cv_errs = [cv_score_poly(xx, yy, deg, k=5,
                         rng=np.random.default_rng(deg))   # fresh rng per degree
           for deg in degrees]

best_deg = list(degrees)[int(np.argmin(cv_errs))]
print("best degree by 5-fold CV:", best_deg)

plt.plot(list(degrees), cv_errs, "o-")
plt.axvline(best_deg, color="gray", ls="--", label=f"best deg = {best_deg}")
plt.xlabel("polynomial degree"); plt.ylabel("mean CV MSE")
plt.title("Hyperparameter selection by cross-validation")
plt.legend(); plt.grid(True)
plt.show()
"""))

a.append(md(r"""
Once cross-validation has chosen the degree, we **retrain on all the
train+validation data** at that degree, and only *then* report the **test** error
as our final honest estimate. We never used the test set to make any choice.
"""))

a.append(md(r"""
### Optional: the same idea in scikit-learn

`scikit-learn` packages all of this. Our from-scratch versions match its
behaviour; the library just saves typing. (Skipped silently if not installed.)
"""))
a.append(code("""
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(PolynomialFeatures(best_deg), LinearRegression())
    # sklearn maximizes score, so it returns NEGATIVE MSE; flip the sign
    neg_mse = cross_val_score(model, xx.reshape(-1, 1), yy,
                              cv=5, scoring="neg_mean_squared_error")
    print("sklearn 5-fold mean MSE:", round(-neg_mse.mean(), 4))
except Exception as e:
    print("scikit-learn not available, skipping:", e)
"""))

# ---------------------------------------------------------------------------
a.append(md(r"""
## Recap

- **Accuracy lies** on imbalanced data — a do-nothing model can score 99%.
- The **confusion matrix** (TP / FP / FN / TN) separates the *types* of error.
- **Precision** = $TP/(TP+FP)$ (punishes false alarms); **recall** =
  $TP/(TP+FN)$ (punishes misses); **F1** is their harmonic mean.
- The **decision threshold** trades precision against recall; the **ROC curve**
  (and its **AUC**) summarizes that trade-off across all thresholds.
- Regression metrics: **MSE**, **RMSE** (same units), **MAE** (robust), and
  **R²** (variance explained; mean-predictor = 0).
- Split **train / validation / test**; pick hyperparameters with **k-fold
  cross-validation**, and touch the **test set only once**.
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **Specificity & balanced accuracy.** Write `specificity = TN / (TN + FP)`
   from the confusion counts. Then compute **balanced accuracy** =
   (recall + specificity) / 2 for the lazy "always healthy" model of Section 1.
   Why is balanced accuracy a better one-number summary than plain accuracy here?

2. **The $F_\beta$ score.** The general $F_\beta = (1+\beta^2)\dfrac{P\cdot R}
   {\beta^2 P + R}$ weights recall $\beta$ times as much as precision. Implement
   it and confirm $F_1$ matches your Section 3 result. Compute $F_2$ (recall-
   heavy) and $F_{0.5}$ (precision-heavy) for the Section 4 data at $t = 0.5$.

3. **Pick a threshold for a goal.** Using the Section 4 scored data, find the
   smallest threshold $t$ that achieves **precision ≥ 0.9**, and report the recall
   you get there. (Sweep `thresholds` and check the precision array.)

4. **CV variance.** Run your 5-fold `cv_score_poly` for degree 4 using **ten
   different shuffles** (ten different rng seeds) and print the mean and standard
   deviation of the ten CV scores. Then repeat with **k = 10** folds. Does more
   folds make the estimate more stable?
"""))

save(os.path.join(CH, "14a_evaluation_metrics.ipynb"), a)


# ===========================================================================
# Notebook 14b — Regularization
# ===========================================================================
b = []

b.append(md(r"""
# Chapter 14b — Regularization (Fixing Overfitting)

> **The big idea.** In Chapter 9 we saw a high-degree polynomial **overfit**:
> it wiggled wildly to chase noise and generalized terribly. **Regularization**
> is the cure. We add a penalty that discourages large weights, so the model
> *prefers* a smoother, simpler fit unless the data really demands complexity.

By the end you will be able to:

- recreate **overfitting** with a high-degree polynomial on noisy data;
- derive and code **L2 (ridge)** regression, including the modified normal
  equation $\theta = (X^\top X + \lambda I)^{-1} X^\top y$;
- **sweep $\lambda$** and read the train-vs-validation **U-curve**, and watch the
  fitted curve get smoother as $\lambda$ grows;
- understand **L1 (lasso)** and why it drives weights to **exactly zero**
  (automatic feature selection);
- restate the **bias–variance** trade-off in terms of $\lambda$;
- see **weight decay** as L2's effect inside a gradient-descent step
  (connecting to Chapter 12).

Run every code cell with **Shift + Enter**.
"""))

b.append(md(r"""
## 0. Setup
"""))
b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

def mse(y_true, y_hat):
    return np.mean((y_hat - y_true)**2)
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 1. Recreating overfitting

We make data from a gentle true curve plus noise, then fit a **high-degree
polynomial** — far more flexible than the data needs. As in Chapter 9, it will
hug the training points and swing wildly between them.
"""))
b.append(code("""
def true_fn(x):
    return np.sin(1.2 * x) + 0.5 * x        # the smooth underlying pattern

n = 25
x = np.sort(rng.uniform(0, 5, n))
y = true_fn(x) + rng.normal(0, 0.35, n)     # noisy observations

# split into train / validation (we'll use validation to tune lambda)
idx = np.arange(n); rng.shuffle(idx)
cut = int(0.6 * n)
tr, va = idx[:cut], idx[cut:]
xtr, ytr = x[tr], y[tr]
xva, yva = x[va], y[va]
print("train size:", xtr.size, " val size:", xva.size)
"""))
b.append(code("""
DEGREE = 12                                  # deliberately too flexible
grid = np.linspace(0, 5, 300)

c = np.polyfit(xtr, ytr, DEGREE)             # plain (unregularized) fit
plt.scatter(xtr, ytr, label="train", color="C0")
plt.scatter(xva, yva, label="val", color="C1", marker="s")
plt.plot(grid, np.polyval(c, grid), "k", label=f"degree {DEGREE} fit")
plt.plot(grid, true_fn(grid), "g--", alpha=0.6, label="truth")
plt.ylim(-2, 5)
plt.title("Overfitting: a degree-12 polynomial chases the noise")
plt.legend(fontsize=8); plt.grid(True)
plt.show()

print("train MSE:", round(mse(ytr, np.polyval(c, xtr)), 4))
print("val   MSE:", round(mse(yva, np.polyval(c, xva)), 4))
"""))

b.append(md(r"""
Tiny train error, large validation error — the signature of overfitting. Notice
the wild swings come from **huge polynomial coefficients** fighting each other.
Regularization attacks exactly that.
"""))
b.append(code("""
print("largest |coefficient| in the overfit model:", round(np.max(np.abs(c)), 1))
# Overfit models tend to have enormous weights. We will penalize that.
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 2. L2 regularization (ridge regression)

To fit a polynomial as linear regression we build a **design matrix** of powers:

$$X = \begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^p \\
\vdots & & & & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^p \end{bmatrix},
\qquad f_\theta(x) = X\theta.$$

Ordinary least squares minimizes $\mathrm{MSE} = \frac{1}{n}\|X\theta - y\|^2$,
solved by the **normal equation** $\theta = (X^\top X)^{-1} X^\top y$.

**Ridge** adds an **L2 penalty** on the weights to the cost:

$$J(\theta) = \frac{1}{n}\|X\theta - y\|^2 \; + \; \lambda \|\theta\|^2,
\qquad \|\theta\|^2 = \sum_j \theta_j^2.$$

Setting $\nabla J = 0$ gives the **modified normal equation**:

$$\boxed{\;\theta = (X^\top X + \lambda I)^{-1} X^\top y\;}$$

The extra $\lambda I$ "shrinks" the weights toward zero — larger $\lambda$ means a
stronger pull, hence a smoother fit. Let's code it from scratch.
"""))
b.append(code("""
def poly_design(x, degree):
    \"\"\"Build the (n, degree+1) matrix of powers [1, x, x^2, ..., x^degree].\"\"\"
    return np.vander(x, degree + 1, increasing=True)

def ridge_fit(x, y, degree, lam):
    \"\"\"Ridge regression via the modified normal equation.\"\"\"
    X = poly_design(x, degree)
    p = X.shape[1]
    I = np.eye(p)
    I[0, 0] = 0.0                 # by convention, do NOT penalize the bias term
    theta = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return theta

def ridge_predict(theta, x):
    return poly_design(x, len(theta) - 1) @ theta
"""))
b.append(code("""
# sanity check: lambda = 0 reproduces ordinary least squares
theta0 = ridge_fit(xtr, ytr, degree=3, lam=0.0)
ref = np.polyfit(xtr, ytr, 3)[::-1]          # polyfit returns high->low power
print("ridge (lam=0):", theta0)
print("polyfit ref  :", ref)
print("match:", np.allclose(theta0, ref))
"""))

b.append(md(r"""
Now fit the **same degree-12** model with a few values of $\lambda$ and watch the
curve relax from wild to smooth.
"""))
b.append(code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, lam in zip(axes, [0.0, 1e-4, 1e-1]):
    theta = ridge_fit(xtr, ytr, DEGREE, lam)
    ax.scatter(xtr, ytr, s=20, color="C0", label="train")
    ax.scatter(xva, yva, s=25, color="C1", marker="s", label="val")
    ax.plot(grid, ridge_predict(theta, grid), "k", label=f"lam={lam}")
    ax.plot(grid, true_fn(grid), "g--", alpha=0.6, label="truth")
    ax.set_ylim(-2, 5); ax.set_title(f"lambda = {lam}")
    ax.legend(fontsize=8); ax.grid(True)
axes[0].set_ylabel("y")
plt.suptitle("Ridge: larger lambda -> smoother fit")
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 3. Sweeping $\lambda$: the regularization U-curve

In Chapter 9 we varied **model complexity** (degree) to get a U-shaped test
curve. Here the degree is **fixed** and we vary the **regularization strength**
$\lambda$ instead. The shape is the same idea, mirrored:

- **$\lambda$ too small** → no penalty → **overfitting** (low train, high val).
- **$\lambda$ too large** → weights crushed toward 0 → **underfitting** (high
  train *and* val).
- **In between** → the sweet spot, the bottom of the U.
"""))
b.append(code("""
lambdas = np.logspace(-6, 1, 40)             # 1e-6 ... 1e1, log-spaced
train_errs, val_errs = [], []
for lam in lambdas:
    theta = ridge_fit(xtr, ytr, DEGREE, lam)
    train_errs.append(mse(ytr, ridge_predict(theta, xtr)))
    val_errs.append(mse(yva, ridge_predict(theta, xva)))
train_errs, val_errs = np.array(train_errs), np.array(val_errs)

best_lam = lambdas[int(np.argmin(val_errs))]
print("best lambda (lowest val error):", best_lam)
"""))
b.append(code("""
plt.plot(lambdas, train_errs, "o-", label="train error")
plt.plot(lambdas, val_errs,  "s-", label="validation error")
plt.axvline(best_lam, color="gray", ls="--", label=f"best lam = {best_lam:.1e}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("regularization strength lambda (log scale)")
plt.ylabel("MSE (log scale)")
plt.title("Train vs validation error vs lambda (the U-curve)")
plt.legend(); plt.grid(True, which="both")
plt.show()
"""))

# ---- Exercise 1 ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Regularization works by **shrinking weights**. For the degree-12 model, compute
the L2 norm $\|\theta\|$ of the fitted weights (excluding the bias) for three
values $\lambda \in \{0, 10^{-3}, 1\}$. Confirm the norm **shrinks** as $\lambda$
grows — that *is* regularization in one number.
"""))
b.append(md("**Solution:**"))
b.append(code("""
for lam in [0.0, 1e-3, 1.0]:
    theta = ridge_fit(xtr, ytr, DEGREE, lam)
    norm = np.linalg.norm(theta[1:])      # skip the bias term theta[0]
    print(f"lambda = {lam:<6}  ||theta|| = {norm:.4f}")
# The weight norm collapses as lambda increases: the penalty literally pulls the
# coefficients toward zero, which is why the fitted curve gets smoother.
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 4. L1 regularization (lasso): sparsity & feature selection

**Lasso** swaps the squared penalty for an **absolute-value** penalty:

$$J(\theta) = \frac{1}{n}\|X\theta - y\|^2 \; + \; \lambda \|\theta\|_1,
\qquad \|\theta\|_1 = \sum_j |\theta_j|.$$

The geometry of $|\cdot|$ has a remarkable effect: instead of merely *shrinking*
weights (as L2 does), L1 drives many of them to **exactly zero**. A weight of
exactly zero means that feature is **dropped** — so lasso performs automatic
**feature selection**, giving a *sparse*, interpretable model.

> **Intuition.** The gradient of $\lambda|\theta_j|$ is a constant $\pm\lambda$
> that does **not** vanish as $\theta_j \to 0$ — it keeps pushing until the weight
> hits zero and sticks. L2's gradient $2\lambda\theta_j$ fades as $\theta_j \to 0$,
> so L2 only shrinks, never zeroes out.

A full lasso solver uses **coordinate descent**; here is a short **gradient-based**
demo using a *subgradient* of $|\theta|$ (i.e. $\mathrm{sign}(\theta)$), which is
enough to see the sparsity emerge.
"""))
b.append(code("""
# build a regression where ONLY a few of many features actually matter
n_samp, n_feat = 60, 10
A = rng.normal(0, 1, (n_samp, n_feat))         # 10 candidate features
true_w = np.zeros(n_feat)
true_w[2] = 3.0                                # only features 2, 5, 7 are real
true_w[5] = -2.0
true_w[7] = 1.5
target = A @ true_w + rng.normal(0, 0.1, n_samp)

def lasso_gd(A, y, lam, lr=0.01, steps=4000):
    \"\"\"Minimize (1/n)||Aw - y||^2 + lam*||w||_1 by (sub)gradient descent.\"\"\"
    n = A.shape[0]
    w = np.zeros(A.shape[1])
    for _ in range(steps):
        grad_mse = (2 / n) * A.T @ (A @ w - y)     # gradient of the MSE part
        grad_l1  = lam * np.sign(w)                # subgradient of lam*||w||_1
        w -= lr * (grad_mse + grad_l1)
    return w

w_lasso = lasso_gd(A, target, lam=0.3)
w_lasso[np.abs(w_lasso) < 1e-2] = 0.0             # tiny weights -> treat as zero
print("true weights :", true_w)
print("lasso weights:", np.round(w_lasso, 2))
print("nonzero count -> true:", int(np.sum(true_w != 0)),
      " lasso:", int(np.sum(w_lasso != 0)))
"""))
b.append(code("""
# compare against ridge on the SAME problem: ridge shrinks but keeps everything
def ridge_gd(A, y, lam, lr=0.01, steps=4000):
    n = A.shape[0]
    w = np.zeros(A.shape[1])
    for _ in range(steps):
        grad = (2 / n) * A.T @ (A @ w - y) + 2 * lam * w   # L2 gradient
        w -= lr * grad
    return w

w_ridge = ridge_gd(A, target, lam=0.3)

idxs = np.arange(n_feat)
width = 0.4
plt.bar(idxs - width/2, w_lasso, width, label="lasso (L1)")
plt.bar(idxs + width/2, w_ridge, width, label="ridge (L2)")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("feature index"); plt.ylabel("learned weight")
plt.title("L1 zeroes out irrelevant features; L2 only shrinks them")
plt.legend(); plt.grid(True, axis="y")
plt.show()
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 5. Bias–variance, restated in terms of $\lambda$

In Chapter 9 the bias–variance trade-off was controlled by **model complexity**.
Regularization gives us a **continuous dial** for the same trade-off:

$$\text{test error} = \text{bias}^2 + \text{variance} + \text{noise}.$$

- **Small $\lambda$** → the model is free to be complex → **low bias, high
  variance** → overfitting (left/low end of the U).
- **Large $\lambda$** → weights forced toward zero → the model is rigid → **high
  bias, low variance** → underfitting (right/high end of the U).
- The **best $\lambda$** balances the two — exactly the bottom of the U-curve we
  found in Section 3.

So $\lambda$ is a smooth knob that slides the model along the bias–variance curve
*without changing its degree* — often easier to tune than discrete complexity.
"""))

# ---- Exercise 2 ----
b.append(md(r"""
---
## ✍️ Exercise 2 (solution included)

Demonstrate the **variance** half of the story. Generate **8 fresh noisy
datasets** from `true_fn`, and for each fit a degree-12 model **twice**: once
unregularized ($\lambda = 0$) and once with strong ridge ($\lambda = 0.1$).
Overlay the fitted curves. The unregularized fits should scatter wildly (high
variance); the regularized ones should cluster tightly.
"""))
b.append(md("**Solution:**"))
b.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for trial in range(8):
    xs = np.sort(rng.uniform(0, 5, 20))
    ys = true_fn(xs) + rng.normal(0, 0.35, 20)
    for ax, lam in zip(axes, [0.0, 0.1]):
        theta = ridge_fit(xs, ys, DEGREE, lam)
        ax.plot(grid, ridge_predict(theta, grid), alpha=0.6)

for ax, lam in zip(axes, [0.0, 0.1]):
    ax.plot(grid, true_fn(grid), "k--", lw=2, label="truth")
    ax.set_ylim(-2, 5); ax.set_title(f"8 fits, lambda = {lam}")
    ax.legend(); ax.grid(True)
axes[0].set_ylabel("y")
plt.suptitle("High variance (left) vs regularized low variance (right)")
plt.tight_layout()
plt.show()
# Left: the curves fly all over the place -> high variance. Right: ridge pins
# them down near the truth -> variance reduced (at the cost of a little bias).
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## 6. Weight decay: L2 inside a gradient-descent step

In Chapter 12 the gradient-descent update was

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathrm{MSE}.$$

What happens when we add the L2 penalty $\lambda\|\theta\|^2$ to the cost? Its
gradient is $2\lambda\theta$, so the update becomes

$$\theta \leftarrow \theta - \eta\bigl(\nabla_\theta \mathrm{MSE} + 2\lambda\theta\bigr)
= \underbrace{(1 - 2\eta\lambda)\,\theta}_{\text{shrink first}} \; - \; \eta\,\nabla_\theta \mathrm{MSE}.$$

Every step **multiplies $\theta$ by a factor just below 1** *before* the usual
gradient move — it literally **decays the weights** toward zero. This is why L2
regularization is called **weight decay** in deep learning. Let's watch the
weight norm shrink during training.
"""))
b.append(code("""
def gd_with_weight_decay(A, y, lam, lr=0.05, steps=200):
    \"\"\"Gradient descent on MSE with L2 weight decay; track the weight norm.\"\"\"
    n = A.shape[0]
    w = rng.normal(0, 1, A.shape[1])           # start from random largish weights
    norms = []
    for _ in range(steps):
        grad_mse = (2 / n) * A.T @ (A @ w - y)
        w = (1 - 2 * lr * lam) * w - lr * grad_mse   # decay, then descend
        norms.append(np.linalg.norm(w))
    return w, norms

# use the Section 4 data; compare no decay vs decay
_, norms_no   = gd_with_weight_decay(A, target, lam=0.0)
_, norms_yes  = gd_with_weight_decay(A, target, lam=0.5)

plt.plot(norms_no,  label="lambda = 0   (no weight decay)")
plt.plot(norms_yes, label="lambda = 0.5 (weight decay)")
plt.xlabel("gradient-descent step"); plt.ylabel("||theta||")
plt.title("Weight decay keeps the weight norm small during training")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
The decay curve settles at a **smaller weight norm** — the same shrinkage ridge
achieves in closed form, now realized step-by-step inside gradient descent. The
two views (penalized cost vs. decayed update) are mathematically the same thing.
"""))

b.append(md(r"""
### Optional: ridge in scikit-learn

`scikit-learn`'s `Ridge` solves the very same problem. (Skipped if unavailable;
note its `alpha` plays the role of our `lambda`.)
"""))
b.append(code("""
try:
    from sklearn.linear_model import Ridge
    X12 = poly_design(xtr, DEGREE)
    model = Ridge(alpha=best_lam, fit_intercept=False)   # bias already in column 0
    model.fit(X12, ytr)
    ours = ridge_fit(xtr, ytr, DEGREE, best_lam)
    print("sklearn vs our ridge weights close:",
          np.allclose(model.coef_, ours, atol=1e-6))
except Exception as e:
    print("scikit-learn not available, skipping:", e)
"""))

# ---------------------------------------------------------------------------
b.append(md(r"""
## Recap

- **Overfitting** shows up as wild curves and **huge weights**; regularization
  penalizes large weights so the model prefers a simpler fit.
- **L2 / ridge:** cost $J = \mathrm{MSE} + \lambda\|\theta\|^2$, solved by the
  modified normal equation $\theta = (X^\top X + \lambda I)^{-1} X^\top y$.
  Larger $\lambda$ → smaller weights → smoother fit.
- Sweeping $\lambda$ traces a **U-curve** of validation error; its bottom is the
  best regularization strength.
- **L1 / lasso:** cost uses $\|\theta\|_1$; it drives weights to **exactly zero**,
  giving sparse models and automatic **feature selection**.
- $\lambda$ is a continuous **bias–variance dial**: small = high variance
  (overfit), large = high bias (underfit).
- **Weight decay** is L2 seen inside a gradient step: each update shrinks
  $\theta$ by $(1 - 2\eta\lambda)$ before descending (connects to Chapter 12).
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **CV-tuned ridge.** Combine the two notebooks: write a function that, for a
   list of $\lambda$ values, computes the **5-fold cross-validation** MSE of the
   degree-12 ridge model and returns the best $\lambda$. Compare it to the single
   validation split's choice from Section 3.

2. **Penalizing the bias.** In `ridge_fit` we set `I[0,0] = 0` so the bias term
   is not penalized. Remove that line and re-run the $\lambda$ sweep. How does the
   fit (especially its overall level) change, and why is it usually wrong to
   shrink the intercept?

3. **Elastic net.** Combine both penalties:
   $J = \mathrm{MSE} + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|^2$. Modify
   `lasso_gd` to add the L2 gradient term, run it on the Section 4 data, and
   describe how the weights differ from pure lasso and pure ridge.

4. **Decay rate.** In `gd_with_weight_decay`, the per-step shrink factor is
   $(1 - 2\eta\lambda)$. For $\eta = 0.05$, what value of $\lambda$ would make the
   weights *grow* instead of decay (i.e. factor > 1 or < 0)? Verify experimentally
   and explain what goes wrong.
"""))

save(os.path.join(CH, "14b_regularization.ipynb"), b)
