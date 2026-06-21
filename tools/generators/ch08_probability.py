"""Generator for Chapter 08 — Probability & Statistics.

Run from anywhere:  python tools/generators/ch08_probability.py
Produces two notebooks in 08-probability-and-statistics/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "08-probability-and-statistics")


# ---------------------------------------------------------------------------
# Notebook 08a — Probability
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 08a — Probability

Probability is the mathematics of **uncertainty**. Machine learning is built on
it: data is noisy, models make predictions with confidence, and we constantly
ask *"how likely is this?"*

In this notebook we take a hands-on view. Instead of only proving formulas, we
**simulate** random experiments thousands of times and watch the theory emerge
from the data. The tool for that is NumPy's modern random generator:

```python
rng = np.random.default_rng(0)   # 0 is a "seed" -> reproducible randomness
```

Run each cell with **Shift + Enter**. Because we fix the seed, you will get the
exact same "random" numbers every time — which is perfect for learning.
"""))

a.append(code("""
import numpy as np
import matplotlib.pyplot as plt

# One generator, used everywhere. The seed 0 makes results reproducible.
rng = np.random.default_rng(0)
print(rng)
"""))

a.append(md(r"""
## 1. Sample spaces and events

A random **experiment** has a set of possible outcomes called the **sample
space** $\Omega$. An **event** is a subset $A \subseteq \Omega$ — a collection
of outcomes we care about.

- Toss a coin: $\Omega = \{H, T\}$.
- Roll a die: $\Omega = \{1,2,3,4,5,6\}$. The event "even" is $A = \{2,4,6\}$.

The **probability** of an event is a number $P(A) \in [0,1]$. For a *fair* die
every outcome is equally likely, so

$$P(A) = \frac{|A|}{|\Omega|} = \frac{\text{favourable outcomes}}{\text{total outcomes}}.$$
"""))
a.append(code("""
omega = [1, 2, 3, 4, 5, 6]      # sample space of a die
even  = [x for x in omega if x % 2 == 0]   # the event "even"
print("sample space:", omega)
print("event 'even':", even)
print("P(even) =", len(even) / len(omega))   # 3/6 = 0.5
"""))

a.append(md(r"""
## 2. Simulating coins and dice

Real probability becomes vivid when we *run the experiment*. The generator can
draw random integers and make random choices for us.
"""))
a.append(code("""
# Toss a fair coin 10 times. 0 = Tails, 1 = Heads.
coins = rng.integers(0, 2, size=10)    # integers in [0, 2) -> 0 or 1
print("coin tosses:", coins)

# Roll a fair die 10 times: integers in [1, 7) -> 1..6
dice = rng.integers(1, 7, size=10)
print("die rolls:  ", dice)
"""))

a.append(md(r"""
## 3. Estimating probability by frequency

If we repeat an experiment $N$ times and event $A$ happens $n_A$ times, the
**relative frequency** $n_A / N$ is an *estimate* of $P(A)$:

$$\widehat{P}(A) = \frac{n_A}{N}.$$

Let us estimate $P(\text{roll} \le 2)$, whose true value is $2/6 \approx 0.333$.
"""))
a.append(code("""
N = 100_000
rolls = rng.integers(1, 7, size=N)        # N die rolls

# Count how many satisfy the event, then divide by N.
hits = np.count_nonzero(rolls <= 2)       # number of 1s and 2s
estimate = hits / N
print("estimate of P(roll <= 2):", estimate)
print("true value             :", 2 / 6)
"""))

a.append(md(r"""
## 4. The Law of Large Numbers

The estimate above was close, but not exact. The **Law of Large Numbers** (LLN)
says: as $N \to \infty$, the relative frequency converges to the true
probability. We can *see* this by plotting the running estimate as we collect
more and more samples.
"""))
a.append(code("""
N = 5000
tosses = rng.integers(0, 2, size=N)          # 1 = Heads, 0 = Tails

# np.cumsum gives running totals: [t0, t0+t1, t0+t1+t2, ...]
running_heads = np.cumsum(tosses)            # cumulative number of heads
trial_number  = np.arange(1, N + 1)          # 1, 2, 3, ..., N
running_estimate = running_heads / trial_number   # estimate after each toss

plt.figure(figsize=(8, 4))
plt.plot(trial_number, running_estimate, label="running estimate of P(Heads)")
plt.axhline(0.5, color="red", linestyle="--", label="true value 0.5")
plt.xlabel("number of tosses")
plt.ylabel("estimated P(Heads)")
plt.title("Law of Large Numbers: the estimate settles down to 0.5")
plt.legend()
plt.grid(True)
plt.show()
"""))

a.append(md(r"""
Early on the estimate jumps around wildly; with more data it locks onto $0.5$.
That settling-down is the LLN in action — and it is *why* averaging over lots of
data works in machine learning.
"""))

a.append(md(r"""
## 5. Random variables

A **random variable** $X$ assigns a number to each outcome. For one die,
$X$ = the face shown. We can describe $X$ by its **distribution**: the
probability of each value. For a fair die, $P(X = k) = 1/6$ for $k = 1,\dots,6$.
"""))
a.append(code("""
N = 60_000
X = rng.integers(1, 7, size=N)            # N samples of the die random variable

# Estimate P(X = k) for each face by counting.
faces = np.arange(1, 7)
counts = np.array([np.count_nonzero(X == k) for k in faces])
probs  = counts / N
for k, p in zip(faces, probs):
    print(f"P(X = {k}) estimated as {p:.4f}   (true 0.1667)")
"""))

a.append(md(r"""
## 6. Expectation and variance

The **expectation** (mean) of $X$ is the long-run average value, weighting each
possible value by its probability:

$$\mathbb{E}[X] = \sum_k k \, P(X = k).$$

The **variance** measures spread around the mean:

$$\operatorname{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big].$$

For a fair die the theory gives $\mathbb{E}[X] = 3.5$ and
$\operatorname{Var}(X) = 35/12 \approx 2.9167$. Let us check both against a big
sample — the sample mean and sample variance are estimates of these.
"""))
a.append(code("""
# Theoretical values from the definition (sum over k of k * P(X=k)):
k = np.arange(1, 7)
p = np.full(6, 1/6)                 # uniform probabilities 1/6 each
EX_theory  = np.sum(k * p)                       # expectation
VarX_theory = np.sum((k - EX_theory)**2 * p)     # variance
print("theory : E[X] =", EX_theory, " Var(X) =", VarX_theory)

# Estimates from samples: np.mean and np.var do exactly the averaging above.
X = rng.integers(1, 7, size=200_000)
print("sample : E[X] =", np.mean(X), " Var(X) =", np.var(X))
"""))

a.append(md(r"""
## 7. Independence

Two events $A$ and $B$ are **independent** if one happening tells us nothing
about the other:

$$P(A \cap B) = P(A)\,P(B).$$

Two separate dice are independent. Let $A$ = "first die is 6" and
$B$ = "second die is 6". Then $P(A) = P(B) = 1/6$, so
$P(A \cap B) = 1/36 \approx 0.0278$.
"""))
a.append(code("""
N = 200_000
die1 = rng.integers(1, 7, size=N)
die2 = rng.integers(1, 7, size=N)

PA  = np.mean(die1 == 6)                 # estimate P(A)
PB  = np.mean(die2 == 6)                 # estimate P(B)
PAB = np.mean((die1 == 6) & (die2 == 6)) # estimate P(A and B)

print("P(A)        :", PA)
print("P(B)        :", PB)
print("P(A and B)  :", PAB)
print("P(A)*P(B)   :", PA * PB, "  <- matches, so A and B are independent")
"""))

# ---- Exercise 1 (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Estimate the probability that the **sum of two dice equals 7** by simulating
$200{,}000$ rolls of a pair of dice. Compare with the true value $6/36 = 1/6$.
"""))
a.append(md("**Solution:**"))
a.append(code("""
N = 200_000
d1 = rng.integers(1, 7, size=N)
d2 = rng.integers(1, 7, size=N)
totals = d1 + d2                          # elementwise sum of the two dice

estimate = np.mean(totals == 7)           # fraction of rolls summing to 7
print("estimate of P(sum = 7):", estimate)
print("true value            :", 6 / 36)
"""))

# ---- Exercise 2 (with solution) ----
a.append(md(r"""
## ✍️ Exercise 2 (solution included)

A random variable $Y$ is the **maximum** of two dice. Estimate $\mathbb{E}[Y]$
from $200{,}000$ samples. (For reference, the exact answer is
$\tfrac{161}{36} \approx 4.472$.)
"""))
a.append(md("**Solution:**"))
a.append(code("""
N = 200_000
d1 = rng.integers(1, 7, size=N)
d2 = rng.integers(1, 7, size=N)
Y = np.maximum(d1, d2)                     # elementwise maximum

print("estimated E[Y]:", np.mean(Y))
print("exact value   :", 161 / 36)
"""))

a.append(md(r"""
## 8. Conditional probability and Bayes' rule

The probability of $A$ **given** that $B$ happened is

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}.$$

Rearranging gives **Bayes' rule**, the engine of probabilistic reasoning:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}.$$

### A worked example: a medical test

A disease affects $1\%$ of people: $P(D) = 0.01$. A test is good but not
perfect:

- if you **have** the disease it tests positive $99\%$ of the time:
  $P(+\mid D) = 0.99$ (the *sensitivity*);
- if you are **healthy** it still falsely tests positive $5\%$ of the time:
  $P(+\mid \neg D) = 0.05$ (the *false-positive rate*).

**Question:** you test positive. What is $P(D \mid +)$, the chance you actually
have the disease? Most people guess "about 99%". Bayes says otherwise.
"""))
a.append(code("""
# Prior and likelihoods
P_D     = 0.01     # P(disease)
P_pos_D = 0.99     # P(+ | disease)
P_pos_nD = 0.05    # P(+ | no disease)

# Total probability of a positive test:
#   P(+) = P(+|D)P(D) + P(+|~D)P(~D)
P_pos = P_pos_D * P_D + P_pos_nD * (1 - P_D)

# Bayes' rule
P_D_given_pos = (P_pos_D * P_D) / P_pos
print("P(disease | positive test) =", round(P_D_given_pos, 4))
"""))

a.append(md(r"""
Only about **16.7%**! Because the disease is rare, the *many* false positives
from healthy people outnumber the true positives. Let us **verify by
simulation**: create a big population, give each person a true disease status,
run the noisy test, then look only at those who tested positive.
"""))
a.append(code("""
N = 1_000_000

# Step 1: who actually has the disease? True with probability 0.01.
has_disease = rng.random(N) < P_D          # rng.random gives uniform [0,1)

# Step 2: run the test. Sick people: positive w.p. 0.99; healthy: w.p. 0.05.
test_prob = np.where(has_disease, P_pos_D, P_pos_nD)
tested_positive = rng.random(N) < test_prob

# Step 3: among the positives, what fraction truly have the disease?
positives = np.count_nonzero(tested_positive)
true_and_positive = np.count_nonzero(has_disease & tested_positive)
print("simulated P(disease | positive):", round(true_and_positive / positives, 4))
print("Bayes formula gave            :", round(P_D_given_pos, 4))
"""))

a.append(md(r"""
The simulation matches the formula. This counter-intuitive result — a positive
test on a rare condition is often a false alarm — is one of the most important
lessons in all of applied probability.
"""))

# ---- Homework ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Simulate rolling **three** dice $200{,}000$ times and estimate the
   probability that all three show the same number. Compare with the exact value
   $6/216$.
2. Estimate $\mathbb{E}[X]$ and $\operatorname{Var}(X)$ where $X$ is the
   **sum** of two dice, both by simulation and from the definition
   $\sum_k k\,P(X=k)$.
3. Repeat the medical-test simulation but with a **rarer** disease,
   $P(D) = 0.001$. Does $P(D \mid +)$ go up or down? Explain why.
4. Two events $A$ = "first die even" and $B$ = "sum of two dice is 7". Estimate
   $P(A)$, $P(B)$ and $P(A\cap B)$ by simulation and decide whether $A$ and $B$
   are independent.
"""))

save(os.path.join(CH, "08a_probability.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 08b — Statistics & Distributions
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 08b — Statistics & Distributions

In 08a we estimated probabilities by simulation. Now we meet the **named
distributions** that describe the most common kinds of randomness, learn the
language of **PMF / PDF / CDF**, summarise data with **mean, median, std**,
measure how two quantities move together with **covariance and correlation**,
and finish with the spectacular **Central Limit Theorem**.

As before we use one reproducible random generator.
"""))

b.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # reproducible randomness
"""))

b.append(md(r"""
## 1. Discrete vs continuous; PMF, PDF, CDF

A random variable is **discrete** if it takes separate values (die faces, counts)
and **continuous** if it can take any value in a range (heights, errors).

- **PMF** (probability *mass* function), for discrete $X$: $p(k) = P(X = k)$.
- **PDF** (probability *density* function), for continuous $X$: a curve $f(x)$
  whose **area** over an interval gives the probability of landing there.
- **CDF** (cumulative distribution function), for *both*:
  $$F(x) = P(X \le x).$$

Key idea: for a continuous variable, $P(X = x) = 0$; only **intervals** have
positive probability, and that probability is the **area under the PDF**.
"""))

b.append(md(r"""
## 2. The uniform distribution

The **continuous uniform** on $[0, 1]$ is "equally likely anywhere": its PDF is
flat, $f(x) = 1$ for $x \in [0,1]$. `rng.random` samples from it.
"""))
b.append(code("""
U = rng.random(10_000)        # 10,000 samples from Uniform(0, 1)

plt.figure(figsize=(7, 4))
# density=True scales the histogram so its total area is 1 -> comparable to a PDF
plt.hist(U, bins=20, density=True, edgecolor="black", alpha=0.7,
         label="histogram of samples")
plt.axhline(1.0, color="red", linewidth=2, label="true PDF  f(x) = 1")
plt.title("Uniform(0, 1): flat density")
plt.xlabel("x"); plt.ylabel("density"); plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
## 3. Bernoulli and binomial

A **Bernoulli($p$)** variable is a single yes/no trial: $1$ with probability $p$,
$0$ otherwise (a biased coin). Add up $n$ independent Bernoulli trials and you
get a **Binomial($n, p$)** variable — the number of successes. Its PMF is

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}.$$
"""))
b.append(code("""
p = 0.3
# Bernoulli: True/False with prob p, converted to 1/0 with astype(int)
bernoulli = (rng.random(15) < p).astype(int)
print("Bernoulli(0.3) samples:", bernoulli)

# Binomial: count successes in n=10 trials, repeated many times
n = 10
samples = rng.binomial(n, p, size=10_000)   # each value is in 0..10
print("a few Binomial(10, 0.3) samples:", samples[:15])
"""))

b.append(md(r"""
Let's overlay the simulated frequencies with the exact binomial PMF.
"""))
b.append(code("""
from math import comb     # comb(n, k) = n-choose-k

n, p = 10, 0.3
samples = rng.binomial(n, p, size=50_000)

ks = np.arange(0, n + 1)
# Exact PMF from the formula, for each k = 0..n
pmf = np.array([comb(n, k) * p**k * (1 - p)**(n - k) for k in ks])
# Estimated PMF: fraction of samples equal to each k
est = np.array([np.mean(samples == k) for k in ks])

plt.figure(figsize=(7, 4))
plt.bar(ks - 0.15, est, width=0.3, label="simulated", color="steelblue")
plt.bar(ks + 0.15, pmf, width=0.3, label="exact PMF", color="orange")
plt.title("Binomial(10, 0.3): simulation vs theory")
plt.xlabel("number of successes k"); plt.ylabel("P(X = k)")
plt.legend(); plt.grid(True, axis="y")
plt.show()
"""))

b.append(md(r"""
## 4. The normal (Gaussian) distribution

The **normal** distribution $\mathcal{N}(\mu, \sigma^2)$ is the famous bell
curve, with mean $\mu$ and standard deviation $\sigma$. Its PDF is

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).$$

We sample with `rng.normal` and overlay the histogram on the exact density.
"""))
b.append(code("""
mu, sigma = 0.0, 1.0
data = rng.normal(mu, sigma, size=20_000)     # standard normal samples

# Grid of x values for the theoretical curve
x = np.linspace(-4, 4, 400)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

plt.figure(figsize=(7, 4))
plt.hist(data, bins=40, density=True, alpha=0.6, edgecolor="black",
         label="histogram of samples")
plt.plot(x, pdf, "r-", linewidth=2, label="exact PDF")
plt.title("Standard normal N(0, 1)")
plt.xlabel("x"); plt.ylabel("density"); plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
### The CDF

The CDF $F(x) = P(X \le x)$ climbs from $0$ to $1$. We can estimate it from
samples by sorting them: the fraction of data $\le x$ is the **empirical CDF**.
"""))
b.append(code("""
data = rng.normal(0, 1, size=2000)
xs = np.sort(data)                       # sorted samples
emp_cdf = np.arange(1, len(xs) + 1) / len(xs)   # 1/N, 2/N, ..., 1

plt.figure(figsize=(7, 4))
plt.plot(xs, emp_cdf, label="empirical CDF")
plt.title("CDF of N(0,1): probability of landing at or below x")
plt.xlabel("x"); plt.ylabel("F(x) = P(X <= x)"); plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
## 5. Summary statistics: mean, median, std

NumPy gives us the standard summaries directly:

- `np.mean`  — the average $\bar{x} = \frac1N\sum_i x_i$;
- `np.median` — the middle value (robust to outliers);
- `np.std`   — the standard deviation, the typical distance from the mean.
"""))
b.append(code("""
data = rng.normal(10, 2, size=10_000)    # mean ~10, std ~2

print("mean   :", np.mean(data))
print("median :", np.median(data))
print("std    :", np.std(data))
print("min/max:", np.min(data), np.max(data))
"""))

# ---- Exercise 1 (with solution) ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Draw $50{,}000$ samples from a normal with mean $\mu = 5$ and standard deviation
$\sigma = 3$. Report the **sample** mean and std, and check they are close to
$5$ and $3$. Then plot a density histogram with the true PDF overlaid.
"""))
b.append(md("**Solution:**"))
b.append(code("""
mu, sigma = 5.0, 3.0
data = rng.normal(mu, sigma, size=50_000)
print("sample mean:", np.mean(data), " sample std:", np.std(data))

x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
pdf = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

plt.figure(figsize=(7, 4))
plt.hist(data, bins=40, density=True, alpha=0.6, edgecolor="black",
         label="samples")
plt.plot(x, pdf, "r-", linewidth=2, label="true PDF")
plt.title("N(5, 9): samples vs density"); plt.xlabel("x"); plt.ylabel("density")
plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
## 6. Covariance and correlation

For *two* variables we ask: do they move **together**? The **covariance** is

$$\operatorname{Cov}(X, Y) = \mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big],$$

positive when they rise together, negative when one rises as the other falls.
Its scale-free version is the **correlation coefficient**
$\rho = \operatorname{Cov}(X,Y) / (\sigma_X \sigma_Y) \in [-1, 1]$.

We'll build $Y$ as "$X$ plus noise" so the two are genuinely related.
"""))
b.append(code("""
N = 2000
X = rng.normal(0, 1, size=N)
Y = 0.8 * X + rng.normal(0, 0.5, size=N)   # Y depends on X, plus noise

# np.cov / np.corrcoef return 2x2 matrices; the off-diagonal is what we want.
cov_matrix  = np.cov(X, Y)
corr_matrix = np.corrcoef(X, Y)
print("covariance  Cov(X, Y) :", cov_matrix[0, 1])
print("correlation rho(X, Y) :", corr_matrix[0, 1])

plt.figure(figsize=(6, 6))
plt.scatter(X, Y, s=8, alpha=0.4)
plt.title(f"Positively correlated data (rho = {corr_matrix[0,1]:.2f})")
plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
plt.show()
"""))

# ---- Exercise 2 (with solution) ----
b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Construct two variables that are **negatively** correlated, e.g. $Y = -X + $
noise, and confirm the correlation coefficient is close to $-1$. Make a scatter
plot.
"""))
b.append(md("**Solution:**"))
b.append(code("""
N = 2000
X = rng.normal(0, 1, size=N)
Y = -X + rng.normal(0, 0.3, size=N)        # negative relationship + small noise

rho = np.corrcoef(X, Y)[0, 1]
print("correlation rho(X, Y):", rho)       # should be close to -1

plt.figure(figsize=(6, 6))
plt.scatter(X, Y, s=8, alpha=0.4, color="darkred")
plt.title(f"Negatively correlated data (rho = {rho:.2f})")
plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
## 7. The Central Limit Theorem

Here is one of the deepest facts in probability. Take **any** distribution with
a finite mean and variance — it need not look normal at all. Draw a sample of
size $n$ and compute its **mean**. Repeat many times. The **Central Limit
Theorem** (CLT) says: the distribution of these sample means becomes
approximately **normal** as $n$ grows, *regardless of the original shape*.

Let us start from a deliberately non-normal variable: the uniform distribution
on $[0,1]$ (a flat box), and watch its sample means turn into a bell curve.
"""))
b.append(code("""
# A single uniform variable is flat -- definitely not bell-shaped:
flat = rng.random(20_000)
plt.figure(figsize=(7, 3.5))
plt.hist(flat, bins=30, density=True, edgecolor="black", alpha=0.7)
plt.title("Original variable: Uniform(0,1) -- flat, not normal")
plt.xlabel("value"); plt.ylabel("density"); plt.grid(True)
plt.show()
"""))
b.append(code("""
# Now take the mean of n=30 uniform samples, 10,000 separate times.
n = 30
num_experiments = 10_000

# Make a (num_experiments x n) array; each ROW is one experiment of n draws.
batches = rng.random((num_experiments, n))
sample_means = batches.mean(axis=1)        # average across each row

# Overlay the normal predicted by the CLT.
# Uniform(0,1) has mean 1/2 and variance 1/12, so the mean of n of them
# has mean 1/2 and std sqrt(1/12 / n).
mu = 0.5
sigma = np.sqrt((1/12) / n)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
pdf = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2*sigma**2))

plt.figure(figsize=(7, 4))
plt.hist(sample_means, bins=40, density=True, alpha=0.6, edgecolor="black",
         label="distribution of sample means")
plt.plot(x, pdf, "r-", linewidth=2, label="normal predicted by the CLT")
plt.title(f"Central Limit Theorem: means of {n} uniforms look normal")
plt.xlabel("sample mean"); plt.ylabel("density"); plt.legend(); plt.grid(True)
plt.show()
"""))

b.append(md(r"""
The flat box became a bell curve! This is why the normal distribution appears
everywhere in statistics and machine learning: anything that is a **sum or
average of many small independent effects** tends to be approximately normal.
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Sample $30{,}000$ values from `rng.binomial(20, 0.5, size=...)`. Plot the
   density histogram and overlay a normal with the same mean and std — the
   binomial is close to normal when $n$ is large.
2. Demonstrate the CLT starting from a **different** non-normal variable, for
   example the exponential distribution `rng.exponential(1.0, size=...)`. Plot
   the histogram of sample means for $n = 50$.
3. Generate $X \sim \mathcal{N}(0,1)$ and a truly **independent**
   $Y \sim \mathcal{N}(0,1)$. Estimate their correlation and explain why it is
   near $0$. Make a scatter plot.
4. For data from `rng.normal(0, 1, size=10000)`, estimate
   $P(-1 \le X \le 1)$ by counting, and compare with the well-known value
   $\approx 0.68$ (the "68% within one standard deviation" rule).
"""))

save(os.path.join(CH, "08b_statistics_distributions.ipynb"), b)
