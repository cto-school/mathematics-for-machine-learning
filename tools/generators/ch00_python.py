"""Generator for Chapter 00 — Python & Jupyter Crash Course.

Run from anywhere:  python tools/generators/ch00_python.py
Produces two notebooks in 00-python-jupyter-crash-course/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "00-python-jupyter-crash-course")


# ---------------------------------------------------------------------------
# Notebook 00a — Python basics
# ---------------------------------------------------------------------------
a = []

a.append(md(r"""
# Chapter 00a — Python Basics (for Mathematicians)

Welcome! This notebook assumes **no programming background**. Run each code cell
with **Shift + Enter** and watch what happens. Change things and re-run — you
cannot break anything.

Think of Python as a precise assistant that does exactly what you write.
"""))

a.append(md(r"""
## 1. Variables = labeled boxes

A variable gives a name to a value. The `=` sign means *"store the value on the
right into the name on the left"* — it is **assignment**, not equality.
"""))
a.append(code("""
x = 3          # store the integer 3 in a box named x
y = 4
print(x + y)   # print = show the result
"""))

a.append(md(r"""
The last value in a cell is shown automatically, so `print` is optional:
"""))
a.append(code("""
x * y          # Jupyter displays the result of the final line
"""))

a.append(md(r"""
## 2. Numbers and arithmetic

| You write | Meaning |
|-----------|---------|
| `+ - * /` | add, subtract, multiply, divide |
| `**`      | power: `2**3` is $2^3 = 8$ |
| `//`      | floor division: `7 // 2` is `3` |
| `%`       | remainder (modulo): `7 % 2` is `1` |
"""))
a.append(code("""
print(2 ** 10)    # 2 to the 10th power
print(7 / 2)      # true division -> 3.5
print(7 // 2)     # floor division -> 3
print(7 % 2)      # remainder      -> 1
"""))

a.append(md(r"""
## 3. Types

Every value has a **type**. The main ones:

- `int`  — whole numbers, e.g. `5`
- `float`— decimals, e.g. `5.0`, `3.14`
- `str`  — text ("strings"), e.g. `"hello"`
- `bool` — `True` or `False`
"""))
a.append(code("""
print(type(5))        # <class 'int'>
print(type(5.0))      # <class 'float'>
print(type("hello"))  # <class 'str'>
print(type(3 > 2))    # <class 'bool'>
"""))

a.append(md(r"""
## 4. Comparisons and booleans

`==` tests equality (note: **two** equals signs). Comparisons return a boolean.
"""))
a.append(code("""
print(3 == 3)     # True
print(3 == 4)     # False
print(3 < 4)      # True
print(3 != 4)     # not equal -> True
"""))

a.append(md(r"""
## 5. Lists — ordered collections

A list holds many values in order. **Indexing starts at 0.**
"""))
a.append(code("""
primes = [2, 3, 5, 7, 11]
print(primes[0])    # first element -> 2
print(primes[1])    # second        -> 3
print(primes[-1])   # last          -> 11
print(len(primes))  # how many      -> 5
"""))
a.append(code("""
# "Slicing" takes a sub-list: start:stop (stop is NOT included)
print(primes[0:3])  # elements 0,1,2 -> [2, 3, 5]
print(primes[2:])   # from index 2 to the end -> [5, 7, 11]
"""))

a.append(md(r"""
## 6. Loops — repeat work

`range(n)` produces `0, 1, ..., n-1`. A `for` loop walks through values.
The indented line(s) are the "body" that repeats.
"""))
a.append(code("""
for k in range(5):
    print(k, k**2)   # print each number and its square
"""))
a.append(code("""
# Loop directly over a list's items
for p in primes:
    print("prime:", p)
"""))

a.append(md(r"""
## 7. Conditionals — make decisions
"""))
a.append(code("""
for n in range(1, 7):
    if n % 2 == 0:
        print(n, "is even")
    else:
        print(n, "is odd")
"""))

a.append(md(r"""
## 8. Combining conditions: `and`, `or`, `not`

Conditions combine with the plain words `and`, `or`, `not` — exactly like logic
on paper. `and` is True only when *both* sides are; `or` is True when *either*
is; `not` flips a boolean.
"""))
a.append(code("""
x = 7
print(x > 0 and x < 10)   # True  -> both hold
print(x < 0 or x == 7)    # True  -> the second holds
print(not (x == 7))       # False -> flips True to False
print(0 <= x <= 10)       # True  -> Python allows chained comparisons
"""))

a.append(md(r"""
## 9. Functions — package a computation

A Python function is just like a math function. `def` defines it; `return`
gives back the result.

$$f(x) = x^2 + 1$$
"""))
a.append(code("""
def f(x):
    return x**2 + 1

print(f(0))   # 1
print(f(3))   # 10
"""))
a.append(code("""
# Functions can take several inputs
def line(x, m, b):
    return m * x + b      # y = m x + b

print(line(2, m=3, b=1))  # 3*2 + 1 = 7
"""))

a.append(md(r"""
## 10. Tuples and dictionaries

A **tuple** is like a list but written with parentheses and cannot be changed
("immutable"). NumPy shapes are tuples, e.g. `(2, 3)`.

A **dictionary** maps **keys** to **values** — a lookup table, much like a
function given by a table of inputs and outputs.
"""))
a.append(code("""
point = (3, 4)              # a tuple
print(point[0], point[1])  # 3 4
x, y = point               # 'unpacking' a tuple into two names
print("x =", x, " y =", y)

grades = {"Alice": 90, "Bob": 82}   # dictionary: name -> score
print(grades["Alice"])     # look up a value by its key -> 90
grades["Cara"] = 75        # add a new key/value pair
for name, score in grades.items():
    print(name, "scored", score)
"""))

a.append(md(r"""
## 11. When things go wrong: reading errors

Everyone's code breaks constantly — that is completely normal. Python prints a
*traceback*; **read the last line first**. The three errors you'll meet on day
one:

- **NameError** — a name doesn't exist (a typo, or you forgot to run the cell
  that defines it).
- **SyntaxError / IndentationError** — the code's structure is off (a missing
  `:` or wrong indentation).
- **TypeError** — incompatible things combined, e.g. a number plus text.

The cell below triggers two errors on purpose (caught with `try/except` so the
notebook keeps running) and prints the message.
"""))
a.append(code("""
# 1) NameError: 'total' was never defined
try:
    print(total)
except NameError as e:
    print("NameError:", e)

# 2) TypeError: you cannot add a number and a string
try:
    print(3 + "apple")
except TypeError as e:
    print("TypeError:", e)
"""))
a.append(md(r"""
In a real cell the error *stops* execution — fix the line named at the bottom of
the traceback and re-run. **Tip:** if results look stale or inconsistent, use
**Kernel ▸ Restart & Run All** to run everything fresh from the top.
"""))

# ---- Exercise (with solution) ----
a.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Write a function `factorial(n)` that returns $n! = 1\cdot 2 \cdots n$ using a
loop. Test it on `factorial(5)` (should be 120).
"""))
a.append(md("**Solution:**"))
a.append(code("""
def factorial(n):
    result = 1
    for k in range(1, n + 1):   # 1, 2, ..., n
        result = result * k
    return result

print(factorial(5))   # 120
print(factorial(0))   # 1  (empty product)
"""))

a.append(md(r"""
## ✍️ Exercise 2 (solution included)

Use a loop to compute the sum $\sum_{k=1}^{100} k$. (You can check it against
the famous formula $n(n+1)/2$.)
"""))
a.append(md("**Solution:**"))
a.append(code("""
total = 0
for k in range(1, 101):
    total = total + k
print(total)              # 5050
print(100 * 101 // 2)     # 5050  (the formula agrees)
"""))

# ---- Homework (no solutions) ----
a.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Write `is_prime(n)` that returns `True` if `n` is prime, else `False`.
   *Hint:* test divisibility by every `d` from 2 up to `n-1`.
2. Using `is_prime`, build a list of all primes below 50.
3. Write `fib(n)` that returns the `n`-th Fibonacci number
   ($0,1,1,2,3,5,\dots$) using a loop.
4. Write a function `mean(values)` that returns the arithmetic mean of a list
   of numbers, without using any library.
"""))

save(os.path.join(CH, "00a_python_basics.ipynb"), a)


# ---------------------------------------------------------------------------
# Notebook 00b — Python for mathematics
# ---------------------------------------------------------------------------
b = []

b.append(md(r"""
# Chapter 00b — Python for Mathematics

Now we use Python the way a mathematician would: evaluate formulas, build
tables, compute sums and products, and draw a first graph.
"""))

b.append(md(r"""
## 1. The `math` module

A *module* is a toolbox you import. `math` holds the standard functions.
"""))
b.append(code("""
import math

print(math.pi)          # 3.14159...
print(math.sqrt(2))     # square root of 2
print(math.exp(1))      # e^1 = e
print(math.log(math.e)) # natural log -> 1.0
print(math.sin(math.pi / 2))  # 1.0
"""))

b.append(md(r"""
## 2. A formula becomes a function, a table becomes a loop

Let $g(x) = \sqrt{x}\,e^{-x}$. We tabulate it on a grid of $x$ values.
"""))
b.append(code("""
import math

def g(x):
    return math.sqrt(x) * math.exp(-x)

for x in [0, 0.5, 1, 1.5, 2, 3]:
    print(f"g({x}) = {g(x):.4f}")   # :.4f shows 4 decimal places
"""))

b.append(md(r"""
## 3. Summation $\sum$ and product $\prod$ as loops

$$S = \sum_{k=1}^{n} \frac{1}{k^2}, \qquad P = \prod_{k=1}^{n} \frac{k}{k+1}$$
"""))
b.append(code("""
n = 1000

S = 0.0
for k in range(1, n + 1):
    S += 1 / k**2
print("sum  ~", S, " (approaches pi^2/6 =", math.pi**2 / 6, ")")

P = 1.0
for k in range(1, n + 1):
    P *= k / (k + 1)
print("prod ~", P, " (this telescopes to 1/(n+1))")
"""))

b.append(md(r"""
Python also has built-in `sum()` and "comprehensions" — a compact loop:
"""))
b.append(code("""
S = sum(1 / k**2 for k in range(1, 1001))   # same result, one line
print(S)
"""))

b.append(md(r"""
## 4. A first picture: plotting a function

We'll meet Matplotlib properly in Chapter 02. Here is a taste: plot
$f(x) = \sin(x)$ on $[0, 2\pi]$.
"""))
b.append(code("""
import math
import matplotlib.pyplot as plt

# build lists of (x, y) points
xs = [i * 0.05 for i in range(0, 126)]   # 0, 0.05, ..., 6.25
ys = [math.sin(x) for x in xs]

plt.plot(xs, ys)
plt.title("y = sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
"""))

b.append(md(r"""
Notice how clumsy it is to build those lists by hand. **This is exactly the
problem NumPy solves** in the next chapter — we'll write `x = np.linspace(0,
2*np.pi, 200)` and `y = np.sin(x)` and be done.
"""))

# ---- Exercises with solutions ----
b.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Approximate $e$ using the series $e = \sum_{k=0}^{n} \dfrac{1}{k!}$ for
`n = 10`. Compare with `math.e`.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import math
approx = sum(1 / math.factorial(k) for k in range(0, 11))
print("series  :", approx)
print("math.e  :", math.e)
print("error   :", abs(approx - math.e))
"""))

b.append(md(r"""
## ✍️ Exercise 2 (solution included)

Tabulate $h(x) = \dfrac{1}{1 + e^{-x}}$ (the *sigmoid*, which we'll use a lot
later) for $x \in \{-3,-2,\dots,3\}$.
"""))
b.append(md("**Solution:**"))
b.append(code("""
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

for x in range(-3, 4):
    print(f"sigmoid({x:2d}) = {sigmoid(x):.4f}")
"""))

# ---- Homework ----
b.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. Approximate $\pi$ with the Leibniz series
   $\pi = 4\sum_{k=0}^{n} \dfrac{(-1)^k}{2k+1}$. How large must `n` be to get
   3 correct decimals?
2. Plot $f(x) = e^{-x^2}$ (the bell curve shape) on $[-3, 3]$ using the
   list-based approach above.
3. Write `deriv(f, x, h=1e-5)` that estimates $f'(x)$ with the difference
   quotient $\frac{f(x+h)-f(x)}{h}$, and test it on $f(x)=x^2$ at $x=3$
   (should be near 6).
4. Compute the harmonic number $H_n=\sum_{k=1}^n 1/k$ for $n=1,10,100,1000$ and
   observe how slowly it grows.
"""))

save(os.path.join(CH, "00b_python_for_math.ipynb"), b)
