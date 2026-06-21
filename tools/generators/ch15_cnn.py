"""Generator for Chapter 15 — Introduction to Convolutional Neural Networks.

Run from anywhere:  python tools/generators/ch15_cnn.py
Produces one notebook in 15-convolutional-neural-networks/.

This chapter is CONCEPTUAL + FORWARD PASS ONLY: we build the pieces of a CNN
(convolution, padding/stride, ReLU, max pooling) by hand in NumPy and run a
single forward pass through a mini-pipeline. Training the conv layer (backprop
through convolution) is deliberately out of scope — we point back to the MLP
backprop in Chapter 13.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from nbutil import md, code, save  # noqa: E402

CH = os.path.join(ROOT, "15-convolutional-neural-networks")


# ===========================================================================
# Notebook 15 — CNNs from scratch (forward pass only)
# ===========================================================================
nb = []

nb.append(md(r"""
# Chapter 15 — Introduction to Convolutional Neural Networks

In **Chapter 13** you built a neural network from scratch: a stack of fully
connected layers, trained by gradient descent and backpropagation. That network
treats its input as a flat list of numbers. For *images*, that turns out to be
wasteful and a little naive.

This chapter is the **conceptual bridge to deep learning**: it introduces the
**convolutional neural network (CNN)**, the architecture behind almost all
modern image recognition. We will build every piece by hand in NumPy and run a
**forward pass** through a small CNN, watching the shapes transform.

What we will do:

1. see *why* fully connected nets are wasteful for images,
2. define and implement **convolution** (really *cross-correlation*) from
   scratch,
3. apply classic kernels — **edge detection, blur, sharpen** — to a small image,
4. understand **padding** and **stride** and the output-size formula,
5. apply **ReLU** (reused from Ch. 13) to a feature map,
6. implement **max pooling** from scratch,
7. assemble the full pipeline
   `[conv -> ReLU -> pool] -> flatten -> fully-connected -> softmax`
   and run one forward pass end to end.

> **Scope.** This is a *forward-pass* tour. We do **not** train the convolution
> here — backpropagation through a conv layer is the same chain-rule idea from
> Chapter 13, just with **weight sharing**, and we point to that at the end.
"""))

nb.append(md(r"""
## 1. Setup

Pure NumPy and Matplotlib, with a fixed random generator
`np.random.default_rng(0)` so every run is identical and reproducible — exactly
as in Chapter 13.
"""))
nb.append(code("""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)   # one reproducible random generator
"""))

# ---------------------------------------------------------------------------
# 2. Motivation
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 2. Why fully connected nets are wasteful for images

Recall the MLP from Chapter 13: every input feature connects to every neuron in
the first hidden layer via a weight matrix $W_1$. That is fine for $2$ inputs.
But an image is a *grid* of pixels, and the numbers explode.

Take a tiny $28 \times 28$ grayscale image (the size of an MNIST digit). To feed
it to an MLP you must **flatten** it into a vector of $28 \times 28 = 784$
numbers. If the first hidden layer has just $100$ neurons, the weight matrix
$W_1$ has

$$784 \times 100 = 78{,}400 \text{ weights} \quad\text{— for one small layer.}$$

A modest color photo of $200 \times 200 \times 3$ flattens to $120{,}000$
inputs; the same hidden layer would need $12{,}000{,}000$ weights. Let's just
print these numbers.
"""))
nb.append(code("""
def mlp_first_layer_params(height, width, channels, hidden):
    n_inputs = height * width * channels      # pixels become a flat vector
    return n_inputs * hidden                  # weights in W1 (ignoring biases)

print("28x28 grayscale -> 100 neurons :", mlp_first_layer_params(28, 28, 1, 100), "weights")
print("200x200 color   -> 100 neurons :", mlp_first_layer_params(200, 200, 3, 100), "weights")
"""))

nb.append(md(r"""
Two problems, not just one:

- **Parameter explosion.** The weight count grows with the number of pixels.
  Large images become impossible to train this way.
- **Ignoring spatial structure.** Flattening throws away *geometry*. Pixel
  $(0,0)$ and pixel $(0,1)$ are neighbours, but after flattening the network has
  no idea — it sees $784$ unrelated numbers. An edge, a corner, or a stroke is a
  *local* pattern, and the MLP cannot exploit that.
- **No translation tolerance.** If the same digit shifts a few pixels to the
  right, every input value changes, and the MLP must re-learn the pattern in its
  new location from scratch.

**The CNN fix.** Instead of one giant weight matrix, use a *small* set of
weights — a **kernel** — and **slide** it across the image, reusing the *same*
weights at every position. This is **weight sharing**. A $3 \times 3$ kernel has
only $9$ weights, yet it can detect its pattern *anywhere* in the image. That
single idea solves all three problems at once.
"""))

# ---------------------------------------------------------------------------
# 3. Build a small image
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 3. A hand-made image (no downloads needed)

We don't need a dataset. Let's build a small $9 \times 9$ grayscale image by
hand: the **letter T** drawn on a grid. A pixel is just a number — here $1.0$ is
white (the letter) and $0.0$ is black (the background).
"""))
nb.append(code("""
# A 9x9 grid. We start all-black (zeros) and paint a white "T".
img = np.zeros((9, 9))
img[1, 1:8] = 1.0        # the horizontal bar across the top
img[1:8, 4] = 1.0        # the vertical stem down the middle
print("image shape:", img.shape)
print(img)
"""))
nb.append(code("""
# Show it. 'gray' maps 0 -> black, 1 -> white. A colorbar shows the scale.
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="gray")
plt.title("hand-made image: the letter T")
plt.colorbar(label="pixel value")
plt.show()
"""))

# ---------------------------------------------------------------------------
# 4. What a convolution IS
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 4. What a convolution really is

A **convolution** (in deep learning we actually compute the closely related
**cross-correlation**, but everyone calls it convolution) slides a small grid
of weights — the **kernel** — across the image. At each position it lays the
kernel over a patch of the image, multiplies overlapping numbers, and **sums**
them into a single output number. That output number says *how strongly this
patch matches the kernel's pattern*.

If the image is $H \times W$ and the kernel is $k \times k$, then for output
position $(i, j)$ the value is

$$
\text{out}[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}
\text{image}[i+u,\; j+v]\; \cdot\; \text{kernel}[u, v].
$$

That inner double sum is just a **dot product** between the kernel and the
image patch beneath it (Chapter 03) — the very same operation a neuron does,
but reused at every location.

We use **valid** padding: the kernel only sits where it *fully* overlaps the
image, so the output is slightly smaller than the input.
"""))
nb.append(code("""
def conv2d(image, kernel):
    \"\"\"2D cross-correlation with 'valid' padding, written with plain loops.

    image  : 2D array, shape (H, W)
    kernel : 2D array, shape (k, k)
    returns: 2D array, shape (H - k + 1, W - k + 1)
    \"\"\"
    H, W = image.shape
    k, _ = kernel.shape

    out_h = H - k + 1            # output rows (valid padding)
    out_w = W - k + 1            # output cols
    out = np.zeros((out_h, out_w))

    for i in range(out_h):                 # slide top-to-bottom
        for j in range(out_w):             # slide left-to-right
            patch = image[i:i+k, j:j+k]    # the k x k window under the kernel
            out[i, j] = np.sum(patch * kernel)   # elementwise multiply, then sum
    return out
"""))
nb.append(code("""
# A tiny sanity check by hand. The kernel below just copies the center pixel
# of each 3x3 patch (a 1 in the middle, zeros elsewhere), so conv2d returns the
# interior of the image unchanged.
identity_kernel = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])
out = conv2d(img, identity_kernel)
print("input shape :", img.shape)        # (9, 9)
print("output shape:", out.shape)        # (7, 7) = 9 - 3 + 1
print("matches interior of image:", np.allclose(out, img[1:8, 1:8]))
"""))

# ---------------------------------------------------------------------------
# 5. Classic kernels
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 5. Classic kernels: edges, blur, sharpen

The magic of convolution is that *different kernels detect different patterns*.
Here are four hand-designed classics. (Real CNNs **learn** kernels like these
from data — more on that at the end.)

- **Sobel-x** responds to *vertical* edges (horizontal change in brightness).
- **Sobel-y** responds to *horizontal* edges (vertical change in brightness).
- **Blur** (a $3\times 3$ averaging kernel) smooths the image.
- **Sharpen** boosts the center pixel relative to its neighbours.

$$
\text{Sobel-x}=\begin{bmatrix} -1 & 0 & 1\\ -2 & 0 & 2\\ -1 & 0 & 1\end{bmatrix},\quad
\text{Sobel-y}=\begin{bmatrix} -1 & -2 & -1\\ 0 & 0 & 0\\ 1 & 2 & 1\end{bmatrix},\quad
\text{blur}=\tfrac19\begin{bmatrix} 1 & 1 & 1\\ 1 & 1 & 1\\ 1 & 1 & 1\end{bmatrix}.
$$
"""))
nb.append(code("""
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=float)

blur = np.ones((3, 3)) / 9.0          # averaging kernel: every weight = 1/9

sharpen = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=float)
"""))
nb.append(code("""
# Apply each kernel to our T image to get four "feature maps".
maps = {
    "input":   img,
    "Sobel-x": conv2d(img, sobel_x),
    "Sobel-y": conv2d(img, sobel_y),
    "blur":    conv2d(img, blur),
    "sharpen": conv2d(img, sharpen),
}

fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, (name, m) in zip(axes, maps.items()):
    im = ax.imshow(m, cmap="gray")
    ax.set_title(name)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle("One image, four kernels -> four feature maps")
plt.show()
"""))

nb.append(md(r"""
Look at the results. **Sobel-x** lights up along the *vertical* stem of the T
(where brightness changes horizontally), while **Sobel-y** lights up along the
*horizontal* bar. **Blur** spreads the white into a soft smear; **sharpen**
exaggerates the strokes. A single kernel is a tiny pattern detector, and a CNN
uses *many* of them in parallel.
"""))

# ---- Exercise 1 (with solution) ----
nb.append(md(r"""
---
## ✍️ Exercise 1 (solution included)

Build a **vertical-line detector** kernel and confirm it responds most strongly
along the T's vertical stem. *Hint:* a column of positive weights flanked by
negative columns, e.g.
$\begin{bmatrix} -1 & 2 & -1\\ -1 & 2 & -1\\ -1 & 2 & -1\end{bmatrix}$,
fires when there is a bright vertical streak in the middle of the patch.
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
vline = np.array([[-1, 2, -1],
                  [-1, 2, -1],
                  [-1, 2, -1]], dtype=float)

resp = conv2d(img, vline)
print("response shape:", resp.shape)

# The largest responses should sit along the vertical stem (column 4 of img,
# which maps to column 3 of the valid-padded output).
ij = np.unravel_index(np.argmax(resp), resp.shape)
print("strongest response at (row, col):", ij)
print("max response value:", resp.max())

plt.figure(figsize=(4, 4))
plt.imshow(resp, cmap="gray")
plt.title("vertical-line detector response")
plt.colorbar()
plt.show()
"""))

# ---------------------------------------------------------------------------
# 6. Padding and stride
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 6. Padding and stride

Two knobs control the geometry of a convolution.

**Padding $p$.** With *valid* padding the output shrinks (a $9\times 9$ image
and a $3\times 3$ kernel give $7\times 7$). To keep the size the same we can pad
the border with $p$ rings of zeros before convolving — *same* padding.

**Stride $s$.** Instead of moving the kernel one pixel at a time ($s=1$), we can
step by $s$ pixels, which *downsamples* the output.

For an input of side $H$, kernel $k$, padding $p$, stride $s$, the output side is

$$
\text{out} = \left\lfloor \frac{H - k + 2p}{s} \right\rfloor + 1.
$$

Let's verify the formula and implement padding + stride.
"""))
nb.append(code("""
def output_size(H, k, p, s):
    return (H - k + 2 * p) // s + 1

# valid (p=0, s=1):  9 -> 7
print("p=0, s=1:", output_size(9, 3, 0, 1))
# same  (p=1, s=1):  9 -> 9
print("p=1, s=1:", output_size(9, 3, 1, 1))
# stride 2 (p=0, s=2): 9 -> 4
print("p=0, s=2:", output_size(9, 3, 0, 2))
"""))
nb.append(code("""
def conv2d_full(image, kernel, padding=0, stride=1):
    \"\"\"Convolution (cross-correlation) with zero-padding and a stride.\"\"\"
    p, s = padding, stride
    if p > 0:
        # pad p rings of zeros on every side
        image = np.pad(image, pad_width=p, mode="constant", constant_values=0)

    H, W = image.shape
    k, _ = kernel.shape
    out_h = (H - k) // s + 1
    out_w = (W - k) // s + 1
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            r, c = i * s, j * s                 # top-left of the window, stepped by s
            patch = image[r:r+k, c:c+k]
            out[i, j] = np.sum(patch * kernel)
    return out
"""))
nb.append(code("""
# Check the implementation against the formula.
print("valid  :", conv2d_full(img, sobel_x, padding=0, stride=1).shape)  # (7, 7)
print("same   :", conv2d_full(img, sobel_x, padding=1, stride=1).shape)  # (9, 9)
print("stride2:", conv2d_full(img, sobel_x, padding=0, stride=2).shape)  # (4, 4)
"""))

# ---------------------------------------------------------------------------
# 7. ReLU on a feature map
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 7. The ReLU step

A convolution is a *linear* operation. Just as in Chapter 13, we follow it with
a nonlinear **activation** so the network can represent curved relationships.
The standard choice in CNNs is **ReLU**:

$$\text{ReLU}(z) = \max(0, z).$$

Applied to a feature map, ReLU simply **clamps every negative number to zero**,
keeping only the positive "this pattern is present" responses. It is the exact
same function from Chapter 13 — here it acts elementwise on a 2D map.
"""))
nb.append(code("""
def relu(z):
    return np.maximum(0.0, z)        # elementwise max with 0 (from Ch. 13)

feature = conv2d(img, sobel_x)       # has both positive and negative values
activated = relu(feature)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
for ax, (name, m) in zip(axes, [("Sobel-x map", feature), ("after ReLU", activated)]):
    im = ax.imshow(m, cmap="gray")
    ax.set_title(name); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.show()

print("min before ReLU:", feature.min(), " min after ReLU:", activated.min())
"""))

# ---------------------------------------------------------------------------
# 8. Max pooling
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 8. Max pooling (downsampling)

After conv + ReLU, a **pooling** layer shrinks the feature map. **Max pooling**
slides a small window (commonly $2\times 2$) over the map and keeps only the
**largest** value in each window.

Why do this?

- **It shrinks the map**, cutting computation for later layers.
- **It adds translation tolerance.** If the strongest response shifts by one
  pixel, the max over a window often stays the same — so the network becomes a
  little invariant to *where exactly* a feature appears.

Here is max pooling from scratch on a tiny example.
"""))
nb.append(code("""
def max_pool(feature_map, size=2, stride=2):
    \"\"\"Non-overlapping max pooling. Keeps the max of each size x size window.\"\"\"
    H, W = feature_map.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = feature_map[r:r+size, c:c+size]
            out[i, j] = np.max(window)        # keep the strongest response
    return out
"""))
nb.append(code("""
# Tiny worked example so you can check the maxima by eye.
tiny = np.array([[1, 3, 2, 4],
                 [5, 6, 1, 2],
                 [0, 1, 8, 3],
                 [2, 1, 4, 7]], dtype=float)
print("input 4x4:\\n", tiny)
print("after 2x2 max pool (-> 2x2):\\n", max_pool(tiny, size=2, stride=2))
# top-left window  [[1,3],[5,6]] -> 6
# top-right window [[2,4],[1,2]] -> 4
# bot-left window  [[0,1],[2,1]] -> 2
# bot-right window [[8,3],[4,7]] -> 8
"""))

# ---- Exercise 2 (with solution) ----
nb.append(md(r"""
---
## ✍️ Exercise 2 (solution included)

Implement **average pooling** `avg_pool(feature_map, size=2, stride=2)` (keep
the *mean* of each window instead of the max) and apply both max and average
pooling to the `tiny` array above. How do the results differ, and why might
*max* pooling be preferred for detecting whether a feature is *present*?
"""))
nb.append(md("**Solution:**"))
nb.append(code("""
def avg_pool(feature_map, size=2, stride=2):
    H, W = feature_map.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = feature_map[r:r+size, c:c+size]
            out[i, j] = np.mean(window)       # average instead of max
    return out

print("max pool:\\n", max_pool(tiny))
print("avg pool:\\n", avg_pool(tiny))
# Max pooling reports the *strongest* response in each region, so it answers
# "is this feature present anywhere here?" loudly. Average pooling dilutes a
# strong single response by mixing in weak neighbours, which can wash out a
# small but important detection. That is why max pooling is the common default.
"""))

# ---------------------------------------------------------------------------
# 9. The full pipeline (forward pass)
# ---------------------------------------------------------------------------
nb.append(md(r"""
## 9. The conceptual CNN pipeline

A real CNN stacks the pieces we just built:

$$
\underbrace{[\,\text{conv} \to \text{ReLU} \to \text{pool}\,]}_{\text{repeat a few times}}
\;\to\; \text{flatten} \;\to\; \text{fully-connected} \;\to\; \text{softmax}.
$$

- The **conv -> ReLU -> pool** block extracts features and shrinks the map.
  Stacking several blocks builds *features of features* (edges -> shapes ->
  objects).
- **Flatten** turns the final 2D map into a 1D vector.
- A **fully-connected** layer (exactly the layer from Chapter 13) maps that
  vector to one score per class.
- **Softmax** turns those scores into probabilities that sum to $1$ (Chapter 11).

Let's run **one forward pass** through a small pipeline and watch the shapes
transform. We use **random kernels** — that is fine for a forward pass; in a
real CNN these kernels are *learned*.
"""))
nb.append(code("""
def softmax(scores):
    # subtract the max for numerical stability, then normalize to sum to 1
    e = np.exp(scores - np.max(scores))
    return e / np.sum(e)
"""))
nb.append(code("""
# ---- a slightly bigger input so the shapes are interesting ----
image = rng.random((16, 16))         # a 16x16 "image" of random pixels
print("input image          :", image.shape)

# ---- block 1: conv -> ReLU -> pool ----
k1 = rng.normal(0, 1, size=(3, 3))   # one random 3x3 kernel
c1 = conv2d(image, k1)               # 16 -> 14   (valid, 3x3)
a1 = relu(c1)                        # same shape, negatives clamped to 0
p1 = max_pool(a1, size=2, stride=2)  # 14 -> 7
print("after conv1 -> ReLU  :", a1.shape)
print("after pool1          :", p1.shape)

# ---- block 2: conv -> ReLU -> pool ----
k2 = rng.normal(0, 1, size=(3, 3))
c2 = conv2d(p1, k2)                  # 7 -> 5
a2 = relu(c2)
p2 = max_pool(a2, size=2, stride=2)  # 5 -> 2
print("after conv2 -> ReLU  :", a2.shape)
print("after pool2          :", p2.shape)
"""))
nb.append(code("""
# ---- flatten -> fully-connected -> softmax ----
flat = p2.flatten()                  # 2x2 map -> length-4 vector
print("flattened            :", flat.shape)

n_classes = 3                        # pretend we classify into 3 categories
W_fc = rng.normal(0, 0.5, size=(flat.size, n_classes))   # the FC weight matrix
b_fc = np.zeros(n_classes)

scores = flat @ W_fc + b_fc          # one raw score per class (Ch. 13 layer)
probs = softmax(scores)              # turn scores into probabilities (Ch. 11)

print("class scores         :", np.round(scores, 3))
print("class probabilities  :", np.round(probs, 3), " (sum =", round(probs.sum(), 3), ")")
print("predicted class      :", int(np.argmax(probs)))
"""))

nb.append(md(r"""
Trace the shapes: the $16\times 16$ image became $14\times 14$, then $7\times 7$
after pooling, then $5\times 5$, then $2\times 2$, then a length-$4$ vector,
finally $3$ class probabilities that sum to $1$. The map **shrinks** as we go
deep while each value summarizes an **ever-larger region** of the original
image. That is the whole forward pass of a CNN.
"""))

nb.append(md(r"""
## 10. What we skipped: training

We used **random** kernels and a **random** fully-connected matrix. The output
is therefore meaningless — but the *plumbing* is exactly right.

Real CNNs **learn** their kernels by **gradient descent**, just like the MLP in
Chapter 13. Each kernel weight is a parameter; we define a loss (e.g.
cross-entropy on the softmax output, Ch. 11) and push gradients **backwards**
through softmax, the fully-connected layer, the pooling, the ReLU, and the
convolution — the same **chain rule** from Chapters 07 and 13.

The one new wrinkle is **weight sharing**: because a kernel is reused at every
position, its gradient is the *sum* of the contributions from all positions it
touched. Conceptually it is still backpropagation; mechanically it is bookkeeping
over all the sliding windows. That derivation is **beyond this introduction** —
revisit the backprop in **Chapter 13b**, and know that a conv layer trains by
*the same idea applied with weight sharing*.

You now understand what a convolutional network *is*: small, shared kernels slid
across an image, nonlinearity, pooling, and a familiar classifier on top.
"""))

# ---- Homework ----
nb.append(md(r"""
---
## 📝 Homework (no solutions provided)

1. **A new letter.** Draw a hand-made $9\times 9$ letter **L** (or **H**) as a
   NumPy array, then run `Sobel-x`, `Sobel-y`, and the `vline` detector on it.
   Which kernel responds most strongly to which stroke, and does it match your
   intuition?
2. **Output-size predictions.** Using only the formula
   $\lfloor (H - k + 2p)/s \rfloor + 1$, predict the output side for
   $(H, k, p, s) = (32, 5, 0, 1),\ (32, 5, 2, 1),\ (28, 3, 1, 2)$. Then confirm
   each with `conv2d_full` on an `rng.random((H, H))` input.
3. **Pooling window.** Apply `max_pool` with `size=3, stride=3` to a
   $9\times 9$ random map and check the output shape against the formula. How
   does a larger pooling window change how much the map shrinks?
4. **Count the parameters.** For the pipeline in Section 9 (two $3\times 3$
   kernels plus the final fully-connected matrix to $3$ classes), count the
   total number of learnable weights. Compare that with an MLP that flattens the
   $16\times 16$ image straight into a $3$-class output layer. Which has fewer
   parameters, and why?
""" ))

save(os.path.join(CH, "15_cnn_from_scratch.ipynb"), nb)
