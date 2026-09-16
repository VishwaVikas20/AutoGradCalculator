# Automatic Gradient Calculator

A from-scratch reverse-mode automatic differentiation engine — no `torch`, no `tensorflow`. Built in two phases: a scalar engine (`Value`) and a matrix engine (`Tensor`), each with a computation graph and a topological-sort backward pass.

## Core Objective

Given a scalar loss $L$, compute

$$\frac{\partial L}{\partial w_i} \quad \text{for all } i \in N$$

for every parameter $w_i$ in the graph.

## Structure

Both `Value` and `Tensor` are graph nodes with the same four attributes:

| Attribute | Meaning |
|---|---|
| `data` | The actual numerical value (`Tensor` stores it as a 2-D float array) |
| `grad` | The accumulated derivative of the final output w.r.t. this node |
| `_prev` | The parent nodes that produced this one |
| `_backward` | A closure stored on the node that applies the chain rule for that specific op |

Each operator does two things: compute the forward value, and attach a `_backward` closure that knows the local derivative.

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), operation="*")
    def backward():
        self.grad  += out.grad * other.data
        other.grad += out.grad * self.data
    out._backward = backward
    return out
```

## Backward Pass

Gradients flow backward via **topological sort** — build the DAG depth-first, seed the loss gradient with 1, then walk the nodes in reverse and fire each `_backward`:

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a}$$

```python
def backward_pass(Loss):
    topo, visited = [], set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(Loss)
    Loss.grad = 1
    for v in reversed(topo):
        v._backward()
```

---

## Phase 1 — `Value` (scalar engine)

### Addition

$$c = a + b \qquad \frac{\partial c}{\partial a} = 1, \quad \frac{\partial c}{\partial b} = 1$$

```python
self.grad  += out.grad
other.grad += out.grad
```

### Multiplication

$$c = a \times b \qquad \frac{\partial c}{\partial a} = b, \quad \frac{\partial c}{\partial b} = a$$

```python
self.grad  += out.grad * other.data
other.grad += out.grad * self.data
```

### Subtraction

$$c = a - b \qquad \frac{\partial c}{\partial a} = 1, \quad \frac{\partial c}{\partial b} = -1$$

```python
self.grad  += out.grad
other.grad -= out.grad
```

### Power

Implemented for a **variable exponent**, so both bases and exponents get gradients:

$$c = a^{b} \qquad \frac{\partial c}{\partial a} = b \cdot a^{b-1}, \quad \frac{\partial c}{\partial b} = a^{b}\ln(a)$$

```python
self.grad  += other.data * out.grad * (self.data ** (other.data - 1))
if self.data > 0:
    other.grad += out.grad * (self.data ** other.data) * math.log(self.data)
```

The `log` branch is guarded because $\ln(a)$ is undefined for $a \le 0$.

### Division

Not a primitive — composed from multiplication and power, so it inherits their gradients for free:

```python
def __truediv__(self, other):
    return self * (other ** -1)
```

### Activation — tanh

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \qquad \frac{d}{dx}\tanh(x) = 1 - \tanh^{2}(x)$$

```python
self.grad += out.grad * (1 - t ** 2)
```

### Activation — ReLU

$$c = \max(0, a) \qquad \frac{\partial c}{\partial a} = \begin{cases} 1 & a > 0 \\ 0 & a \le 0 \end{cases}$$

```python
self.grad += out.grad * v   # v = 1 if t > 0 else 0
```

### Also implemented

`__neg__`, `__radd__`, `__rmul__`, `__rtruediv__` — so `Value` objects interoperate with plain Python numbers on either side (`2 * x` works, not just `x * 2`).

### Verification

Checked against a hand-derived analytic gradient:

$$f = x^{2} + 2xy - y^{3} \qquad \frac{\partial f}{\partial x} = 2x + 2y, \qquad \frac{\partial f}{\partial y} = 2x - 3y^{2}$$

At $x = 2,\ y = -3$:

| Quantity | Analytic | Engine |
|---|---|---|
| $f$ | 19 | 19 |
| $\partial f / \partial x$ | −2 | −2 |
| $\partial f / \partial y$ | −23 | −23 |

---

## Phase 2 — `Tensor` (matrix engine)

Same design, but `data` is a 2-D NumPy array (`np.atleast_2d`) and `grad` is a zero array of matching shape.

### Matrix multiplication

$$C = AB \qquad \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}B^{T}, \qquad \frac{\partial L}{\partial B} = A^{T}\frac{\partial L}{\partial C}$$

```python
self.grad  += out.grad @ other.data.T
other.grad += self.data.T @ out.grad
```

### Elementwise addition and subtraction

$$C = A + B \qquad \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}$$

The local derivative is a matrix of ones, so the incoming gradient passes straight through (negated for the right operand of a subtraction).

```python
self.grad  += out.grad
other.grad += out.grad     # -= for __sub__
```

### Elementwise multiplication (Hadamard)

$$C = A \odot B \qquad \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot B$$

```python
self.grad  += out.grad * other.data
other.grad += self.data * out.grad
```

### Activations

`relu` uses `np.maximum(0, data)` with a boolean mask on the backward pass; `tanh` uses `np.tanh`. Both are the vectorized versions of the scalar derivations above.

```python
self.grad += out.grad * (self.data > 0)    # relu
self.grad += out.grad * (1 - t ** 2)       # tanh
```

### Verification

$$X = \begin{bmatrix} 1 & 2 \end{bmatrix}, \qquad W = \begin{bmatrix} 0.5 & -0.5 \\ 0.2 & 0.8 \end{bmatrix}, \qquad A = \mathrm{ReLU}(XW)$$

| Output | Value |
|---|---|
| $A$ | `[[0.9, 1.1]]` |
| $\partial L / \partial W$ | `[[1, 1], [2, 2]]` |
| $\partial L / \partial X$ | `[[0, 1]]` |

---

## Known Issues

**Broadcasting is not handled.** If a 1×3 bias row is added to a 3×3 matrix, NumPy copies that row three times so the shapes match. The backward pass has to detect that stretching and **sum the gradients back down** to the original 1×3 shape. Right now `__add__`, `__sub__` and `__mul__` all assume the operand shapes already match, so a broadcast forward pass produces a shape mismatch on the backward pass.

**No `zero_grad`.** Gradients accumulate with `+=` by design, but nothing resets them — calling the backward pass twice on the same graph double-counts every gradient.

**`__neg__` never attaches its closure.** The `backward` function is defined but `out._backward = backward` is missing, so negation silently drops gradients.

**Parent tuples.** `Value.tanh`, `Value.relu` and `Value.__neg__` pass `(self)` rather than `(self,)`, which is a bare object rather than a tuple — the topological sort can't iterate it.

**`Tensor` is missing `__pow__` and `__truediv__`.**

## Next Steps

- Fix the parent-tuple and `__neg__` closure bugs above
- Add broadcast-aware gradient reduction to the `Tensor` elementwise ops
- Add `zero_grad` to both engines
- Add `__pow__` / `__truediv__` to `Tensor`
- Gradient-check everything against finite differences

---

## Why I built this

Frameworks like PyTorch hide the chain-rule bookkeeping behind `.backward()`. Building this from scratch was about understanding exactly what that call does: constructing a DAG on the forward pass, then walking it in reverse topological order, applying local derivatives at each node.
