# Automatic Gradient Calculator

A from-scratch autograd engine — no `torch`, no `tensorflow`. Implements reverse-mode automatic differentiation via computation graphs and topological sort.

## Core Objective

Given a scalar loss $L$, compute

$$\frac{\partial L}{\partial w_i} \quad \text{for all } i \in N$$

for every parameter $w_i$ in the graph.

## Structure

Everything is built on a `Value` (scalar) / `Tensor` (array) object with four attributes:

| Attribute | Meaning |
|---|---|
| `data` | The actual numerical value |
| `grad` | The accumulated derivative of the final output w.r.t. this variable |
| `_prev` | A set of parent nodes |
| `_backward` | A function stored on the object that knows exactly how to apply the chain rule for that op |

## Build Phases

- **Phase 1:** Build `Value` (scalar engine) ✅
- **Phase 2:** Build `Tensor` (array/matrix engine)

## Forward Pass

Standard operators (`__add__`, `__mul__`, `__sub__`, `__truediv__`, `__pow__`, `__neg__`) each build a node and record how to backprop through themselves:

```python
c = a * b
c._prev = (a, b)
c.data = a.data * b.data
```

## Mathematical Primitives

- **Basic arithmetic:** Addition, Multiplication, Subtraction, Division, Power
- **Activation functions:** ReLU, tanh

## Backward Pass

Gradients flow backward through the graph via **topological sort**, applying the chain rule at each node:

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} \cdot \frac{\partial c}{\partial a}$$

---

## `Value` (scalar) — Operator Derivations

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
other.grad += out.grad * (-1)
```

### Power

$$c = a^{n} \qquad \frac{\partial c}{\partial a} = n \cdot a^{n-1}$$

```python
self.grad += n * out.grad * (self.data ** (n - 1))
```

### Also implemented

- Negation — `__neg__(self)`
- Division — `__truediv__(self, other)`
- Power — `__pow__(self, other)`

### Activation — tanh

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \qquad \frac{d}{dx}\tanh(x) = 1 - \tanh^{2}(x)$$

```python
self.grad += out.grad * (1 - out.data ** 2)
```

### Activation — ReLU

$$c = \max(0, a) \qquad \frac{\partial c}{\partial a} = \begin{cases} 1 & a > 0 \\ 0 & a \le 0 \end{cases}$$

```python
self.grad += out.grad * (self.data > 0)
```

The boolean acts as 1 when `True` and 0 when `False`.

---

## `Tensor` (matrix) — Operator Derivations

### Matrix multiplication

$$C = A B \qquad \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B^{T}, \quad \frac{\partial L}{\partial B} = A^{T} \frac{\partial L}{\partial C}$$

```python
self.grad  += out.grad @ other.data.T
other.grad += self.data.T @ out.grad
```

### Elementwise addition

$$C = A + B \qquad \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}$$

The local derivative is a matrix of ones with the same shape as the input:

```python
self.grad += out.grad * np.ones_like(self.data)
```

### ⚠️ Known Issue — Broadcasting

If a 1×3 bias row is added to a 3×3 matrix, NumPy automatically copies that bias row three times so the shapes match. In the backward pass, this stretching has to be detected and the gradients **summed back up** so they match the original 1×3 shape.

**Status:**

- [x] `__mul__`
- [ ] `__sub__`
- [ ] `__add__` — still doesn't handle broadcasting

---

## Next steps

- Handle broadcasting correctly in `__add__` and `__sub__` backward passes
- Complete the `Tensor` division and power operators
- Add gradient-checking tests against finite differences

---

## Why I built this

Frameworks like PyTorch hide the chain-rule bookkeeping behind `.backward()`. Building this from scratch was about understanding exactly what that call does: constructing a DAG on the forward pass, then walking it in reverse topological order, applying local derivatives at each node.
