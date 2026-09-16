# Automatic Gradient Calculator

A from-scratch autograd engine — no `torch`, no `tensorflow`. Implements reverse-mode automatic differentiation via computation graphs and topological sort.

## Core Objective

Given a scalar loss $L$, compute:

$$\left\{ \frac{\partial L}{\partial w_i} : i \in N \right\}$$

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

Standard operators (`--add--`, `--mul--`, `--sub--`, `--truediv--`, `--pow--`, `--neg--`) each build a node and record how to backprop through themselves:

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

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d}\cdot\frac{\partial d}{\partial c}\cdot\frac{\partial c}{\partial a} = \frac{\partial L}{\partial c}\cdot\frac{\partial c}{\partial a}$$

---

## `Value` (scalar) — Operator Derivations

### Addition: $c = a + b$

$$\text{self.grad} \;+\!= \text{out.grad}$$
$$\text{other.grad} \;+\!= \text{out.grad}$$

### Multiplication: $c = a \times b$

$$\text{self.grad} \;+\!= \text{out.grad} \times \text{other.data}$$
$$\text{other.grad} \;+\!= \text{out.grad} \times \text{self.data}$$

### Subtraction: $c = a - b$

$$\frac{\partial L}{\partial a} = \frac{\partial L}{\partial c}\times 1 \qquad\qquad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial c}\times(-1)$$

### Power: $c = a^{n}$

$$\frac{\partial L}{\partial a} = (\text{out.grad})\,(n\,a^{\,n-1})$$
$$\text{self.grad} \;+\!= n\,(\text{out.grad})\,(\text{self.data}^{\,n-1})$$

### Also implemented
- Negation: `--neg--(self)`
- Division: `--truediv--(self, other)`
- Power: `--pow--(self, other)`

### Activation — tanh: $c = \tanh(a)$

$$\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} \qquad\qquad \frac{d}{dx}\tanh(x) = 1-\tanh^2(x)$$

$$\text{self.grad} \;+\!= (\text{out.grad})\times\left(1-(\text{out.data})^{2}\right)$$

### Activation — ReLU: $c = \max(0, a)$

$$\text{local derivative} = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

$$\text{self.grad} \;+\!= (\text{out.grad}) \times (\text{self.data} > 0)$$

*(boolean treated as 1 if `True`, 0 if `False`)*

---

## `Tensor` (matrix) — Operator Derivations

### Matrix multiplication: $C = A \cdot B$

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\,B^{T} \qquad\qquad \frac{\partial L}{\partial B} = A^{T}\,\frac{\partial L}{\partial C}$$

### Elementwise addition: $C = A + B$

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\times \texttt{np.ones\_like(self.data)}$$

*(a matrix of the same shape as `self.data`, `out.data`, `other.data`)*

### ⚠️ Known Issue — Broadcasting

If a 1×3 bias row is added to a 3×3 matrix, NumPy automatically copies that bias row three times so the shapes match. In the backward pass, this stretching has to be detected and the gradients **summed back up** so they match the original shape (1×3).

**Status:**
- [x] `mul` — done
- [ ] `sub`
- [ ] `add` — still doesn't handle broadcasting

---

## Why I built this

Frameworks like PyTorch hide the chain-rule bookkeeping behind `.backward()`. Building this from scratch was about understanding exactly what that call does: constructing a DAG on the forward pass, then walking it in reverse topological order, applying local derivatives at each node.
