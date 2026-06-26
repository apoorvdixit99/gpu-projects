# Micrograd

A from-scratch implementation of a scalar-valued autograd engine and a small neural network library built on top of it.

**Reference:** [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy  
**Lecture:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

## What this is

`engine.py` implements a `Value` class that wraps a scalar and tracks every operation applied to it. Each operation records a `_backward` closure that computes local gradients via the chain rule. Calling `.backward()` on any output node runs a reverse-mode topological traversal — automatically differentiating the full expression graph with no external dependencies.

`nn.py` layers `Neuron`, `Layer`, and `MLP` on top of the engine using only `Value` arithmetic. A neuron computes `tanh(w·x + b)`; an MLP stacks layers of neurons and exposes `.parameters()` so a training loop can zero and update every weight in one pass.

---

## Supported operations

| Operation | Method | Notes |
|---|---|---|
| Addition | `__add__`, `__radd__` | scalar promotion |
| Multiplication | `__mul__`, `__rmul__` | scalar promotion |
| Power | `__pow__` | int/float exponents |
| Subtraction | `__sub__`, `__neg__` | built from add + mul |
| Division | `__truediv__` | built from pow |
| Tanh | `.tanh()` | activation |
| Exp | `.exp()` | used to build tanh manually |
| Backprop | `.backward()` | reverse topo sort |

---

## Project structure

```
project-7-micrograd/
├── src/
│   ├── engine.py        Value class — scalar autograd engine
│   └── nn.py            Neuron, Layer, MLP built on Value
├── results/             Training outputs (gitignored)
├── micrograd.ipynb      Walkthrough notebook — engine derivation, viz, MLP training
├── ISSUES.md            Log of issues encountered
├── requirements.txt
└── README.md
```

---

## How to run

> Activate the shared venv from the `Nvidia/` parent directory first:
> ```powershell
> .venv\Scripts\Activate.ps1
> cd project-7-micrograd
> ```

**Notebook** — interactive walkthrough of the engine and training loop:
```powershell
jupyter notebook micrograd.ipynb
```

**Use the engine directly:**
```python
from src.engine import Value
from src.nn import MLP

model = MLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(30):
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    for p in model.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
```

---

## Key design decisions

**Why scalar-valued instead of tensor-valued?**  
Operating on individual scalars makes every intermediate result a node in the computation graph. The backward pass is then a literal traversal of Python objects — no matrix calculus, no shape bookkeeping. The tradeoff is that training is orders of magnitude slower than a vectorized engine; the gain is that every gradient formula is legible in the `_backward` closures.

**Why reverse-mode (backprop) instead of forward-mode?**  
A network with P parameters has P outputs in forward-mode AD — one per parameter — but only one output in reverse-mode. For neural networks, where P >> number of loss scalars, reverse-mode computes all gradients in a single backward pass. Forward-mode would require P separate passes.

**Why `+=` for gradient accumulation?**  
A node can be used in multiple downstream operations (e.g., a weight shared across neurons). Each use contributes an independent gradient term; they must be summed, not overwritten. Using `+=` in every `_backward` closure handles fan-out correctly without special-casing.

**Why zero gradients before each backward call?**  
`_backward` accumulates into `.grad` with `+=`. Without zeroing, gradients from previous iterations pile up and corrupt the update. The training loop zeros all parameter gradients before calling `loss.backward()` so each iteration starts clean.
