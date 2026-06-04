# Deep Learning Framework in C++

A lightweight, header-only deep learning framework written from scratch in modern C++. It implements a PyTorch-style dynamic computational graph with reverse-mode automatic differentiation, a small layer API, common optimizers, and end-to-end demos that train an MLP on **XOR** and **MNIST**.

The project is built around teaching/learning the internals of autograd and neural network training — every component (tensors, gradients, layers, optimizers, data loading) is implemented from first principles with no external ML dependencies.

---

## Features

- **Header-only**: just `#include` what you need — no build system required for use.
- **Reverse-mode autodiff** with a dynamic computation graph (built per forward pass).
- **Tensor abstraction** with reference-counted ownership (`std::shared_ptr`) for safe graph memory management.
- **Matrix backend** with OpenMP-parallelized matmul, add, subtract, scalar multiply, and elementwise multiply.
- **Layers**: `Linear` (Xavier init), `ReLU`, `Sigmoid`, `Tanh`, and a `Sequential` container.
- **Losses**: `mse_loss` and a numerically-stable fused `softmax_cross_entropy`.
- **Optimizers**: `SGD` and `Adam` (with bias correction).
- **Data loading**: a shuffling mini-batch `DataLoader` and a CSV loader for MNIST.
- **Model serialization**: simple binary save of trained parameters.

---

## Repository Layout

| File | Purpose |
| --- | --- |
| [new_matrix.hpp](new_matrix.hpp) | Matrix class — storage, indexing, matmul, transpose, elementwise ops (OpenMP-parallel). |
| [matrix.hpp](matrix.hpp) | Earlier templated matrix prototype (kept for reference; `new_matrix.hpp` is the active backend). |
| [tensor.hpp](tensor.hpp) | `Tensor` class, computational graph, and `backward()` (topological sort + reverse pass). Includes `add`, `subtract`, `multiply`, `matmul`. |
| [tensor_ops.hpp](tensor_ops.hpp) | Differentiable ops: bias broadcast (`add_bias`), `relu`, `sigmoid`, `tanh_op`. |
| [layers.hpp](layers.hpp) | `Layer` base class and `Linear` (fully connected) layer. |
| [activations.hpp](activations.hpp) | `ReLULayer`, `SigmoidLayer`, `TanhLayer` wrappers. |
| [sequential.hpp](sequential.hpp) | `Sequential` model container that chains layers and collects parameters. |
| [losses.hpp](losses.hpp) | `mse_loss`, `softmax_cross_entropy` (stable). |
| [optimizers.hpp](optimizers.hpp) | `SGD`, `Adam`. |
| [dataloader.hpp](dataloader.hpp) | Shuffling mini-batch iterator. |
| [mnist_loader.hpp](mnist_loader.hpp) | CSV loader for MNIST (pixels normalized to [0,1]). |
| [test_tensor.cpp](test_tensor.cpp) | Sanity check for the autograd engine on `Y = X·W + B`. |
| [xor_demo.cpp](xor_demo.cpp) | Trains a tiny MLP (2 → 8 → 1) to learn XOR with MSE + SGD. |
| [mnist_demo.cpp](mnist_demo.cpp) | Trains an MLP (784 → 128 → 64 → 10) on MNIST with Adam + cross-entropy. |
| `data/` | Place `mnist_train.csv` and `mnist_test.csv` here. |

---

## Build

The framework is header-only, so building means compiling one of the demo `.cpp` files. A C++17 compiler is required. OpenMP is optional but recommended for speed.

### Linux / macOS (g++ or clang++)

```bash
g++ -std=c++17 -O3 -fopenmp xor_demo.cpp   -o xor_demo
g++ -std=c++17 -O3 -fopenmp mnist_demo.cpp -o mnist_demo
g++ -std=c++17 -O3 -fopenmp test_tensor.cpp -o test_tensor
```

### Windows (MSVC)

```powershell
cl /std:c++17 /O2 /openmp xor_demo.cpp
cl /std:c++17 /O2 /openmp mnist_demo.cpp
cl /std:c++17 /O2 /openmp test_tensor.cpp
```

### Windows (MinGW g++)

```powershell
g++ -std=c++17 -O3 -fopenmp xor_demo.cpp   -o xor_demo.exe
g++ -std=c++17 -O3 -fopenmp mnist_demo.cpp -o mnist_demo.exe
```

If you don't have OpenMP, drop `-fopenmp` / `/openmp` — the code falls back to a serial loop via `#ifdef _OPENMP`.

---

## Running the Demos

### Autograd sanity check

```bash
./test_tensor
```

Computes `Y = X·W + B` and prints the gradients of `Y`, `B`, `W`, `X`. Expected values are documented inline in [test_tensor.cpp](test_tensor.cpp).

### XOR

```bash
./xor_demo
```

Trains a `2 → 8 → 1` MLP (tanh hidden, sigmoid output) on the XOR truth table for 3000 epochs and prints the learned predictions.

### MNIST

Download MNIST in CSV format and place the files at:

```
data/mnist_train.csv
data/mnist_test.csv
```

Each CSV row is `label,p1,p2,...,p784` (a common public format). Then:

```bash
./mnist_demo                                    # uses defaults
./mnist_demo --epochs 5 --batch 128             # tweak training
./mnist_demo --train data/mnist_train.csv \
             --test  data/mnist_test.csv \
             --model mnist_model.bin \
             --epochs 3 --batch 128 --max-train 0
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--train PATH` | `data/mnist_train.csv` | Training CSV path |
| `--test PATH`  | `data/mnist_test.csv`  | Test CSV path |
| `--model PATH` | `mnist_model.bin`      | Where to save parameters |
| `--epochs N`   | `3`                    | Training epochs |
| `--batch N`    | `128`                  | Mini-batch size |
| `--max-train N`| `0` (all)              | Cap on training samples (for fast smoke tests) |

After training, the model parameters are dumped to a binary file via `save_model`.

---

## Minimal Example

```cpp
#include "sequential.hpp"
#include "activations.hpp"
#include "losses.hpp"
#include "optimizers.hpp"

bool Tensor::grad_enabled = true;

int main() {
    Sequential model;
    model.add(std::make_shared<Linear>(784, 128));
    model.add(std::make_shared<ReLULayer>());
    model.add(std::make_shared<Linear>(128, 10));

    Adam opt(model.parameters(), 1e-3f);

    // X: (N, 784), targets: vector<int> of length N
    auto Xt = std::make_shared<Tensor>(X, /*requires_grad=*/false);

    opt.zero_grad();
    auto logits = model.forward(Xt);
    auto loss   = softmax_cross_entropy(logits, targets);
    loss->backward();
    opt.step();
}
```

---

## How the Autograd Works

Each differentiable op (e.g. `matmul`, `add`, `relu`) builds an output `Tensor` that stores:

1. The result `data` (a `Matrix`).
2. Its `parents` in the graph.
3. A `backward_func` lambda that knows how to push gradient into each parent.

`Tensor::backward()`:

1. Topologically sorts the graph reachable from the output (DFS, output last).
2. Seeds the output gradient with ones.
3. Walks the sort in reverse, invoking each node's `backward_func` to accumulate parent gradients.

Inference-time graph building can be disabled by setting `Tensor::grad_enabled = false`, which both saves memory and avoids the cost of recording the graph (the MNIST demo uses this during evaluation).

---

## Notes & Limitations

- CPU only; no GPU/CUDA backend.
- Single precision (`float`) throughout (`typedef float datatype` in `new_matrix.hpp`).
- Designed for learning — small models train comfortably; large-scale workloads are out of scope.
- The MNIST CSV loader expects the standard `label, pixel0, ..., pixel783` format with pixel values in `[0, 255]`.

---

## License

No license specified yet — add one before publishing or reusing this code.
