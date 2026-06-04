#include <iostream>
#include <iomanip>
#include "tensor.hpp"

// Helper function to print a tensor's data and its accumulated gradients
void print_tensor_state(const std::string& name, TensorRef t) {
    std::cout << "--- " << name << " ---" << std::endl;
    std::cout << "Data:" << std::endl;
    t->data.print();
    std::cout << "Gradients:" << std::endl;
    t->grad.print();
}

int main() {
    std::cout << "======================================\n";
    std::cout << "     Tensor Verification\n";
    std::cout << "======================================\n\n";

    // 1. Setup the inputs and weights
    // X: Input of shape (1x2)
    Matrix x_mat(1, 2);
    x_mat(0, 0) = 1.0f; 
    x_mat(0, 1) = 2.0f;
    auto X = std::make_shared<Tensor>(x_mat, true); // requires_grad = true

    // W: Weights of shape (2x2)
    Matrix w_mat(2, 2);
    w_mat(0, 0) = 2.0f; w_mat(0, 1) = 0.0f;
    w_mat(1, 0) = 0.0f; w_mat(1, 1) = 2.0f;
    auto W = std::make_shared<Tensor>(w_mat, true);

    // B: Bias of shape (1x2)
    Matrix b_mat(1, 2);
    b_mat(0, 0) = 1.0f; 
    b_mat(0, 1) = 1.0f;
    auto B = std::make_shared<Tensor>(b_mat, true);

    // 2. The Forward Pass: Y=(X * W)+B
    std::cout << "[1/3] Executing Forward Pass: Y = (X * W) + B...\n\n";
    auto Z = matmul(X, W);
    auto Y = add(Z, B);

    // 3. The Backward Pass
    std::cout << "[2/3] Executing Backward Pass (Reverse AutoDiff)...\n\n";
    Y->backward();

    // 4. Verify the Results
    std::cout << "[3/3] Verifying Gradients...\n\n";

    print_tensor_state("Tensor Y (Output)", Y);
    // EXPECTED GRADIENT: [1.0, 1.0] (Base case set by the backward function)

    print_tensor_state("Tensor B (Bias)", B);
    // EXPECTED GRADIENT: [1.0, 1.0] (dY/dB = 1)

    print_tensor_state("Tensor W (Weights)", W);
    // EXPECTED GRADIENT (dY/dW = X^T * dY):
    // [1.0, 1.0]
    // [2.0, 2.0]

    print_tensor_state("Tensor X (Inputs)", X);
    // EXPECTED GRADIENT (dY/dX = dY * W^T):
    // [2.0, 2.0]

    std::cout << "======================================\n";
    std::cout << " If gradients match the expected values above,\n";
    std::cout << " your Day 1 computational graph is complete!\n";
    std::cout << "======================================\n";

    return 0;
}