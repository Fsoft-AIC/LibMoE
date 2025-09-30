import numpy as np
from numerical_exp.model import MoEModel

def main():
    print("Starting test...")
    model = MoEModel(num_experts=3, K=2, num_shared=1)
    print('Model created successfully!')
    
    X = np.random.randn(10)
    Y = np.random.randn(10)
    print(f"X.shape={X.shape}, Y.shape={Y.shape}")
    
    model.initialize_parameters(X, Y)
    print('Parameters initialized successfully!')
    
    gating_probs = model.compute_gating_probs(X)
    print('Gating probabilities computed successfully!')
    
    likelihoods = model.compute_expert_likelihoods(X, Y)
    print('Expert likelihoods computed successfully!')
    
    responsibilities, gating_probs = model.e_step(X, Y)
    print('E-step completed successfully!')
    
    model.m_step(X, Y, responsibilities, gating_probs)
    print('M-step completed successfully!')
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
