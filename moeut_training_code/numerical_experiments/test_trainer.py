import numpy as np
from numerical_exp.model import MoEModel
from numerical_exp.trainer import MoETrainer

def main():
    print("Starting trainer test...")
    model = MoEModel(num_experts=3, K=2, num_shared=1)
    print('Model created successfully!')
    
    X = np.random.randn(100)
    Y = np.random.randn(100)
    print(f"X.shape={X.shape}, Y.shape={Y.shape}")
    
    trainer = MoETrainer(model, max_iter=5, verbose=True)
    print('Trainer created successfully!')
    
    trainer.fit(X, Y)
    print('Training completed successfully!')
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
