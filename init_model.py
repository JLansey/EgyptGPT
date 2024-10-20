from transformers import pipeline
import torch

def initialize_model():
    device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
    pipe = pipeline("text2text-generation", model="mattiadc/hiero-transformer", device=device)
    print(f"Model initialized successfully on {'GPU' if device == 0 else 'CPU'}.")
    return pipe

if __name__ == "__main__":
    initialize_model()