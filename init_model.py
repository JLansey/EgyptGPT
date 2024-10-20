# init_model.py
from transformers import pipeline

def initialize_model():
    pipe = pipeline("text2text-generation", model="mattiadc/hiero-transformer")
    print("Model initialized successfully.")
    return pipe

if __name__ == "__main__":
    initialize_model()
