# load the training data
git submodule update --init --recursive

# unzip training data
unzip  hiero_transformer/training_data.zip -d data/egypt_char
unzip  hiero_transformer/test_and_validation_data.zip -d data/egypt_char


# install the required packages for transformers
pip install torch numpy transformers datasets tiktoken wandb tqdm
pip install triton


pip install matviz