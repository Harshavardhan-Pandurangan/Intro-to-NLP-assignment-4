# NLP Assignment 4
## Harshavardhan P - 2021111003

<br>
These are the parameters to run the class `RNNTrainer`. This class can do all the tasks required in the assignment with proper parameters. The class is defined in the all the 4 files required.

## Parameters
- `train_data` - loaded list of class labels and description sentences
- `test_data` - loaded list of class labels and description sentences
- `train_embeddings` - True or False to train embeddings
- `train_model` - True or False to train model
- `batch_size` - batch size for training model
- `embedding_dim` - dimension of the embeddings
- `embedding_type` - SVD or Skipgram
- `context_size` - context size for SVD or Skipgram
- `hidden_size` - hidden size for RNN
- `epochs` - number of epochs for training
- `save_embeddings` - True or False to save embeddings
- `embeddings_path` - path to save or load embeddings
- `save_model` - True or False to save model
- `load_model` - True or False to load model
- `model_path` - path to save or load model
- `logging` - True or False to logging to wandb
- `run_name` - name of the run in wandb

> Note: Context size in SVD and Skip Gram is window size, so it is 1 more than the number of words on each side of the word.
