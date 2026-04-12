import torch
import matplotlib.pyplot as plt
import seaborn as sns
import attention
from main import tokenizer_from_state
from transformer import TransformerLM
import numpy as np

def main():
    # 1. Setup and Load Checkpoint
    # Make sure to update this path to where your actual checkpoint is saved!
    checkpoint_path = "checkpoint_step_8000.pt" 
    device = torch.device("cpu") # CPU is fine for a single forward pass
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct the tokenizer and model from the checkpoint
    tokenizer = tokenizer_from_state(checkpoint["tokenizer_state"])
    model_config = checkpoint["model_config"]
    model = TransformerLM(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Very important: put model in evaluation mode!
    # sampled = tokenizer.detokenize(
    #                     model.better_sample_continuation(
    #                         tokenizer.tokenize("hello"),
    #                         500,
    #                         temperature=0.7,
    #                         topK=5,
    #                     )
    #                 )
    # print(f"Sampled text: {sampled}")

    # 2. Prepare the Input Text
    # Keep it short (15-25 chars) so the heatmap is readable
    text = "Hello. This is a test of attention visualization." 
    tokens = tokenizer.tokenize(text)
    
    # Create x tensor and add a batch dimension: shape becomes (1, Sequence_Length)
    x = torch.tensor([tokens]).to(device) 

    # 3. Clear the trap and run the forward pass
    attention.SAVED_ATTENTION_MATRICES.clear()
    
    with torch.no_grad(): # We don't need gradients for visualization
        model(x)
        
    matrices = attention.SAVED_ATTENTION_MATRICES
    print(f"Successfully caught {len(matrices)} attention matrices!")
    
    # 4. Extract and Plot a specific Layer and Head
    n_layers = model_config["n_layers"]
    n_heads = model_config["n_heads"]
    
    # Choose which layer and head to look at (0-indexed)
    # layer_idx = 5
    # head_idx = 4
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
    
            # Calculate which matrix in the flat list corresponds to this layer and head
            matrix_idx = (layer_idx * n_heads) + head_idx
            
            # Extract the specific matrix and remove the batch dimension -> shape (N, N)
            attention_matrix = matrices[matrix_idx][0].numpy()
            
            # Get the actual characters for the labels (so we can read the text on the axes)
            char_labels = [tokenizer.vocab[t] for t in tokens]
            # Replace newlines or spaces with readable strings if necessary
            char_labels = [repr(c).strip("'") for c in char_labels] 

            # 5. Draw the Heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attention_matrix, 
                xticklabels=char_labels, 
                yticklabels=char_labels, 
                cmap="viridis", 
                cbar_kws={'label': 'Attention Weight'}
            )
            
            plt.title(f"Attention Heatmap: Layer {layer_idx + 1}, Head {head_idx + 1}")
            plt.xlabel("Key (The character being attended TO)")
            plt.ylabel("Query (The current character looking back)")
            
            # Ensure the plot is rendered correctly
            plt.tight_layout()
            # plt.show()
            
            plt.savefig(f'figs/attention_heatmap_layer_{layer_idx + 1}_head_{head_idx + 1}.png')  # Save the plot to a file
            plt.close()


if __name__ == "__main__":
    main()