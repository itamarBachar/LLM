import torch
import attention
import os
import tempfile
import data

def test_kqv():
    """Test that kqv correctly computes Keys, Queries, and Values from input."""
    B, N, D = 2, 4, 8  # batch_size, sequence_length, embedding_dim
    n_heads = 2
    d_head = D // n_heads
    
    # Create a KQV linear layer for one head
    kqv_matrix = attention.create_kqv_matrix(D, n_heads)
    
    # Create input with shape (B, N, D)
    x = torch.randn(B, N, D)
    
    # Compute k, q, v
    k, q, v = attention.kqv(x, kqv_matrix)
    
    # Verify shapes: each should be (B, N, d_head)
    assert k.shape == (B, N, d_head), f"k shape {k.shape} != {(B, N, d_head)}"
    assert q.shape == (B, N, d_head), f"q shape {q.shape} != {(B, N, d_head)}"
    assert v.shape == (B, N, d_head), f"v shape {v.shape} != {(B, N, d_head)}"
    
    # Verify that k, q, v are different (not all the same due to different weight matrices)
    assert not torch.allclose(k, q), "k and q should be different"
    assert not torch.allclose(q, v), "q and v should be different"
    
    print("✓ test_kqv passed")

def test_attention_scores():
    # Test case 1: Simple example with manually computed expected output
    # Using orthogonal unit vectors for easy computation
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # shape (1, 2, 2)
    b = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # shape (1, 3, 2)
    
    # Manual computation: A[i,j] = (a[i] · b[j]) / sqrt(d)
    # d = 2, so sqrt(d) = sqrt(2) ≈ 1.414213562
    expected_output = torch.tensor([
        [[1.0/2**0.5, 0.0, 1.0/2**0.5],
         [0.0, 1.0/2**0.5, 1.0/2**0.5]]
    ])  # shape (1, 2, 3)
    
    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output), f"Output {A} != expected {expected_output}"
    
    # Test case 2: Multi-batch test with identity-like matrices
    a = torch.eye(3).unsqueeze(0)  # shape (1, 3, 3) - identity matrix
    b = torch.eye(3).unsqueeze(0)  # shape (1, 3, 3) - identity matrix
    
    # For identity matrices, a @ b^T = I, so A = I / sqrt(3)
    expected_output = torch.eye(3).unsqueeze(0) / (3**0.5)
    
    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output), f"Output {A} != expected {expected_output}"
    
    # Test case 3: Test with batch size > 1
    a = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])  # shape (2, 1, 2)
    b = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])  # shape (2, 1, 2)
    
    # Batch 0: (2 · 1 + 0 · 0) / sqrt(2) = 2 / sqrt(2)
    # Batch 1: (0 · 1 + 3 · 1) / sqrt(2) = 3 / sqrt(2)
    expected_output = torch.tensor([
        [[2.0 / 2**0.5]],
        [[3.0 / 2**0.5]]
    ])
    
    A = attention.attention_scores(a, b)
    assert torch.allclose(A, expected_output), f"Output {A} != expected {expected_output}"
    
    print("✓ test_attention_scores passed")

def test_self_attention():
    # Test case 1: Simple one-position output from weighted sum over values
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape (1, 2, 2)
    A = torch.tensor([[[0.0, 1.0]]])  # shape (1, 1, 2)

    # softmax([0, 1]) = [e^0/(e^0+e^1), e^1/(e^0+e^1)]
    w0 = 1.0 / (1.0 + torch.exp(torch.tensor(1.0)))
    w1 = torch.exp(torch.tensor(1.0)) / (1.0 + torch.exp(torch.tensor(1.0)))
    expected_output = torch.tensor([[[w0 * 1.0 + w1 * 3.0, w0 * 2.0 + w1 * 4.0]]])

    sa = attention.self_attention(v, A)
    assert torch.allclose(sa, expected_output), f"Output {sa} != expected {expected_output}"

    # Test case 2: Uniform attention should average the values
    v = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])  # shape (1, 2, 2)
    A = torch.zeros(1, 1, 2)  # shape (1, 1, 2), softmax -> [0.5, 0.5]
    expected_output = torch.tensor([[[3.0, 5.0]]])

    sa = attention.self_attention(v, A)
    assert torch.allclose(sa, expected_output), f"Output {sa} != expected {expected_output}"

    # Test case 3: Batch size > 1
    v = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 2.0], [4.0, 0.0]]
    ])  # shape (2, 2, 2)
    A = torch.tensor([
        [[1.0, 0.0]],
        [[0.0, 0.0]]
    ])  # shape (2, 1, 2)

    # Batch 0 uses softmax([1,0]); batch 1 is uniform average
    s0 = torch.exp(torch.tensor(1.0)) / (1.0 + torch.exp(torch.tensor(1.0)))
    s1 = 1.0 / (1.0 + torch.exp(torch.tensor(1.0)))
    expected_output = torch.tensor([
        [[s0, s1]],
        [[3.0, 1.0]]
    ])

    sa = attention.self_attention(v, A)
    assert torch.allclose(sa, expected_output), f"Output {sa} != expected {expected_output}"

    print("✓ test_self_attention passed")


def test_tokenizer_save_load_round_trip():
    tokenizer = data.CharTokenizer()
    tokenizer.train(["hello world", "another line"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(path)
        loaded = data.CharTokenizer.load(path)

    assert loaded.vocab == tokenizer.vocab
    assert loaded.stoi == tokenizer.stoi
    assert loaded.detokenize(loaded.tokenize("hello")) == "hello"

    print("✓ test_tokenizer_save_load_round_trip passed")

