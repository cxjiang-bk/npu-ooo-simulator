from __future__ import annotations


def attention_block():
    """Return a small pre-norm attention block without head reshape."""

    import torch

    class AttentionBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)
            self.q_proj = torch.nn.Linear(8, 8)
            self.k_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)
            self.out_proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            hidden = self.norm(x)
            q = self.q_proj(hidden)
            k = self.k_proj(hidden)
            v = self.v_proj(hidden)
            probabilities = torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)),
                dim=-1,
            )
            return x + self.out_proj(torch.matmul(probabilities, v))

    return AttentionBlock().eval()


def attention_micrograph():
    """Return the QK-softmax-PV subset currently verified with torch-xla."""

    import torch

    class AttentionMicrograph(torch.nn.Module):
        def forward(self, q, k, v):
            scores = torch.matmul(q, k.transpose(-2, -1))
            probabilities = torch.softmax(scores, dim=-1)
            return torch.matmul(probabilities, v)

    return AttentionMicrograph().eval()
