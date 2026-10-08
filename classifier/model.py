"""Our trainable feed-forward network; MiniLM is a separate fixed encoder."""

from torch import nn

INTENTS = ["SUMMARY", "COMPARISON", "METHODOLOGY", "RESULTS", "LIMITATIONS", "DEFINITION"]
MODEL_CONFIG = {"input_dim": 384, "hidden_dims": [128, 64], "dropout": 0.2, "num_classes": 6}


class IntentClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dims=(128, 64), dropout=0.2, num_classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, embeddings):
        # Raw logits; CrossEntropyLoss applies the appropriate normalization.
        return self.network(embeddings)
