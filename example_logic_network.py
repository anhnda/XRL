"""
Simple Example: Neural Logic Network
=====================================
Demonstrates the learnable neural logic network on a toy problem.

This script:
1. Creates synthetic data with logical structure
2. Trains a simple neural logic network
3. Extracts and displays learned rules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_logic_layer import LearnableNeuralLogicLayer, binarize_sae_features


def generate_toy_data(n_samples=1000, n_features=10, n_actions=3):
    """
    Generate synthetic data with clear logical rules.

    Rules:
        action_0: f0 AND f1
        action_1: f2 AND NOT f3
        action_2: f4 OR (f5 AND f6)
    """
    features = torch.randint(0, 2, (n_samples, n_features)).float()
    actions = torch.zeros(n_samples, dtype=torch.long)

    for i in range(n_samples):
        f = features[i]

        # Rule for action 0: f0 AND f1
        if f[0] == 1 and f[1] == 1:
            actions[i] = 0
        # Rule for action 1: f2 AND NOT f3
        elif f[2] == 1 and f[3] == 0:
            actions[i] = 1
        # Rule for action 2: f4 OR (f5 AND f6)
        elif f[4] == 1 or (f[5] == 1 and f[6] == 1):
            actions[i] = 2
        else:
            # Random action for samples that don't match any rule
            actions[i] = torch.randint(0, n_actions, (1,)).item()

    return features, actions


def train_simple_logic_network():
    """Train a neural logic network on toy data"""

    print("\n" + "=" * 70)
    print("SIMPLE NEURAL LOGIC NETWORK EXAMPLE")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic data...")
    print("True rules:")
    print("  action_0 ← f0 ∧ f1")
    print("  action_1 ← f2 ∧ ¬f3")
    print("  action_2 ← f4 ∨ (f5 ∧ f6)")

    train_features, train_actions = generate_toy_data(n_samples=1000)
    test_features, test_actions = generate_toy_data(n_samples=200)

    # Create model
    n_features = 10
    n_actions = 3
    n_clauses_per_action = 5

    model = LearnableNeuralLogicLayer(
        n_features=n_features,
        n_actions=n_actions,
        n_clauses_per_action=n_clauses_per_action,
        initial_temp=5.0,
        l0_penalty=1e-3
    )

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_epochs = 100

    print(f"\nTraining for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        # Anneal temperature
        temp = max(0.1, 5.0 * (0.95 ** epoch))
        model.update_temperature(temp)

        model.train()
        optimizer.zero_grad()

        # Forward
        logits = model(train_features)

        # Loss
        action_loss = F.cross_entropy(logits, train_actions)
        complexity_loss = model.complexity_penalty()
        loss = action_loss + complexity_loss

        # Backward
        loss.backward()
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_features)
                test_acc = (test_logits.argmax(1) == test_actions).float().mean()

            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"Test Acc: {test_acc:.3f} | Temp: {temp:.3f}")

    # Extract rules
    print("\n" + "=" * 70)
    print("LEARNED RULES")
    print("=" * 70)

    feature_names = [f"f{i}" for i in range(n_features)]
    action_names = ["action_0", "action_1", "action_2"]

    rules = model.extract_rules(
        feature_names=feature_names,
        action_names=action_names,
        min_weight_threshold=0.1
    )

    for action_name, clauses in rules.items():
        print(f"\n{action_name} ←")
        if clauses == ["⊥"]:
            print("  (never)")
        else:
            for i, clause in enumerate(clauses):
                print(f"    {clause}")
                if i < len(clauses) - 1:
                    print("  ∨")

    # Statistics
    print("\n" + "=" * 70)
    print("RULE STATISTICS")
    print("=" * 70)

    stats = model.count_active_rules(threshold=0.1)
    print(f"Total clauses: {stats['total_clauses']}")
    print(f"Non-empty clauses: {stats['non_empty_clauses']}")
    print(f"Average literals per clause: {stats['avg_literals_per_clause']:.2f}")

    # Final accuracy
    model.eval()
    with torch.no_grad():
        train_logits = model(train_features)
        test_logits = model(test_features)

        train_acc = (train_logits.argmax(1) == train_actions).float().mean()
        test_acc = (test_logits.argmax(1) == test_actions).float().mean()

    print(f"\nFinal Train Accuracy: {train_acc:.3f}")
    print(f"Final Test Accuracy: {test_acc:.3f}")

    print("\n" + "=" * 70)
    print("Compare with true rules:")
    print("  action_0 ← f0 ∧ f1")
    print("  action_1 ← f2 ∧ ¬f3")
    print("  action_2 ← f4 ∨ (f5 ∧ f6)")
    print("=" * 70)


if __name__ == "__main__":
    train_simple_logic_network()
