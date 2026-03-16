"""
Interpret continuous concepts as binary for human understanding
Train with continuous, interpret as binary!
"""

import torch
import numpy as np

def binarize_concepts(concepts):
    """
    Convert continuous concepts to binary for interpretation

    Args:
        concepts: [batch, hidden_dim] tensor with TopK sparsity

    Returns:
        binary_concepts: [batch, hidden_dim] binary {0, 1}
    """
    return (concepts > 0).float()


def interpret_sample(concepts, concept_names):
    """
    Get human-readable interpretation of a sample

    Args:
        concepts: [hidden_dim] tensor
        concept_names: dict mapping concept_idx -> name

    Returns:
        active_concepts: list of active concept names
    """
    binary = (concepts > 0).numpy()
    active_indices = np.where(binary)[0]

    active_concepts = [concept_names[idx] for idx in active_indices]
    return active_concepts


# Example usage
if __name__ == "__main__":
    # Load continuous model (trained normally)
    from ConceptExtractor import ActionConceptModel
    checkpoint = torch.load('./concept_models/concept_model.pt', map_location='cpu')

    config = checkpoint['config']
    model = ActionConceptModel(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        n_actions=config['n_actions'],
        k=config['k'],
        predictor_type=config['predictor_type']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load sample data
    sample_data = torch.load('./concept_models/sample_data.pt', map_location='cpu')
    features = sample_data['features']
    actions = sample_data['actions']

    # Define concept names (from your analysis!)
    concept_names = {
        1: "Early Exploration (Turn Right)",
        2: "At Door (Ready to Toggle)",
        3: "Key Detected (Pickup Phase)",
        5: "Clear Path (Navigate Forward)"
    }

    # Get concepts for a sample
    with torch.no_grad():
        _, concepts = model.sae(features[:10])

    # Analyze each sample
    action_names = ['Turn Left', 'Turn Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']

    print("Sample Analysis (Binary Interpretation):")
    print("=" * 80)

    for i in range(10):
        sample_concepts = concepts[i]
        action = actions[i].item()

        # Binarize for interpretation
        binary = binarize_concepts(sample_concepts)
        active = interpret_sample(sample_concepts, concept_names)

        print(f"\nSample {i}:")
        print(f"  Continuous: {sample_concepts.numpy()}")
        print(f"  Binary:     {binary.numpy()}")
        print(f"  Active concepts: {active}")
        print(f"  Action: {action_names[action]}")

        # Explain action based on concepts
        if 2 in np.where(binary.numpy())[0]:
            print(f"  → At door, likely to Toggle or approach")
        if 3 in np.where(binary.numpy())[0]:
            print(f"  → Key detected, likely to Pickup or orient")
        if 5 in np.where(binary.numpy())[0]:
            print(f"  → Path clear, likely to move Forward")
        if 1 in np.where(binary.numpy())[0]:
            print(f"  → Exploring, likely to Turn Right")
