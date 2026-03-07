"""
Export trained PPO model to ONNX format for ONNX Runtime Web deployment
"""

import os
import sys
import argparse
import torch
import onnx
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path


def load_model(model_path):
    """Load trained Stable-Baselines3 PPO model"""
    print(f"[ONNX-EXPORT] Loading model from {model_path}")
    model = PPO.load(model_path, device='cpu')
    print("[ONNX-EXPORT] Model loaded successfully")
    return model


def extract_policy_network(model):
    """
    Extract the policy network from the PPO model.
    Returns logits for action selection.
    """
    print("[ONNX-EXPORT] Extracting policy network...")

    class PolicyLogits(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            # Extract CNN features from observation
            features = self.features_extractor(obs)
            # Pass through policy MLP
            latent_pi, _ = self.mlp_extractor(features)
            # Get action logits
            logits = self.action_net(latent_pi)
            return logits

    policy_net = PolicyLogits(model.policy)
    policy_net.eval()
    print("[ONNX-EXPORT] Policy network extracted")
    return policy_net


def export_to_onnx(policy_network, output_path, observation_shape=(1, 1, 84, 84)):
    """
    Export policy network to ONNX format.

    Args:
        policy_network: Extracted policy network
        output_path: Path to save .onnx file
        observation_shape: Input shape (batch, channels=1 for grayscale, height, width)
    """
    print(f"[ONNX-EXPORT] Exporting to ONNX...")
    print(f"[ONNX-EXPORT] Input shape: {observation_shape} (grayscale)")
    print(f"[ONNX-EXPORT] Output path: {output_path}")

    # Create dummy input
    dummy_input = torch.randn(*observation_shape)

    # Export to ONNX
    torch.onnx.export(
        policy_network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # Compatible with onnxruntime-web
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_logits'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'}
        }
    )

    print(f"[ONNX-EXPORT] SUCCESS: Model exported to {output_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[ONNX-EXPORT] ONNX model verified successfully")

    return output_path


def create_usage_guide(output_dir, model_info):
    """Create a usage guide markdown file"""
    usage_path = output_dir / "USAGE.md"

    usage_content = f"""# ONNX Model Usage Guide

## Model Information
- **Format**: ONNX (Open Neural Network Exchange)
- **Runtime**: ONNX Runtime Web
- **Input Shape**: {model_info['input_shape']}
- **Output Shape**: {model_info['output_shape']}
- **Actions**: 9 discrete actions (0-8)

## Action Space
0. Idle
1. Left
2. Right
3. Jump
4. Sprint + Right
5. Duck
6. Jump + Left
7. Jump + Right
8. Sprint + Jump + Right

## Loading in JavaScript

```javascript
// Load ONNX Runtime Web
import * as ort from 'onnxruntime-web';

// Load the model
const session = await ort.InferenceSession.create('model.onnx');

// Prepare input (84x84x1 grayscale image, normalized to 0-255)
const inputData = new Float32Array(84 * 84 * 1);
// ... fill with grayscale pixel data ...

// Create tensor (batch=1, channels=1 (grayscale), height=84, width=84)
const tensor = new ort.Tensor('float32', inputData, [1, 1, 84, 84]);

// Run inference
const feeds = {{ observation: tensor }};
const results = await session.run(feeds);

// Get action logits
const logits = results.action_logits.data;

// Select action with highest logit (greedy policy)
const action = logits.indexOf(Math.max(...logits));
```

## Notes
- Input images should be GRAYSCALE format, channels-first (CHW) order
- Convert RGB to grayscale: Gray = 0.299*R + 0.587*G + 0.114*B
- Pixel values should be in range [0, 255]
- The model outputs raw logits - use argmax for action selection
- For stochastic policy, apply softmax to logits and sample
"""

    with open(usage_path, 'w') as f:
        f.write(usage_content)

    print(f"[ONNX-EXPORT] Usage guide created: {usage_path}")


def main():
    parser = argparse.ArgumentParser(description='Export PPO model to ONNX')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to .zip model file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for ONNX files')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path)

    # Extract policy network
    policy_net = extract_policy_network(model)

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    observation_shape = (1, 1, 84, 84)  # Batch=1, Channels=1 (grayscale), H=84, W=84
    export_to_onnx(policy_net, str(onnx_path), observation_shape)

    # Create usage guide
    model_info = {
        'input_shape': '[1, 1, 84, 84] (grayscale)',
        'output_shape': '[1, 9]'
    }
    create_usage_guide(output_dir, model_info)

    print("\n" + "="*60)
    print("[ONNX-EXPORT] EXPORT COMPLETE!")
    print("="*60)
    print(f"Model: {onnx_path}")
    print(f"Usage Guide: {output_dir / 'USAGE.md'}")
    print("="*60)


if __name__ == "__main__":
    main()
