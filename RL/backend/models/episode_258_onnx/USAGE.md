# ONNX Model Usage Guide

## Model Information
- **Format**: ONNX (Open Neural Network Exchange)
- **Runtime**: ONNX Runtime Web
- **Input Shape**: [1, 1, 84, 84] (grayscale)
- **Output Shape**: [1, 9]
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
const feeds = { observation: tensor };
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
