# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a comprehensive neural networks educational repository (CST-435-Group) containing 7+ distinct AI/ML projects demonstrating different architectures and applications. Each project is self-contained with its own training scripts, models, and interfaces.

## Project Structure

The repository is organized into independent project directories:

```
CST-435-Group/
├── AIT-204-Genetic-Alg/    # Genetic algorithm education package
├── ANN_Project/             # Artificial Neural Network (NBA team selection)
├── CNN_Project/             # Convolutional Neural Network (fruit classification)
├── GAN/                     # Generative Adversarial Network (tank image generation)
├── NLP/                     # Natural Language Processing (sentiment analysis)
├── RL/                      # Reinforcement Learning (platformer game AI)
├── rnn-text-generator/      # Recurrent Neural Network (text generation)
└── launcher/                # Unified web interface for ANN, CNN, NLP projects
```

**Important**: Each project operates independently. When working on one project, do not modify files in other projects unless explicitly requested.

## Common Development Commands

### Python Projects (ANN, CNN, GAN, RL, NLP, RNN)

**Install dependencies:**
```bash
cd <project-directory>
pip install -r requirements.txt
```

**Train models:**
```bash
# GAN: Train dual-conditional tank GAN
cd GAN
python train_gan_dual_conditional.py

# RL: Train platformer agent
cd RL/backend
python training/train_agent.py

# CNN: Train fruit classifier
cd CNN_Project
python train_model.py

# RNN: Train text generator with optimal settings
cd rnn-text-generator/backend
python app/train_optimal.py

# ANN: Uses pre-trained model, training code in main.py
```

**Run Streamlit interfaces (where available):**
```bash
# CNN Project
cd CNN_Project
streamlit run streamlit_app.py

# ANN Project
cd ANN_Project
streamlit run streamlit_app.py
```

### Web Applications (NLP, RNN, Launcher)

**Backend (FastAPI):**
```bash
cd <project>/backend
pip install -r requirements.txt
python main.py
# Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (React):**
```bash
cd <project>/frontend
npm install
npm start           # Development server
npm run build       # Production build
```

### Genetic Algorithm Project

**Run working application:**
```bash
cd AIT-204-Genetic-Alg/shakespeare-ga
npm install
npm start
```

## Architecture Overview by Project

### 1. GAN (Generative Adversarial Network)

**Purpose**: Generate realistic tank images conditioned on tank type and view angle

**Key Files**:
- `train_gan_dual_conditional.py` - Main training script with dual conditioning
- `models_dual_conditional.py` - Generator/Discriminator architectures
- `generate_dual_conditional.py` - Image generation from trained models
- `GAN/backend/gan_router.py` - FastAPI endpoint for web interface

**Training**:
- Dual conditional: tank type (24 classes) + view angle (8 classes)
- 20x20 RGB images
- Discriminator uses AdaIN for conditional normalization
- Generator uses class embeddings + label smoothing
- Typical training: 200-500 epochs with adaptive learning rates

**Models Directory**:
- `models_dual_conditional/` contains checkpoints by epoch
- Each checkpoint includes generator, discriminator, and training stats

### 2. RL (Reinforcement Learning)

**Purpose**: Train AI agent to play side-scrolling platformer game

**Key Components**:
- `backend/training/environment.py` - Custom Gym environment
- `backend/training/map_generator.py` - Procedural level generation
- `backend/training/train_agent.py` - PPO training with Stable-Baselines3
- `backend/training/player.py` - Player class with physics

**Training**:
- Algorithm: PPO (Proximal Policy Optimization)
- Observation: 84x84 visual input
- Training duration: 4-8 hours on GPU for 1M steps
- Models saved to `backend/models/`

**Export Options**:
- `export_model.py` - Convert PyTorch to TensorFlow.js for browser
- `export_model_onnx.py` - Export to ONNX format

**Important**: The project uses behavioral cloning + RL hybrid approach. See `train_behavioral_cloning.py` and `train_hybrid.py`.

### 3. NLP (Natural Language Processing)

**Purpose**: Sentiment analysis on 7-point scale (-3 to +3)

**Architecture**:
- Backend: FastAPI + DistilBERT (Hugging Face Transformers)
- Frontend: React 18 + Vite + Tailwind CSS
- Originally for movie reviews, adapted for hospital reviews

**Key Files**:
- `nlp-react/backend/main.py` - FastAPI endpoints
- `nlp-react/backend/model.py` - SentimentAnalyzer class
- `nlp-react/frontend/src/App.jsx` - Main React component
- `nlp-react/frontend/src/services/api.js` - API client

**Deployment**:
- Frontend: Vercel (React static site)
- Backend: Render.com or Railway.app (Python FastAPI)
- See `NLP/nlp-react/CLAUDE.md` for detailed architecture

### 4. RNN (Recurrent Neural Network)

**Purpose**: Text generation using LSTM trained on classical literature

**Architecture**:
- Backend: PyTorch LSTM with word-level tokenization
- Frontend: React TypeScript with Recharts visualizations
- Training data: Bible translations, Shakespeare, classic literature

**Key Files**:
- `backend/app/text_generator.py` - LSTM model and training logic
- `backend/app/train_optimal.py` - Best training configuration (10k vocab, 256 units)
- `backend/app/main.py` - FastAPI server with model switching
- `frontend/src/App.tsx` - Main React component

**Training Variants**:
- `train.py` - Basic (150 units, full vocab)
- `train_optimal.py` - Recommended (256 units, 10k vocab, best accuracy)
- `train_improved.py` - Large (256 units, 3 layers, full vocab)

**Important Preprocessing Rule**: Regex `r'\b[a-zA-Z]+(?:\'[a-z]+)?\b'` must match between `extract_vocabulary.py` and `text_generator.py` to prevent unknown tokens.

**See**: `rnn-text-generator/CLAUDE.md` for detailed guidance

### 5. CNN (Convolutional Neural Network)

**Purpose**: Fruit classification with optimizer comparison

**Key Files**:
- `train_model.py` - Main training script
- `streamlit_app.py` - Interactive web interface with analytics
- `cnn_standalone.py` - Standalone inference script
- `optimizer_comparison.py` - Compare SGD, Adam, RMSprop, AdaGrad

**Training**:
- Dataset: Fruits-360 (preprocessed 100x100 images)
- Architecture: Conv layers + Batch Norm + MaxPool + FC layers
- Augmentation: Random flips, rotation, color jitter

**Running**:
```bash
cd CNN_Project
streamlit run streamlit_app.py  # Opens at http://localhost:8501
```

### 6. ANN (Artificial Neural Network)

**Purpose**: NBA team selection using neural network

**Key Files**:
- `main.py` - Combined training + Streamlit interface
- `streamlit_app.py` - Standalone Streamlit app
- `src/model.py` - Neural network model definition
- `src/select_team.py` - Team selection logic
- `best_model.pth` - Pre-trained model

**Running**:
```bash
cd ANN_Project
streamlit run streamlit_app.py
```

### 7. Genetic Algorithm (Educational Package)

**Purpose**: Interactive teaching tool for genetic algorithms

**Structure**:
- `genetic-algorithm-handout.html` - Interactive browser-based explanation
- `shakespeare-ga/` - Complete React TypeScript application
- `STUDENT-INSTRUCTIONS.md` - Step-by-step implementation guide

**Demo**:
- Target: Evolve random text into Shakespeare quotes
- Demonstrates selection, crossover, mutation, elitism
- Real-time visualization with fitness charts

**See**: `AIT-204-Genetic-Alg/CLAUDE.md` for detailed architecture

### 8. Launcher (Unified Interface)

**Purpose**: Single web interface providing unified access to ANN, CNN, NLP, RNN, RL, and optionally GAN projects through a centralized FastAPI gateway with lazy-loaded models.

#### Architecture Overview

The Launcher uses a **microservices-inspired architecture** where a single FastAPI gateway routes requests to project-specific routers, each of which lazily loads models from the original project directories. This allows all projects to be accessed through one interface without duplicating code.

```
Frontend (Vercel/localhost:3000)
    │
    ├─ /ann   → ANN page (lazy loaded React component)
    ├─ /cnn   → CNN page (lazy loaded React component)
    ├─ /nlp   → NLP page (lazy loaded React component)
    ├─ /rnn   → RNN page (lazy loaded React component)
    ├─ /rl    → RL page (lazy loaded React component)
    └─ /gan   → GAN page (lazy loaded React component, if available)
    │
    ▼ HTTP API Requests
    │
Backend Gateway (Railway/localhost:8000)
    │
    ├─ /api/ann/*  → routers/ann.py
    │                  └─ Lazy loads: ANN_Project/best_model.pth
    │                  └─ Imports: ANN_Project/src/*
    │
    ├─ /api/cnn/*  → routers/cnn.py
    │                  └─ Lazy loads: CNN_Project/models/*
    │                  └─ Recreates CNN architecture
    │
    ├─ /api/nlp/*  → routers/nlp.py
    │                  └─ Lazy loads: DistilBERT from Hugging Face
    │                  └─ Imports: NLP/nlp-react/backend/*
    │
    ├─ /api/rnn/*  → routers/rnn.py
    │                  └─ Lazy loads: rnn-text-generator/backend/saved_models/*
    │                  └─ Imports: rnn-text-generator/backend/app/*
    │
    ├─ /api/rl/*   → routers/rl.py
    │                  └─ Lazy loads: RL/backend/models/*
    │                  └─ Imports: RL/backend/training/*
    │
    └─ /api/gan/*  → GAN/backend/gan_router.py (optional)
                       └─ Lazy loads: GAN/models_dual_conditional/*
```

#### Backend: Lazy Loading Architecture

**Why Lazy Loading?**
- Railway/local environments have limited RAM (512MB-8GB)
- Each project's model is 100-500MB
- Loading all models at startup would cause OOM errors
- Models only load when first requested for that project

**How It Works:**

Each router (e.g., `routers/ann.py`) follows this pattern:

```python
# Global model storage - starts as None
ann_model = None
ann_preprocessor = None
ann_data = None

def load_ann_model():
    """Lazy load model on first request"""
    global ann_model, ann_preprocessor, ann_data

    # Return cached model if already loaded
    if ann_model is not None:
        return ann_model, ann_preprocessor, ann_data

    # Locate the original project directory
    possible_paths = [
        Path(__file__).parent.parent.parent / "ANN_Project",  # Local dev
        Path("/app") / ".." / "ANN_Project",                  # Railway deployment
    ]

    # Import from original project
    from src.model import create_model
    from src.select_team import TeamSelector

    # Load pre-trained model
    model = torch.load("ANN_Project/best_model.pth")

    # Cache in global variables
    ann_model = model

    return ann_model, ann_preprocessor, ann_data

@router.post("/select-team")
async def select_team(request: TeamSelectionRequest):
    """Endpoint that triggers lazy loading"""
    model, preprocessor, data = load_ann_model()  # Loads on first call

    # Use model for prediction
    result = model.predict(...)
    return result
```

**Key Points:**
- Models start as `None` (no memory used)
- First API request triggers `load_*_model()` function
- Model loads from original project directory (no code duplication)
- Model cached in global variable for subsequent requests
- If backend restarts, models reload on next request

#### Project Integration Strategy

The launcher **does not duplicate code** - it imports directly from original projects:

**ANN Router:**
```python
# Adds ANN_Project to sys.path
sys.path.insert(0, "../ANN_Project")

# Imports existing code
from src.model import create_model
from src.select_team import TeamSelector
```

**CNN Router:**
```python
# Recreates same CNN architecture as CNN_Project/train_model.py
class FruitCNN(nn.Module):
    # Same layers, same hyperparameters
```

**NLP Router:**
```python
# Imports from NLP project
sys.path.insert(0, "../NLP/nlp-react/backend")
from model import SentimentAnalyzer
```

**RNN Router:**
```python
# Imports from RNN project
sys.path.insert(0, "../rnn-text-generator/backend")
from app.text_generator import TextGenerator
```

**RL Router:**
```python
# Imports from RL project
sys.path.insert(0, "../RL/backend")
from training.environment import PlatformerEnv
from training.map_generator import MapGenerator
```

This means **changes to original projects automatically affect the launcher** - no synchronization needed.

#### Frontend: Page-Based Routing with Lazy Loading

The React frontend uses **React Router** with **lazy loading** for each project page:

**File: `launcher/frontend/src/App.jsx`**
```javascript
import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

// Lazy load pages - only loads code when user navigates to that page
const ANNProject = lazy(() => import('./pages/ANNProject'));
const CNNProject = lazy(() => import('./pages/CNNProject'));
const NLPProject = lazy(() => import('./pages/NLPProject'));
const RNNProject = lazy(() => import('./pages/RNNProject'));
const RLProject = lazy(() => import('./pages/RLProject'));

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/ann" element={<ANNProject />} />
          <Route path="/cnn" element={<CNNProject />} />
          <Route path="/nlp" element={<NLPProject />} />
          <Route path="/rnn" element={<RNNProject />} />
          <Route path="/rl" element={<RLProject />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
```

**Benefits of Frontend Lazy Loading:**
- Initial bundle only contains Home page
- Each project page loads on-demand when clicked
- Faster initial page load
- Smaller JavaScript bundles

#### API Communication

**File: `launcher/frontend/src/services/api.js`**
```javascript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const annAPI = {
  selectTeam: (data) => axios.post(`${API_BASE_URL}/api/ann/select-team`, data),
  getDataInfo: () => axios.get(`${API_BASE_URL}/api/ann/data-info`)
};

export const cnnAPI = {
  predict: (formData) => axios.post(`${API_BASE_URL}/api/cnn/predict`, formData),
  getModelInfo: () => axios.get(`${API_BASE_URL}/api/cnn/model/info`)
};

export const nlpAPI = {
  analyze: (text) => axios.post(`${API_BASE_URL}/api/nlp/analyze`, { text }),
  getExamples: () => axios.get(`${API_BASE_URL}/api/nlp/examples`)
};
```

#### Deployment Configuration

**Frontend (Vercel):**
- Deploy directory: `launcher/frontend`
- Build command: `npm run build`
- Environment variable: `VITE_API_URL=https://your-backend.railway.app` (or localhost for local dev)
- Framework: Vite (auto-detected)

**Backend (Railway or Local):**
- Deploy directory: `launcher/backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment: Python 3.11
- **Important**: Backend needs access to parent directories (ANN_Project, CNN_Project, etc.)

#### Running Locally

**Terminal 1 - Backend:**
```bash
cd launcher/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd launcher/frontend
npm install
npm run dev  # Runs on http://localhost:3000 or http://localhost:5173
```

**Environment Setup:**
```bash
# launcher/frontend/.env
VITE_API_URL=http://localhost:8000
```

#### Health Monitoring

The gateway provides health checks for all services:

```bash
# Check which models are loaded
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "services": {
    "ann": {"status": "not_loaded", "model_loaded": false},
    "cnn": {"status": "not_loaded", "model_loaded": false},
    "nlp": {"status": "ready", "model_loaded": true},
    "rnn": {"status": "not_loaded", "model_loaded": false},
    "gan": {"status": "unavailable", "available": false}
  }
}
```

Models show `"not_loaded"` until first request, then switch to `"ready"`.

#### Key Files

**Backend:**
- `launcher/backend/main.py` - FastAPI gateway, router registration, CORS config
- `launcher/backend/routers/ann.py` - ANN endpoints with lazy loading
- `launcher/backend/routers/cnn.py` - CNN endpoints with lazy loading
- `launcher/backend/routers/nlp.py` - NLP endpoints with lazy loading
- `launcher/backend/routers/rnn.py` - RNN endpoints with lazy loading
- `launcher/backend/routers/rl.py` - RL endpoints with lazy loading
- `launcher/backend/routers/docs.py` - Project documentation endpoints

**Frontend:**
- `launcher/frontend/src/App.jsx` - React Router with lazy-loaded pages
- `launcher/frontend/src/pages/ANNProject.jsx` - ANN interface
- `launcher/frontend/src/pages/CNNProject.jsx` - CNN interface
- `launcher/frontend/src/pages/NLPProject.jsx` - NLP interface
- `launcher/frontend/src/pages/RNNProject.jsx` - RNN interface
- `launcher/frontend/src/pages/RLProject.jsx` - RL interface
- `launcher/frontend/src/services/api.js` - Axios client with typed endpoints

**Documentation:**
- `launcher/ARCHITECTURE.md` - Complete architectural documentation
- `launcher/README.md` - Setup and deployment instructions

#### Important Development Notes

**When Modifying Projects:**
- Changes to `ANN_Project/src/` automatically affect launcher ANN endpoints
- Changes to `CNN_Project/train_model.py` architecture require updating `routers/cnn.py`
- Changes to `NLP/nlp-react/backend/model.py` automatically affect launcher NLP endpoints
- The launcher is a **consumer** of the projects, not a duplicate

**Memory Considerations:**
- Only one project's model loads at a time (unless multiple are requested)
- If you need all models loaded: ensure >4GB RAM available
- Railway free tier can handle 2-3 models simultaneously

**Path Resolution:**
- Backend uses multiple path resolution strategies (local dev vs Railway)
- If imports fail, check `sys.path` modifications in router files
- GAN router is optional (loads only if GAN/backend/gan_router.py exists)

**Testing the Launcher:**
```bash
# Test backend is running
curl http://localhost:8000/

# Test health of all services
curl http://localhost:8000/health

# Test specific project (triggers lazy load)
curl -X POST http://localhost:8000/api/ann/select-team \
  -H "Content-Type: application/json" \
  -d '{"method": "balanced", "start_year": "1996-97", "end_year": "2019-20"}'
```

## Critical Development Notes

### File Path Conventions
- Always use paths relative to the project directory when running scripts
- Training scripts expect to be run from their respective project roots
- Model paths are typically `models/`, `saved_models/`, or `backend/models/`

### GPU vs CPU
- All projects support both GPU (CUDA) and CPU training
- GPU is strongly recommended for GAN, RL, and RNN training
- CNN and ANN train reasonably fast on CPU
- PyTorch auto-detects CUDA availability

### Model File Formats
- **PyTorch**: `.pt`, `.pth` files (most projects)
- **ONNX**: `.onnx` files (RL export option)
- **TensorFlow.js**: `model.json` + weight shards (RL browser deployment)
- **Pickled**: `.pkl` files (tokenizers, preprocessors)

### Windows-Specific Considerations
- DataLoader `num_workers=0` to avoid multiprocessing issues
- Use `python -m pip install` for reliable package installation
- PowerShell may require quotes for certain commands

### Common Issues

**Import Errors**: Check that you're in the correct directory and dependencies are installed
**CUDA Out of Memory**: Reduce batch size or use CPU
**Model Not Found**: Verify model files exist in expected location (check README for each project)
**Port Already in Use**: Default ports are 8000 (backend), 3000 (frontend), 8501 (Streamlit)

## Testing Individual Projects

### Before Making Changes
1. Read the project's README.md if available
2. Check for project-specific CLAUDE.md in subdirectories
3. Verify dependencies are installed
4. Test existing functionality before modifications

### After Making Changes
1. Test training/inference pipelines
2. Verify model loading/saving works
3. Check web interfaces (if applicable)
4. Ensure no cross-project contamination

## Project-Specific CLAUDE.md Files

Several projects have detailed CLAUDE.md files with additional guidance:
- `AIT-204-Genetic-Alg/CLAUDE.md` - Genetic algorithm implementation details
- `NLP/nlp-react/CLAUDE.md` - Full-stack NLP architecture
- `rnn-text-generator/CLAUDE.md` - RNN training and generation specifics

**Always check for project-specific CLAUDE.md before making significant changes.**

## Git Workflow

This is an active development repository:
- Main branch: `main`
- Recent commits show integration work (AI to RL, launcher updates)
- Clean working directory (no uncommitted changes at session start)

When committing changes:
- Focus commits on single project when possible
- Include project prefix in commit messages (e.g., "GAN: fix training script")
- Test before committing, especially for cross-project changes (launcher)
