import * as tf from '@tensorflow/tfjs'
import ort from './onnxInit'  // Use pre-configured ONNX Runtime

/**
 * AI Player Class
 * Handles loading and running the trained RL model for AI gameplay
 * Supports both TensorFlow.js and ONNX models
 */
class AIPlayer {
  constructor() {
    this.model = null
    this.session = null // For ONNX Runtime
    this.isLoaded = false
    this.isLoading = false // Prevent concurrent loads
    this.modelFormat = null // 'tfjs' or 'onnx'
    this.modelType = null // 'cnn' or 'bc' (behavioral cloning)
    this.inputShape = [84, 84, 3] // Height, Width, Channels (for TensorFlow.js)
    this.numActions = 6 // Default for TensorFlow.js models

    // BC model normalization parameters (from training)
    this.bcNormalizeMean = null
    this.bcNormalizeStd = null
  }

  /**
   * Load the model (automatically detects TensorFlow.js vs ONNX)
   * @param {string} modelPath - Path to model.json or model.onnx
   * @param {Function} onProgress - Progress callback (progress, message)
   * @returns {Promise<boolean>} Success status
   */
  async loadModel(modelPath = '/models/rl/tfjs_model/model.json', onProgress = null) {
    // Prevent concurrent loads
    if (this.isLoading) {
      console.log('[AI] Model is already loading, skipping duplicate request')
      return this.isLoaded
    }

    this.isLoading = true

    try {
      console.log('[AI] Loading model from:', modelPath)
      if (onProgress) onProgress(0, 'Starting model load...')

      // Detect model format from file extension
      if (modelPath.endsWith('.onnx')) {
        return await this.loadONNXModel(modelPath, onProgress)
      } else {
        return await this.loadTFJSModel(modelPath, onProgress)
      }
    } catch (error) {
      console.error('[AI] Failed to load model:', error)
      if (onProgress) onProgress(100, `Error: ${error.message}`)
      this.isLoaded = false
      return false
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Load TensorFlow.js model
   * @param {string} modelPath - Path to model.json
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<boolean>} Success status
   */
  async loadTFJSModel(modelPath, onProgress = null) {
    try {
      console.log('[AI] Loading TensorFlow.js model...')
      if (onProgress) onProgress(10, 'Fetching TensorFlow.js model...')

      this.model = await tf.loadGraphModel(modelPath)
      if (onProgress) onProgress(60, 'Model loaded, initializing...')

      this.modelFormat = 'tfjs'
      this.inputShape = [84, 84, 3] // HWC format
      this.numActions = 6 // TensorFlow.js models have 6 actions
      this.isLoaded = true

      console.log('[AI] TensorFlow.js model loaded successfully')
      console.log('[AI] Model inputs:', this.model.inputs.map(i => `${i.name}: ${i.shape}`))
      console.log('[AI] Model outputs:', this.model.outputs.map(o => `${o.name}: ${o.shape}`))

      // Warm up model
      if (onProgress) onProgress(80, 'Warming up model...')
      await this.warmUp()

      if (onProgress) onProgress(100, 'Ready!')
      return true
    } catch (error) {
      console.error('[AI] Failed to load TensorFlow.js model:', error)
      this.isLoaded = false
      return false
    }
  }

  /**
   * Load ONNX model
   * @param {string} modelPath - Path to model.onnx
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<boolean>} Success status
   */
  async loadONNXModel(modelPath, onProgress = null) {
    try {
      console.log('[AI] Loading ONNX model...')
      console.log('[AI] Model path:', modelPath)
      if (onProgress) onProgress(10, 'Fetching ONNX model file...')
      console.log('[AI] Fetching model from:', modelPath)
      console.log('[AI] ONNX Runtime already configured (via onnxInit.js)')
      console.log('[AI] WASM path:', ort.env.wasm.wasmPaths)
      console.log('[AI] Proxy disabled:', !ort.env.wasm.proxy)

      // Fetch the .onnx file as ArrayBuffer
      console.log('[AI] Fetching .onnx file...')
      const modelResponse = await fetch(modelPath)
      if (!modelResponse.ok) {
        throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`)
      }
      const modelBuffer = await modelResponse.arrayBuffer()
      console.log('[AI] Model file fetched:', modelBuffer.byteLength, 'bytes')

      if (onProgress) onProgress(20, 'Fetching model weights...')

      // Also fetch the .onnx.data file (external weights)
      const dataPath = modelPath + '.data'
      console.log('[AI] Fetching .onnx.data file from:', dataPath)
      const dataResponse = await fetch(dataPath)

      let sessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      }

      if (dataResponse.ok) {
        const dataBuffer = await dataResponse.arrayBuffer()
        console.log('[AI] External data file fetched:', dataBuffer.byteLength, 'bytes')

        // Provide external data to ONNX Runtime
        sessionOptions.externalData = [
          {
            data: new Uint8Array(dataBuffer),
            path: 'model.onnx.data'
          }
        ]
        if (onProgress) onProgress(30, 'Loading model with external weights...')
      } else {
        console.log('[AI] No external data file found, model may be self-contained')
        if (onProgress) onProgress(30, 'Loading model...')
      }

      // Create session with the model buffer
      console.log('[AI] Creating InferenceSession with WASM-only backend...')
      this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions)

      console.log('[AI] InferenceSession created successfully')
      if (onProgress) onProgress(70, 'Model loaded, initializing...')

      this.modelFormat = 'onnx'

      // Detect model type based on path (path-based detection to avoid accessing private properties)
      // BC models are stored in paths containing "bc" or "behavioral"
      const isBCModel = modelPath.toLowerCase().includes('bc') || modelPath.toLowerCase().includes('behavioral')

      if (isBCModel) {
        // BC model: [batch_size, 16 features]
        this.modelType = 'bc'
        this.inputShape = [16] // Feature vector
        this.numActions = 4 // BC models: left, right, jump, sprint

        // Set BC normalization parameters (from export_bc_to_onnx.py output)
        this.bcNormalizeMean = new Float32Array([
          3.00547363e+03, 3.79342102e+02, 6.06058455e+00, 7.63759851e-01,
          2.27081388e-01, 5.05694103e+00, 9.04971848e+01, 5.16902395e-02,
          1.05831474e+02, 5.87535133e+01, 6.02987185e-02, 1.16671135e+02,
          8.86899662e+00, 8.26654688e+03, -1.33909332e+02, 3.42415839e-01
        ])
        this.bcNormalizeStd = new Float32Array([
          2.7070098e+03, 1.9233147e+02, 2.9470716e+00, 8.1185551e+00,
          4.2121437e-01, 9.1525330e+01, 5.8899105e+01, 2.2008422e-01,
          6.7948883e+01, 1.4398834e+02, 2.3593968e-01, 2.1963815e+02,
          9.7333290e+01, 3.6122949e+03, 2.2437267e+02, 6.7818624e-01
        ])

        console.log('[AI] Model type: Behavioral Cloning (BC) - Feature-based')
        console.log('[AI] Input: 16 features (player state + environment)')
        console.log('[AI] Output: 4 actions (left, right, jump, sprint)')
      } else {
        // CNN model: [batch_size, 3, 84, 84]
        this.modelType = 'cnn'
        this.inputShape = [3, 84, 84] // CHW format
        this.numActions = 9 // CNN models from PPO training

        console.log('[AI] Model type: CNN (Convolutional) - Image-based')
        console.log('[AI] Input: RGB images 84x84')
        console.log('[AI] Output: 9 actions')
      }

      this.isLoaded = true

      console.log('[AI] ONNX model loaded successfully')
      console.log('[AI] Input names:', this.session.inputNames)
      console.log('[AI] Output names:', this.session.outputNames)
      console.log('[AI] Execution provider: WebAssembly (WASM)')

      // Warm up model (with error handling for concurrent session issues)
      if (onProgress) onProgress(85, 'Warming up model with test inference...')
      console.log('[AI] Starting model warm-up...')
      try {
        await this.warmUp()
        console.log('[AI] Model warm-up complete')
      } catch (warmupError) {
        // If warmup fails (e.g., concurrent session calls in React StrictMode),
        // log it but don't fail the entire load - model is still usable
        console.warn('[AI] Warm-up failed (non-critical):', warmupError.message)
        console.log('[AI] Model loaded successfully despite warm-up issue')
      }

      if (onProgress) onProgress(100, 'AI Ready!')
      return true
    } catch (error) {
      console.error('[AI] Failed to load ONNX model:', error)
      console.error('[AI] Error details:', error.message)
      console.error('[AI] Stack trace:', error.stack)
      this.isLoaded = false
      return false
    }
  }

  /**
   * Warm up the model with a dummy prediction
   * This helps improve performance for the first real prediction
   */
  async warmUp() {
    if (!this.isLoaded) return

    console.log('[AI] Warming up model...')

    if (this.modelFormat === 'onnx') {
      // ONNX warm-up
      let dummyInput, tensor
      if (this.modelType === 'bc') {
        // BC model: 16 features
        dummyInput = new Float32Array(1 * 16).fill(0)
        tensor = new ort.Tensor('float32', dummyInput, [1, 16])
      } else {
        // CNN model: 3x84x84 image
        dummyInput = new Float32Array(1 * 3 * 84 * 84).fill(0)
        tensor = new ort.Tensor('float32', dummyInput, [1, 3, 84, 84])
      }
      const feeds = { [this.session.inputNames[0]]: tensor }
      await this.session.run(feeds)
    } else {
      // TensorFlow.js warm-up
      const dummyInput = tf.zeros([1, ...this.inputShape])
      const prediction = await this.model.predict(dummyInput)

      // Clean up
      dummyInput.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
    }

    console.log('[AI] Model warm-up complete')
  }

  /**
   * Preprocess canvas frame for TensorFlow.js model (HWC format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {tf.Tensor} Preprocessed tensor [1, 84, 84, 3]
   */
  preprocessFrameTFJS(canvas, playerX, cameraX) {
    return tf.tidy(() => {
      // Get image data from canvas
      const ctx = canvas.getContext('2d')

      // Create a view centered on the player
      const viewWidth = 400
      const viewHeight = 300
      const viewX = Math.max(0, playerX - cameraX - viewWidth / 2)
      const viewY = 0

      // Get image data
      const imageData = ctx.getImageData(
        viewX,
        viewY,
        Math.min(viewWidth, canvas.width - viewX),
        Math.min(viewHeight, canvas.height - viewY)
      )

      // Convert to tensor and resize to 84x84
      let tensor = tf.browser.fromPixels(imageData)

      // Resize to model input size (84x84)
      tensor = tf.image.resizeBilinear(tensor, [84, 84])

      // Convert RGB to grayscale using luminosity formula
      // Gray = 0.299*R + 0.587*G + 0.114*B
      const weights = tf.tensor1d([0.299, 0.587, 0.114])
      tensor = tf.sum(tensor.mul(weights), -1, true)

      // Normalize to [0, 1]
      tensor = tensor.div(255.0)

      // Add batch dimension [1, 84, 84, 1]
      tensor = tensor.expandDims(0)

      return tensor
    })
  }

  /**
   * Preprocess canvas frame for ONNX model (CHW format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {ort.Tensor} Preprocessed ONNX tensor [1, 3, 84, 84]
   */
  preprocessFrameONNX(canvas, playerX, cameraX) {
    const ctx = canvas.getContext('2d')

    // Create a view centered on the player
    const viewWidth = 400
    const viewHeight = 300
    const viewX = Math.max(0, playerX - cameraX - viewWidth / 2)
    const viewY = 0

    // Get image data
    const imageData = ctx.getImageData(
      viewX,
      viewY,
      Math.min(viewWidth, canvas.width - viewX),
      Math.min(viewHeight, canvas.height - viewY)
    )

    // Create temporary canvas for resizing
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = 84
    tempCanvas.height = 84
    const tempCtx = tempCanvas.getContext('2d')

    // Draw and resize to 84x84
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = imageData.width
    srcCanvas.height = imageData.height
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0)
    tempCtx.drawImage(srcCanvas, 0, 0, imageData.width, imageData.height, 0, 0, 84, 84)

    // Get resized image data
    const resizedData = tempCtx.getImageData(0, 0, 84, 84)
    const pixels = resizedData.data

    // Convert to grayscale CHW format: [1, 1, 84, 84]
    const input = new Float32Array(1 * 1 * 84 * 84)

    for (let i = 0; i < 84; i++) {
      for (let j = 0; j < 84; j++) {
        const idx = (i * 84 + j) * 4
        const r = pixels[idx] / 255.0
        const g = pixels[idx + 1] / 255.0
        const b = pixels[idx + 2] / 255.0

        // Convert RGB to grayscale using luminosity formula
        // Gray = 0.299*R + 0.587*G + 0.114*B
        const gray = 0.299 * r + 0.587 * g + 0.114 * b

        // Single grayscale channel
        input[i * 84 + j] = gray
      }
    }

    return new ort.Tensor('float32', input, [1, 1, 84, 84])
  }

  /**
   * Preprocess canvas frame (automatically uses correct format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {tf.Tensor|ort.Tensor} Preprocessed tensor
   */
  preprocessFrame(canvas, playerX, cameraX) {
    if (this.modelFormat === 'onnx') {
      return this.preprocessFrameONNX(canvas, playerX, cameraX)
    } else {
      return this.preprocessFrameTFJS(canvas, playerX, cameraX)
    }
  }

  /**
   * Extract 16 game features for BC model from game object
   * @param {object} game - Game state object
   * @param {object} aiPlayer - AI player object
   * @returns {Float32Array} 16 features
   */
  extractGameFeatures(game, aiPlayer) {
    // Extract features similar to training data collection
    const features = new Float32Array(16)

    features[0] = aiPlayer.x || 0
    features[1] = aiPlayer.y || 0
    features[2] = aiPlayer.velocityX || 0
    features[3] = aiPlayer.velocityY || 0
    features[4] = aiPlayer.isOnGround ? 1.0 : 0.0

    // Platform below (closest platform below player)
    let platformBelow = null
    let minDistBelow = Infinity
    game.map.platforms.forEach(p => {
      if (p.y > aiPlayer.y && Math.abs(p.x - aiPlayer.x) < 200) {
        const dist = p.y - aiPlayer.y
        if (dist < minDistBelow) {
          minDistBelow = dist
          platformBelow = p
        }
      }
    })
    features[5] = platformBelow ? platformBelow.x : 0
    features[6] = platformBelow ? platformBelow.y : 0
    features[7] = platformBelow && platformBelow.type === 'ice' ? 1.0 : 0.0

    // Platform ahead (next platform in direction of goal)
    let platformAhead = null
    let minDistAhead = Infinity
    const goalDirection = game.goal.x > aiPlayer.x ? 1 : -1
    game.map.platforms.forEach(p => {
      const inDirection = goalDirection > 0 ? p.x > aiPlayer.x : p.x < aiPlayer.x
      if (inDirection) {
        const dist = Math.abs(p.x - aiPlayer.x) + Math.abs(p.y - aiPlayer.y)
        if (dist < minDistAhead) {
          minDistAhead = dist
          platformAhead = p
        }
      }
    })
    features[8] = platformAhead ? platformAhead.x : 0
    features[9] = platformAhead ? platformAhead.y : 0
    features[10] = platformAhead && platformAhead.type === 'ice' ? 1.0 : 0.0

    // Enemy position (closest enemy or 0)
    let closestEnemy = null
    let minEnemyDist = Infinity
    if (game.enemies && game.enemies.length > 0) {
      game.enemies.forEach(e => {
        if (e.isAlive) {
          const dist = Math.abs(e.x - aiPlayer.x) + Math.abs(e.y - aiPlayer.y)
          if (dist < minEnemyDist) {
            minEnemyDist = dist
            closestEnemy = e
          }
        }
      })
    }
    features[11] = closestEnemy ? closestEnemy.x : 0
    features[12] = closestEnemy ? closestEnemy.y : 0

    // Goal position
    features[13] = game.goal ? game.goal.x : 0
    features[14] = game.goal ? game.goal.y : 0

    // Difficulty (encode as number: easy=0, medium=1, hard=2)
    const difficultyMap = { 'easy': 0.0, 'medium': 1.0, 'hard': 2.0 }
    features[15] = difficultyMap[game.difficulty] || 0.0

    return features
  }

  /**
   * Preprocess BC features (normalize)
   * @param {Float32Array} features - Raw 16 features
   * @returns {ort.Tensor} Normalized features tensor [1, 16]
   */
  preprocessFeaturesBC(features) {
    const normalized = new Float32Array(16)

    // Apply normalization: (x - mean) / std
    for (let i = 0; i < 16; i++) {
      normalized[i] = (features[i] - this.bcNormalizeMean[i]) / this.bcNormalizeStd[i]
    }

    return new ort.Tensor('float32', normalized, [1, 16])
  }

  /**
   * Predict action from current game state
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @param {object} game - Full game state (required for BC models)
   * @param {object} aiPlayer - AI player object (required for BC models)
   * @returns {Promise<number>} Action index
   */
  async predictAction(canvas, playerX, cameraX, game = null, aiPlayer = null) {
    if (!this.isLoaded) {
      console.warn('[AI] Model not loaded')
      return 0 // Default to idle
    }

    try {
      if (this.modelFormat === 'onnx') {
        return await this.predictActionONNX(canvas, playerX, cameraX, game, aiPlayer)
      } else {
        return await this.predictActionTFJS(canvas, playerX, cameraX)
      }
    } catch (error) {
      console.error('[AI] Prediction error:', error)
      return 0 // Default to idle on error
    }
  }

  /**
   * Predict action using TensorFlow.js model
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @returns {Promise<number>} Action index (0-5)
   */
  async predictActionTFJS(canvas, playerX, cameraX) {
    try {
      // Preprocess frame
      const inputTensor = this.preprocessFrameTFJS(canvas, playerX, cameraX)

      // Run inference
      const prediction = await this.model.predict(inputTensor)

      // Get action probabilities
      let actionProbs
      if (Array.isArray(prediction)) {
        actionProbs = prediction[0]
      } else {
        actionProbs = prediction
      }

      // Apply softmax if not already applied
      const probs = tf.softmax(actionProbs)

      // Get action with highest probability
      const action = await probs.argMax(-1).data()

      // Clean up tensors
      inputTensor.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
      probs.dispose()

      return action[0]
    } catch (error) {
      console.error('[AI] TensorFlow.js prediction error:', error)
      return 0
    }
  }

  /**
   * Predict action using ONNX model
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @param {object} game - Full game state (for BC models)
   * @param {object} aiPlayer - AI player object (for BC models)
   * @returns {Promise<number>} Action index
   */
  async predictActionONNX(canvas, playerX, cameraX, game = null, aiPlayer = null) {
    try {
      let inputTensor

      if (this.modelType === 'bc') {
        // BC model: Use game features
        if (!game || !aiPlayer) {
          console.error('[AI] BC model requires game and aiPlayer objects')
          return 0
        }
        const features = this.extractGameFeatures(game, aiPlayer)
        inputTensor = this.preprocessFeaturesBC(features)
      } else {
        // CNN model: Use canvas screenshot
        inputTensor = this.preprocessFrameONNX(canvas, playerX, cameraX)
      }

      // Run inference
      const feeds = { [this.session.inputNames[0]]: inputTensor }
      const results = await this.session.run(feeds)

      // Get output
      const outputName = this.session.outputNames[0]
      const output = results[outputName].data

      // BC models output probabilities directly (sigmoid), CNN models output logits
      let probs
      if (this.modelType === 'bc') {
        // BC: Output is already probabilities from sigmoid
        // Actions are independent binary: [left, right, jump, sprint]
        // Pick action with highest probability > 0.5
        probs = Array.from(output)

        // Find best action (highest probability above threshold)
        let bestAction = 0  // Default: idle (no action)
        let bestProb = 0.3  // Threshold - only act if confident

        for (let i = 0; i < probs.length; i++) {
          if (probs[i] > bestProb) {
            bestProb = probs[i]
            bestAction = i + 1  // +1 because actions are: 0=idle, 1=left, 2=right, 3=jump, 4=sprint
          }
        }

        // Debug: Log probabilities occasionally
        if (Math.random() < 0.05) { // 5% chance to log
          console.log('[AI-BC] Raw probs:', probs.map(x => (x * 100).toFixed(1) + '%'))
          console.log('[AI-BC] Selected action:', ['idle', 'left', 'right', 'jump', 'sprint'][bestAction])
        }

        return bestAction
      } else {
        // CNN: Apply softmax to logits
        const maxLogit = Math.max(...output)
        const expValues = Array.from(output).map(x => Math.exp(x - maxLogit))
        const sumExp = expValues.reduce((a, b) => a + b, 0)
        probs = expValues.map(x => x / sumExp)

        // Get action with highest probability (greedy)
        const action = probs.indexOf(Math.max(...probs))

        // Debug: Log probabilities occasionally
        if (Math.random() < 0.05) { // 5% chance to log
          console.log('[AI-CNN] Logits:', Array.from(output).map(x => x.toFixed(2)))
          console.log('[AI-CNN] Probs:', probs.map(x => (x * 100).toFixed(1) + '%'))
          console.log('[AI-CNN] Selected action:', action, '| Top 3:',
            probs.map((p, i) => ({action: i, prob: p}))
              .sort((a, b) => b.prob - a.prob)
              .slice(0, 3)
              .map(x => `${x.action}:${(x.prob * 100).toFixed(1)}%`).join(', ')
          )
        }

        return action
      }
    } catch (error) {
      console.error('[AI] ONNX prediction error:', error)
      return 0
    }
  }

  /**
   * Predict action with exploration (stochastic policy)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @param {number} temperature - Sampling temperature (default 1.0)
   * @returns {Promise<number>} Sampled action index
   */
  async predictActionStochastic(canvas, playerX, cameraX, temperature = 1.0) {
    if (!this.isLoaded) {
      return 0
    }

    try {
      const inputTensor = this.preprocessFrame(canvas, playerX, cameraX)
      const prediction = await this.model.predict(inputTensor)

      let actionProbs
      if (Array.isArray(prediction)) {
        actionProbs = prediction[0]
      } else {
        actionProbs = prediction
      }

      // Apply temperature scaling
      const scaledLogits = actionProbs.div(temperature)
      const probs = tf.softmax(scaledLogits)

      // Sample from distribution
      const probsArray = await probs.data()
      const action = this.sampleFromDistribution(probsArray)

      // Clean up
      inputTensor.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
      scaledLogits.dispose()
      probs.dispose()

      return action
    } catch (error) {
      console.error('[AI] Stochastic prediction error:', error)
      return 0
    }
  }

  /**
   * Sample action from probability distribution
   * @param {Float32Array} probs - Action probabilities
   * @returns {number} Sampled action index
   */
  sampleFromDistribution(probs) {
    const rand = Math.random()
    let cumSum = 0

    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i]
      if (rand < cumSum) {
        return i
      }
    }

    return probs.length - 1
  }

  /**
   * Convert action index to action name
   * @param {number} actionIndex - Action index (0-5)
   * @returns {string} Action name
   */
  getActionName(actionIndex) {
    const actions = ['idle', 'left', 'right', 'jump', 'sprint_right', 'duck']
    return actions[actionIndex] || 'unknown'
  }

  /**
   * Unload model and free resources
   */
  dispose() {
    if (this.model) {
      this.model.dispose()
      this.model = null
    }
    if (this.session) {
      this.session = null
    }
    this.isLoaded = false
    this.modelFormat = null
    console.log('[AI] Model disposed')
  }

  /**
   * Check if model is loaded and ready
   * @returns {boolean}
   */
  isReady() {
    return this.isLoaded && (this.model !== null || this.session !== null)
  }

  /**
   * Get model info
   * @returns {object} Model information
   */
  getModelInfo() {
    if (!this.isLoaded) {
      return {
        loaded: false,
        message: 'Model not loaded'
      }
    }

    const info = {
      loaded: true,
      format: this.modelFormat,
      inputShape: this.inputShape,
      numActions: this.numActions
    }

    if (this.modelFormat === 'tfjs') {
      info.backend = tf.getBackend()
      info.memory = tf.memory()
    } else if (this.modelFormat === 'onnx') {
      info.runtime = 'ONNX Runtime Web'
      info.executionProviders = ['webgl', 'wasm']
    }

    return info
  }
}

export default AIPlayer
