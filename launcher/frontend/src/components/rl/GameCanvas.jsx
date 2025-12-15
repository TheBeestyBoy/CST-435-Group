import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapGenerator } from './MapGenerator'
import { Player } from './Player'
import { AIGameInstance } from './AIGameInstance'
import { rlAPI } from '../../services/api'
import './GameCanvas.css'

// Scale factor for displaying distance (distance is calculated in pixels, so we scale it down)
const DISTANCE_SCALE = 20

// Training data collection
const TRAINING_DATA_BATCH_SIZE = 500 // Send every 500 frames
const COLLECT_TRAINING_DATA = true // Set to false to disable data collection

/**
 * Main game canvas component
 * Handles rendering, input, and game loop
 * @param {boolean} enableAI - Enable AI opponent
 * @param {string} episodeModelPath - Path to episode-specific AI model (optional)
 * @param {number} playingEpisode - Episode number being played against (optional)
 * @param {function} onGameComplete - Callback when game completes with time and stats
 * @param {string} difficulty - Game difficulty (easy, medium, hard)
 */
export default function GameCanvas({ onGameEnd, enableAI = false, episodeModelPath = null, playingEpisode = null, onGameComplete, difficulty = 'easy', playerColor = '#4CAF50', username = 'Anonymous' }) {
  const canvasRef = useRef(null)
  const navigate = useNavigate()
  const [gameState, setGameState] = useState('playing') // playing, won, lost
  const [stats, setStats] = useState({ score: 0, distance: 0, aiScore: 0, aiDistance: 0, time: 0 })

  // Training data collection
  const trainingDataRef = useRef([])  // Batch for periodic sending
  const allGameplayDataRef = useRef([])  // ALL data for score validation
  const sessionIdRef = useRef(`${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
  const frameNumberRef = useRef(0)
  const [aiStatus, setAiStatus] = useState(enableAI ? 'loading' : 'disabled') // loading, ready, error, disabled
  const aiStatusRef = useRef(enableAI ? 'loading' : 'disabled') // Ref for game loop access
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [gameTime, setGameTime] = useState(0)

  // Determine which model to use - episode model or default trained model
  const modelPath = episodeModelPath || '/models/rl/tfjs_model/model.json'

  // Helper to update AI status (both state and ref)
  const updateAiStatus = (status) => {
    console.log('[AI-STATUS] Updating status:', aiStatusRef.current, '→', status)
    setAiStatus(status)
    aiStatusRef.current = status
  }

  // Debug logging
  console.log('═══════════════════════════════════════════════')
  console.log('[GameCanvas] Component rendered with props:')
  console.log('  enableAI:', enableAI)
  console.log('  episodeModelPath:', episodeModelPath)
  console.log('  playingEpisode:', playingEpisode)
  console.log('  Final modelPath:', modelPath)
  console.log('═══════════════════════════════════════════════')

  // Game objects
  const gameRef = useRef({
    player: null,
    aiInstance: null, // Separate game instance for AI
    map: null,
    mapSeed: null, // Store seed for AI instance
    cameraX: 0,
    keys: {},
    lastTime: 0,
    animationFrame: null,
    startTime: 0,
    elapsedTime: 0,
    countdown: null, // Countdown value (3, 2, 1) or null when done
    countdownStartTime: null // When countdown started
  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const game = gameRef.current

    // Store canvas in game object
    game.canvas = canvas

    // Initialize game with seeded map (so AI can use same map)
    const mapGen = new MapGenerator(1920, 1080, 32)
    game.mapSeed = Date.now() // Generate seed once
    game.map = mapGen.generateMap(game.mapSeed, difficulty)

    // Create player at spawn position
    game.player = new Player(game.map.spawn.x, game.map.spawn.y)
    console.log('[GAME] Player spawned at:', game.map.spawn.x, game.map.spawn.y)
    console.log('[GAME] Map seed:', game.mapSeed)

    game.cameraX = 0
    game.jumpKeyWasPressed = false

    // Add references for BC model feature extraction
    game.difficulty = difficulty
    game.enemies = game.map.enemies
    game.goal = game.map.goal

    // Initialize AI if enabled - uses separate game instance
    if (enableAI) {
      console.log('\n🤖 ═══════════ AI INITIALIZATION STARTED ═══════════')
      console.log('[AI-INIT] Creating separate AI game instance...')
      console.log('[AI-INIT] AI will run in isolated environment')
      console.log('[AI-INIT] AI will use identical map instance (seed:', game.mapSeed, ')')
      console.log('[AI-INIT] Model path:', modelPath)

      updateAiStatus('loading')
      setLoadingProgress(0)
      setLoadingMessage('Initializing AI...')

      // Use the same map instance for AI (ensures identical layout for fair race)
      // No need to regenerate - player and AI share the exact same platforms
      const aiMap = game.map

      // Create AI instance with callbacks for status updates
      const onAIReady = () => {
        console.log('✅ [AI-READY] AI opponent loaded and ready!')
        updateAiStatus('ready')
        setLoadingProgress(100)
        setLoadingMessage('Get Ready!')

        // Start countdown after AI is ready
        console.log('[COUNTDOWN] Starting 3... 2... 1... countdown')
        game.countdown = 3
        game.countdownStartTime = performance.now()
      }

      const onAIError = (errorMsg) => {
        console.error('❌ [AI-ERROR] AI failed:', errorMsg)
        updateAiStatus('error')
        setLoadingMessage(`Error: ${errorMsg}`)
      }

      game.aiInstance = new AIGameInstance(aiMap, modelPath, difficulty, onAIReady, onAIError)

      console.log('[AI-INIT] AI instance created')
      console.log('🤖 ════════════════════════════════════════════════\n')
    } else {
      console.log('[AI-INIT] AI disabled by user, skipping initialization')
      updateAiStatus('disabled')
      // No countdown needed when playing solo
      game.countdown = null
    }

    // Clear all keys helper
    const clearAllKeys = () => {
      game.keys = {}
      game.jumpKeyWasPressed = false
    }

    // Keyboard controls - attach to canvas for better focus control
    const handleKeyDown = (e) => {
      // Prevent default for game keys first
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'Shift', 'a', 'A', 'd', 'D', 'w', 'W', 's', 'S'].includes(e.key)) {
        e.preventDefault()
      }

      // Only set to true if not already true (prevents key repeat)
      if (!game.keys[e.key]) {
        game.keys[e.key] = true
      }

      // Safety: If Shift is pressed while no movement keys are held, don't register it
      // This prevents issues with browser Shift key handling
      if (e.key === 'Shift') {
        const hasMovementKey =
          ('ArrowLeft' in game.keys) || ('ArrowRight' in game.keys) ||
          ('a' in game.keys) || ('A' in game.keys) ||
          ('d' in game.keys) || ('D' in game.keys)

        if (!hasMovementKey) {
          // Shift pressed alone - clear any stuck keys
          delete game.keys['ArrowLeft']
          delete game.keys['ArrowRight']
          delete game.keys['a']
          delete game.keys['A']
          delete game.keys['d']
          delete game.keys['D']
          // NOTE: Don't reset jump flag here - only in keyUp to prevent repeated jumps
        }
      }
    }

    const handleKeyUp = (e) => {
      // Completely remove the key from the object
      delete game.keys[e.key]

      // Reset jump flag when jump keys are released
      if ([' ', 'ArrowUp', 'w', 'W'].includes(e.key)) {
        game.jumpKeyWasPressed = false
      }

      // Clear movement keys when Shift is released to prevent stuck movement
      if (e.key === 'Shift') {
        delete game.keys['ArrowLeft']
        delete game.keys['ArrowRight']
        delete game.keys['a']
        delete game.keys['A']
        delete game.keys['d']
        delete game.keys['D']
        // Also reset jump flag when Shift is released to prevent stuck jumps
        game.jumpKeyWasPressed = false
      }
    }

    // Clear keys when window loses focus
    const handleBlur = () => {
      clearAllKeys()
    }

    // Ensure canvas is focusable
    canvas.setAttribute('tabindex', '0')
    canvas.focus()

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', handleBlur)
    canvas.addEventListener('blur', handleBlur)

    // Game loop
    const gameLoop = (timestamp) => {
      let deltaTime = timestamp - game.lastTime
      game.lastTime = timestamp

      // Clamp deltaTime to prevent huge spikes on first frame or lag
      // Max 33ms (2 frames at 60fps) to prevent physics from breaking
      if (deltaTime > 33) {
        console.warn('[GAME] Large deltaTime detected:', deltaTime, 'ms - clamping to 33ms')
        deltaTime = 33
      }

      // Skip first frame entirely if deltaTime is 0 (initialization frame)
      if (deltaTime === 0) {
        game.animationFrame = requestAnimationFrame(gameLoop)
        return
      }

      // Handle countdown (when AI is ready and game hasn't started yet)
      if (game.countdown !== null) {
        const countdownElapsed = (timestamp - game.countdownStartTime) / 1000
        const newCountdown = Math.max(0, 3 - Math.floor(countdownElapsed))

        if (newCountdown !== game.countdown) {
          game.countdown = newCountdown
          console.log('[COUNTDOWN]', game.countdown || 'GO!')
        }

        // Render static scene with countdown overlay
        renderGame(ctx, game)

        // Draw countdown overlay
        ctx.save()
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

        ctx.fillStyle = '#FFD700'
        ctx.font = 'bold 120px Arial'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'

        const countdownText = game.countdown > 0 ? game.countdown.toString() : 'GO!'
        ctx.fillText(countdownText, canvas.width / 2, canvas.height / 2)

        // Add stroke for better visibility
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 4
        ctx.strokeText(countdownText, canvas.width / 2, canvas.height / 2)

        ctx.restore()

        // When countdown reaches 0 and has shown "GO!" for 1 second, start the game
        if (game.countdown === 0 && countdownElapsed >= 4) {
          console.log('[COUNTDOWN] Finished! Starting game...')
          game.countdown = null
          game.countdownStartTime = null
          game.startTime = timestamp // Reset start time for accurate game timer
          game.lastTime = timestamp
        }

        game.animationFrame = requestAnimationFrame(gameLoop)
        return
      }

      // Calculate frame-rate independent delta (normalized to 60 FPS)
      // This ensures consistent speed across all devices
      const normalizedDelta = deltaTime / (1000 / 60) // 60 FPS baseline

      // Update elapsed time
      game.elapsedTime = (timestamp - game.startTime) / 1000 // Convert to seconds
      setGameTime(game.elapsedTime)

      // Update with deltaTime for frame-rate independence
      updateGame(game, normalizedDelta)

      // Render
      renderGame(ctx, game)

      // Collect gameplay data for score validation (ALWAYS when player is alive)
      // Also collect for AI training when not playing against AI
      if (game.player.isAlive) {
        // Calculate frame number based on elapsed time at 60 FPS (not loop iterations)
        // This ensures frame count matches time for backend validation
        frameNumberRef.current = Math.floor(game.elapsedTime * 60)

        // Extract state features
        const stateFeatures = extractStateFeatures(game)

        // Add player score to state for reward calculation
        const currentState = {
          ...stateFeatures,
          player_score: game.player.score
        }

        // Calculate RL-style reward
        const reward = calculateReward(game.prevState, currentState, game)

        // Store current state for next frame's reward calculation
        game.prevState = currentState

        // Determine actions from current keys pressed
        const actionLeft = !!(game.keys['ArrowLeft'] || game.keys['a'] || game.keys['A'])
        const actionRight = !!(game.keys['ArrowRight'] || game.keys['d'] || game.keys['D'])
        const actionJump = !!(game.keys['ArrowUp'] || game.keys[' '] || game.keys['w'] || game.keys['W'])
        const actionSprint = !!(game.keys['Shift'])

        // Create gameplay data point
        const dataPoint = {
          ...stateFeatures,
          action_left: actionLeft,
          action_right: actionRight,
          action_jump: actionJump,
          action_sprint: actionSprint,
          reward: reward,  // RL-style reward
          frame_number: frameNumberRef.current,
          difficulty: difficulty,
          timestamp: new Date().toISOString()
        }

        // ALWAYS add to gameplay recording for score validation
        allGameplayDataRef.current.push(dataPoint)

        // Only send to training endpoint when NOT playing against AI
        if (COLLECT_TRAINING_DATA && !enableAI) {
          // Add to batch for periodic sending to training endpoint
          trainingDataRef.current.push(dataPoint)

          // Send batch if it reaches the batch size
          if (trainingDataRef.current.length >= TRAINING_DATA_BATCH_SIZE) {
            const batchToSend = [...trainingDataRef.current]
            trainingDataRef.current = [] // Clear the batch
            sendTrainingDataBatch(batchToSend)
          }
        }
      }

      // Check game state
      const humanWon = game.player.checkGoalReached(game.map.goal)
      const humanLost = !game.player.isAlive
      const aiWon = enableAI && game.aiInstance && game.aiInstance.checkGoalReached()
      const aiLost = enableAI && game.aiInstance && !game.aiInstance.getAIPosition().isAlive

      if (humanWon || (enableAI && aiLost)) {
        const finalTime = game.elapsedTime
        setGameState('won')
        const aiPos = game.aiInstance ? game.aiInstance.getAIPosition() : null
        const finalStats = {
          score: game.player.score,
          distance: Math.floor(game.player.distance / DISTANCE_SCALE),
          aiScore: aiPos ? aiPos.score : 0,
          aiDistance: aiPos ? Math.floor(aiPos.distance / DISTANCE_SCALE) : 0,
          time: finalTime
        }
        setStats(finalStats)

        // Send remaining training data
        if (COLLECT_TRAINING_DATA && trainingDataRef.current.length > 0) {
          sendTrainingDataBatch(trainingDataRef.current, {
            game_outcome: 'won',
            final_time: finalTime,
            final_score: game.player.score,
            final_distance: Math.floor(game.player.distance)
          })
          trainingDataRef.current = []
        }

        // Notify parent component with completion data (raw distance for backend storage)
        if (onGameComplete) {
          // Copy gameplay data for score validation (ALWAYS collected)
          const gameplayRecording = [...allGameplayDataRef.current]

          onGameComplete({
            won: true,
            time: finalTime,
            score: game.player.score,
            distance: Math.floor(game.player.distance),
            jumps: game.player.jumpCount || 0,
            trainingData: gameplayRecording  // Include full gameplay recording
          })

          // Clear gameplay recording for next game
          allGameplayDataRef.current = []
        }
      } else if (humanLost || (enableAI && aiWon)) {
        const finalTime = game.elapsedTime
        setGameState('lost')
        const aiPos = game.aiInstance ? game.aiInstance.getAIPosition() : null
        const finalStats = {
          score: game.player.score,
          distance: Math.floor(game.player.distance / DISTANCE_SCALE),
          aiScore: aiPos ? aiPos.score : 0,
          aiDistance: aiPos ? Math.floor(aiPos.distance / DISTANCE_SCALE) : 0,
          time: finalTime
        }
        setStats(finalStats)

        // Send remaining training data
        if (COLLECT_TRAINING_DATA && trainingDataRef.current.length > 0) {
          sendTrainingDataBatch(trainingDataRef.current, {
            game_outcome: 'lost',
            final_time: finalTime,
            final_score: game.player.score,
            final_distance: Math.floor(game.player.distance)
          })
          trainingDataRef.current = []
        }

        // Notify parent component (raw distance for backend storage)
        if (onGameComplete) {
          // Copy gameplay data (ALWAYS collected, even for losses)
          const gameplayRecording = [...allGameplayDataRef.current]

          onGameComplete({
            won: false,
            time: finalTime,
            score: game.player.score,
            distance: Math.floor(game.player.distance),
            jumps: game.player.jumpCount || 0,
            trainingData: gameplayRecording  // Include full gameplay recording
          })

          // Clear gameplay recording for next game
          allGameplayDataRef.current = []
        }
      } else {
        game.animationFrame = requestAnimationFrame(gameLoop)
      }
    }

    // Start game loop
    console.log('[GAME] Starting game loop...')
    const now = performance.now()
    game.lastTime = now
    game.startTime = now
    console.log('[GAME] Game loop started at:', now)
    game.animationFrame = requestAnimationFrame(gameLoop)

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleBlur)
      canvas.removeEventListener('blur', handleBlur)
      if (game.animationFrame) {
        cancelAnimationFrame(game.animationFrame)
      }
      // Cleanup AI instance
      if (game.aiInstance) {
        game.aiInstance.dispose()
      }
    }
  }, [])

  const updateGame = (game, deltaTime = 1) => {
    const { player, keys, map } = game

    if (!player.isAlive) return

    // Handle input - always reset movement first
    player.stopMovement()

    // Check which keys are currently pressed (must exist in keys object)
    const leftPressed = ('ArrowLeft' in keys) || ('a' in keys) || ('A' in keys)
    const rightPressed = ('ArrowRight' in keys) || ('d' in keys) || ('D' in keys)
    const jumpPressed = (' ' in keys) || ('ArrowUp' in keys) || ('w' in keys) || ('W' in keys)
    const sprintPressed = ('Shift' in keys)
    const duckPressed = ('ArrowDown' in keys) || ('s' in keys) || ('S' in keys)

    // Safety: If jump flag is set but no jump key is pressed, reset it
    // This prevents the flag from getting stuck
    if (game.jumpKeyWasPressed && !jumpPressed) {
      game.jumpKeyWasPressed = false
    }

    // Movement - only move if one direction is pressed
    if (leftPressed && !rightPressed) {
      player.moveLeft()
    } else if (rightPressed && !leftPressed) {
      player.moveRight()
    }

    // Jump - only jump once per key press (not continuous while held)
    if (jumpPressed && !game.jumpKeyWasPressed) {
      console.log('[JUMP] Attempting jump - isOnGround:', player.isOnGround, 'velocityY:', player.velocityY, 'y:', player.y)
      player.jump()
      game.jumpKeyWasPressed = true
    }

    player.sprint(sprintPressed)
    player.duck(duckPressed)

    // Update player physics with deltaTime for frame-rate independence
    player.update(map.platforms, deltaTime)

    // Update camera (smooth follow) - scaled by deltaTime for consistent smoothing
    const targetCameraX = player.x - 400
    game.cameraX += (targetCameraX - game.cameraX) * 0.1 * deltaTime

    // Check coin collection
    map.coins.forEach(coin => {
      player.collectCoin(coin)
    })

    // Update enemies with deltaTime
    map.enemies.forEach(enemy => {
      enemy.x += enemy.direction * enemy.speed * deltaTime

      // Simple AI: turn around at platform edges
      const onPlatform = map.platforms.find(p =>
        enemy.x >= p.x && enemy.x <= p.x + p.width &&
        Math.abs(enemy.y + enemy.height - p.y) < 5
      )

      if (!onPlatform || enemy.x < 0 || enemy.x > map.width) {
        enemy.direction *= -1
      }

      // Check collision with player
      if (player.checkEnemyCollision(enemy)) {
        player.die()
      }
    })

    // Update AI instance (runs in separate isolated environment)
    if (enableAI && game.aiInstance) {
      game.aiInstance.update(deltaTime)
    }

    // Update stats display
    const aiPos = game.aiInstance ? game.aiInstance.getAIPosition() : null
    setStats({
      score: player.score,
      distance: Math.floor(player.distance / DISTANCE_SCALE),
      aiScore: aiPos ? aiPos.score : 0,
      aiDistance: aiPos ? Math.floor(aiPos.distance / DISTANCE_SCALE) : 0,
      time: game.elapsedTime || 0
    })
  }

  const renderGame = (ctx, game) => {
    const { player, map, cameraX } = game
    const canvas = ctx.canvas

    // Clear canvas
    ctx.fillStyle = '#87CEEB' // Sky blue
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Save context
    ctx.save()
    ctx.translate(-cameraX, 0)

    // Draw platforms
    map.platforms.forEach(platform => {
      // Color based on platform type
      if (platform.type === 'ground') {
        ctx.fillStyle = '#8B4513'  // Brown for ground
      } else if (platform.type === 'ice') {
        ctx.fillStyle = '#4DA6FF'  // Light blue for ice (slippery)
      } else {
        ctx.fillStyle = '#2E8B57'  // Green for normal platforms
      }
      ctx.fillRect(platform.x, platform.y, platform.width, platform.height)

      // Platform outline
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(platform.x, platform.y, platform.width, platform.height)
    })

    // Draw coins
    map.coins.forEach(coin => {
      if (!coin.collected) {
        ctx.fillStyle = '#FFD700'
        ctx.beginPath()
        ctx.arc(coin.x, coin.y, coin.radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = '#FFA500'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    // Draw enemies
    map.enemies.forEach(enemy => {
      ctx.fillStyle = '#FF00FF'
      ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height)
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(enemy.x, enemy.y, enemy.width, enemy.height)
    })

    // Draw goal
    ctx.fillStyle = '#00FF00'
    ctx.fillRect(map.goal.x, map.goal.y, map.goal.width, map.goal.height)
    ctx.strokeStyle = '#006400'
    ctx.lineWidth = 3
    ctx.strokeRect(map.goal.x, map.goal.y, map.goal.width, map.goal.height)

    // Goal flag
    ctx.fillStyle = '#FFD700'
    ctx.font = 'bold 60px Arial'
    ctx.fillText('🏁', map.goal.x + 5, map.goal.y - 10)

    // Draw player (human)
    ctx.fillStyle = playerColor || '#4CAF50' // Use custom color or default green
    ctx.fillRect(player.x, player.y, player.width, player.height)
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 2
    ctx.strokeRect(player.x, player.y, player.width, player.height)

    // Draw player label
    ctx.fillStyle = '#FFFFFF'
    ctx.font = 'bold 14px Arial'
    ctx.fillText('YOU', player.x + 2, player.y - 5)

    // Draw AI player if enabled (from separate AI game instance)
    if (enableAI && game.aiInstance) {
      const aiPos = game.aiInstance.getAIPosition()
      if (aiPos && aiPos.isAlive) {
        ctx.fillStyle = '#FF6B6B' // Red for AI
        ctx.fillRect(aiPos.x, aiPos.y, aiPos.width, aiPos.height)
        ctx.strokeStyle = '#000'
        ctx.lineWidth = 2
        ctx.strokeRect(aiPos.x, aiPos.y, aiPos.width, aiPos.height)

        // Draw AI label
        ctx.fillStyle = '#FFFFFF'
        ctx.font = 'bold 14px Arial'
        ctx.fillText('AI', aiPos.x + 6, aiPos.y - 5)
      }
    }

    // Restore context
    ctx.restore()

    // Helper to format time
    const formatTime = (seconds) => {
      const mins = Math.floor(seconds / 60)
      const secs = Math.floor(seconds % 60)
      return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    // Draw UI (fixed position)
    if (enableAI && game.aiInstance) {
      const aiPos = game.aiInstance.getAIPosition()

      // Split view: Human on left, AI on right
      ctx.fillStyle = 'rgba(76, 175, 80, 0.9)' // Green for human
      ctx.fillRect(10, 10, 300, 120)

      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 18px Arial'
      ctx.fillText('YOU (Human)', 20, 35)
      ctx.font = 'bold 16px Arial'
      ctx.fillText(`Score: ${player.score}`, 20, 60)
      ctx.fillText(`Distance: ${Math.floor(player.distance / DISTANCE_SCALE)}m`, 20, 85)
      ctx.fillText(`Time: ${formatTime(game.elapsedTime || 0)}`, 20, 110)

      ctx.fillStyle = 'rgba(255, 107, 107, 0.9)' // Red for AI
      ctx.fillRect(canvas.width - 310, 10, 300, 120)

      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 18px Arial'
      ctx.fillText('AI Opponent', canvas.width - 290, 35)
      ctx.font = 'bold 16px Arial'
      ctx.fillText(`Score: ${aiPos ? aiPos.score : 0}`, canvas.width - 290, 60)
      ctx.fillText(`Distance: ${aiPos ? Math.floor(aiPos.distance / DISTANCE_SCALE) : 0}m`, canvas.width - 290, 85)

      // Show episode number if playing against episode model
      if (playingEpisode !== null) {
        ctx.font = '14px Arial'
        ctx.fillStyle = '#FFD700' // Gold color for episode indicator
        ctx.fillText(`Episode ${playingEpisode}`, canvas.width - 290, 108)
      }

      // AI status indicator (use ref for game loop access)
      const currentAiStatus = aiStatusRef.current
      if (currentAiStatus === 'loading') {
        ctx.fillStyle = 'rgba(255, 193, 7, 0.9)'
        ctx.fillRect(canvas.width - 310, 120, 300, 40)
        ctx.fillStyle = '#000'
        ctx.font = '14px Arial'
        ctx.fillText('AI Loading...', canvas.width - 290, 145)
      } else if (currentAiStatus === 'ready') {
        ctx.fillStyle = 'rgba(76, 175, 80, 0.9)' // Green
        ctx.fillRect(canvas.width - 310, 120, 300, 40)
        ctx.fillStyle = '#FFF'
        ctx.font = '14px Arial'
        ctx.fillText('🤖 AI Ready', canvas.width - 290, 145)
      } else if (currentAiStatus === 'error') {
        ctx.fillStyle = 'rgba(244, 67, 54, 0.9)'
        ctx.fillRect(canvas.width - 310, 120, 300, 40)
        ctx.fillStyle = '#FFF'
        ctx.font = '14px Arial'
        ctx.fillText('AI Error - Playing Solo', canvas.width - 290, 145)
      }
    } else {
      // Solo play
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(10, 10, 300, 105)

      ctx.fillStyle = '#FFD700'
      ctx.font = 'bold 20px Arial'
      ctx.fillText(`Score: ${player.score}`, 20, 40)
      ctx.fillText(`Distance: ${Math.floor(player.distance / DISTANCE_SCALE)}m`, 20, 70)
      ctx.fillText(`Time: ${formatTime(game.elapsedTime || 0)}`, 20, 95)
    }

    // Controls hint - Bottom left instead of top right
    const controlsY = canvas.height - 95 // 95px from bottom
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(10, controlsY, 300, 85)
    ctx.fillStyle = '#FFFFFF'
    ctx.font = '16px Arial'
    ctx.fillText('← → : Move', 20, controlsY + 25)
    ctx.fillText('Space: Jump', 20, controlsY + 50)
    ctx.fillText('Shift: Sprint', 20, controlsY + 75)
  }

  const handlePlayAgain = () => {
    // Go back to intro screen to start a new game
    onGameEnd()
  }

  const handleBackToHome = () => {
    // Navigate to home page
    navigate('/')
  }

  // Helper function to extract state features for training data
  const extractStateFeatures = (game) => {
    const { player, map } = game
    const platforms = map.platforms
    const enemies = map.enemies || []

    // Find nearest platform below (for landing prediction)
    let nearestBelow = null
    let minDistBelow = Infinity
    for (const platform of platforms) {
      if (platform.y > player.y + player.height && platform.x < player.x + 500 && platform.x + platform.width > player.x - 100) {
        const dist = Math.abs(platform.x - player.x) + (platform.y - player.y)
        if (dist < minDistBelow) {
          minDistBelow = dist
          nearestBelow = platform
        }
      }
    }

    // Find nearest platform ahead (for jumping prediction)
    let nearestAhead = null
    let minDistAhead = Infinity
    for (const platform of platforms) {
      if (platform.x > player.x && platform.x < player.x + 600) {
        const dist = Math.abs(platform.x - player.x) + Math.abs(platform.y - player.y)
        if (dist < minDistAhead) {
          minDistAhead = dist
          nearestAhead = platform
        }
      }
    }

    // Find nearest enemy ahead (for avoidance)
    let nearestEnemy = null
    let minDistEnemy = Infinity
    for (const enemy of enemies) {
      // Only consider enemies ahead of player and within reasonable range
      if (enemy.x > player.x - 100 && enemy.x < player.x + 800) {
        const dist = Math.abs(enemy.x - player.x) + Math.abs(enemy.y - player.y)
        if (dist < minDistEnemy) {
          minDistEnemy = dist
          nearestEnemy = enemy
        }
      }
    }

    return {
      player_x: player.x,
      player_y: player.y,
      player_vx: player.velocityX,
      player_vy: player.velocityY,
      player_on_ground: player.isOnGround,
      platform_below_x: nearestBelow ? nearestBelow.x - player.x : null,
      platform_below_y: nearestBelow ? nearestBelow.y - player.y : null,
      platform_below_is_ice: nearestBelow ? nearestBelow.type === 'ice' : false,
      platform_ahead_x: nearestAhead ? nearestAhead.x - player.x : null,
      platform_ahead_y: nearestAhead ? nearestAhead.y - player.y : null,
      platform_ahead_is_ice: nearestAhead ? nearestAhead.type === 'ice' : false,
      enemy_x: nearestEnemy ? nearestEnemy.x - player.x : null,
      enemy_y: nearestEnemy ? nearestEnemy.y - player.y : null,
      goal_x: map.goal.x - player.x,
      goal_y: map.goal.y - player.y
    }
  }

  // Helper function to calculate RL-style reward (matches backend RL environment)
  const calculateReward = (prevState, currentState, game) => {
    const { player, map } = game
    let reward = 0.0

    // RL reward weights (matches environment.py)
    const REWARD_WEIGHTS = {
      progress: 0.1,
      coin: 10.0,
      goal: 1000.0,
      death: -100.0,
      time: -0.01
    }

    // Progress reward (encourage moving right)
    if (prevState) {
      const distanceDelta = currentState.player_x - prevState.player_x
      reward += distanceDelta * REWARD_WEIGHTS.progress
    }

    // Time penalty (encourage efficiency)
    reward += REWARD_WEIGHTS.time

    // Coin collection (check if score increased)
    if (prevState && player.score > (prevState.player_score || 0)) {
      reward += REWARD_WEIGHTS.coin
    }

    // Goal reached (big reward)
    if (player.checkGoalReached(map.goal)) {
      reward += REWARD_WEIGHTS.goal
    }

    // Death penalty
    if (!player.isAlive) {
      reward += REWARD_WEIGHTS.death
    }

    return reward
  }

  // Helper function to send training data batch to backend
  const sendTrainingDataBatch = async (batch, sessionMetadata = {}) => {
    if (batch.length === 0) return

    try {
      const trainingBatch = {
        session_id: sessionIdRef.current,
        username: username || 'Anonymous',
        difficulty: difficulty,
        data_points: batch,
        session_metadata: sessionMetadata
      }

      await rlAPI.submitTrainingData(trainingBatch)
      console.log(`[Training Data] Sent ${batch.length} data points to server`)
    } catch (error) {
      console.error('[Training Data] Failed to send batch:', error)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(1)
    return mins > 0 ? `${mins}:${secs.padStart(4, '0')}` : `${secs}s`
  }

  return (
    <div className="game-canvas-wrapper">
      <canvas
        ref={canvasRef}
        width={1920}
        height={1080}
        className="game-canvas"
      />

      {gameState !== 'playing' && (
        <div className="game-overlay">
          <div className="game-result">
            <h2>{enableAI ? (stats.distance > stats.aiDistance ? '🎉 You Win!' : '🤖 AI Wins!') : (gameState === 'won' ? '🎉 You Win!' : '💀 Game Over')}</h2>

            {enableAI ? (
              <div className="final-stats-race">
                <div className="player-stats human-stats">
                  <h3>🧑 You</h3>
                  <p>Score: <strong>{stats.score}</strong></p>
                  <p>Distance: <strong>{stats.distance}m</strong></p>
                  <p>Time: <strong>{formatTime(stats.time)}</strong></p>
                  <p className={stats.distance > stats.aiDistance ? 'winner' : ''}>
                    {stats.distance > stats.aiDistance ? '👑 Winner' : ''}
                  </p>
                </div>
                <div className="vs-divider">VS</div>
                <div className="player-stats ai-stats">
                  <h3>🤖 AI</h3>
                  <p>Score: <strong>{stats.aiScore}</strong></p>
                  <p>Distance: <strong>{stats.aiDistance}m</strong></p>
                  <p>Time: <strong>{formatTime(stats.time)}</strong></p>
                  <p className={stats.aiDistance > stats.distance ? 'winner' : ''}>
                    {stats.aiDistance > stats.distance ? '👑 Winner' : ''}
                  </p>
                </div>
              </div>
            ) : (
              <div className="final-stats">
                <p>Score: <strong>{stats.score}</strong></p>
                <p>Distance: <strong>{stats.distance}m</strong></p>
                <p>Time: <strong>{formatTime(stats.time)}</strong></p>
              </div>
            )}

            <div className="game-buttons">
              <button onClick={handlePlayAgain} className="restart-btn">
                🔄 Race Again
              </button>
              <button onClick={handleBackToHome} className="menu-btn">
                🏠 Back to Menu
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI Loading Progress Overlay */}
      {aiStatus === 'loading' && (
        <div className="game-overlay loading-overlay">
          <div className="loading-container">
            <h2>🤖 Loading AI Opponent</h2>
            <div className="loading-progress-wrapper">
              <div className="loading-progress-bar">
                <div
                  className="loading-progress-fill"
                  style={{ width: `${loadingProgress}%` }}
                />
              </div>
              <div className="loading-percentage">{Math.round(loadingProgress)}%</div>
            </div>
            <p className="loading-message">{loadingMessage}</p>
            <div className="loading-spinner">⏳</div>
          </div>
        </div>
      )}
    </div>
  )
}
