"""
RL Platformer Router
Proxies requests to RL backend
"""

from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import subprocess
import os
import sys
import json
import asyncio
import time
import signal
from pathlib import Path
import bcrypt
import jwt
from datetime import datetime, timedelta
import uuid

router = APIRouter(prefix="/rl", tags=["RL Platformer"])

# Global training process tracking
training_process: Optional[subprocess.Popen] = None
training_pid: Optional[int] = None

# Find RL backend path
RL_BACKEND_PATH = None
for possible_path in [
    Path(__file__).parent.parent.parent.parent / "RL" / "backend",
    Path(__file__).parent.parent.parent / "RL" / "backend",
]:
    if possible_path.exists():
        RL_BACKEND_PATH = possible_path
        break

if RL_BACKEND_PATH:
    sys.path.insert(0, str(RL_BACKEND_PATH))

# Launcher backend path (for BC models stored in launcher/backend/models)
LAUNCHER_BACKEND_PATH = Path(__file__).parent.parent


class TrainingStatus(BaseModel):
    """Training status response"""
    is_training: bool
    current_step: int
    total_steps: int
    best_reward: float
    avg_reward: float


class ModelInfo(BaseModel):
    """Model information"""
    is_loaded: bool
    model_path: Optional[str]
    input_shape: Optional[tuple]
    num_actions: int


class ScoreEntry(BaseModel):
    """Score entry for leaderboard"""
    name: str
    time: float
    score: int
    distance: int
    difficulty: str = 'easy'  # easy, medium, hard
    completion_token: str  # Required: proof that player reached the goal
    training_data: List[Dict[str, Any]]  # Required: gameplay recording for validation
    timestamp: Optional[str] = None
    date: Optional[str] = None


class UserRegister(BaseModel):
    """User registration request"""
    username: str
    password: str


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class UserProfileUpdate(BaseModel):
    """User profile update request"""
    current_password: Optional[str] = None  # Required for password change
    new_password: Optional[str] = None
    new_username: Optional[str] = None
    player_color: Optional[str] = None  # Hex color code


class AdminAuth(BaseModel):
    """Admin authentication for sensitive operations"""
    admin_password: str


class GameMetrics(BaseModel):
    """Game metrics for a single session"""
    jumps: int
    points: int
    distance: int
    time_played: float


class TrainingDataPoint(BaseModel):
    """Single state-action pair for AI training"""
    # Player state
    player_x: float
    player_y: float
    player_vx: float
    player_vy: float
    player_on_ground: bool

    # Nearest platforms (relative positions)
    platform_below_x: Optional[float] = None  # Relative X to nearest platform below
    platform_below_y: Optional[float] = None  # Relative Y to nearest platform below
    platform_below_is_ice: bool = False  # Is platform below ice (slippery)?
    platform_ahead_x: Optional[float] = None  # Relative X to nearest platform ahead
    platform_ahead_y: Optional[float] = None  # Relative Y to nearest platform ahead
    platform_ahead_is_ice: bool = False  # Is platform ahead ice (slippery)?

    # Nearest enemy (relative position)
    enemy_x: Optional[float] = None  # Relative X to nearest enemy
    enemy_y: Optional[float] = None  # Relative Y to nearest enemy

    # Goal position (relative)
    goal_x: float
    goal_y: float

    # Action taken by human
    action_left: bool
    action_right: bool
    action_jump: bool
    action_sprint: bool

    # RL-style reward for this transition (optional, for hybrid training)
    reward: Optional[float] = None  # Immediate reward for this state-action transition

    # Metadata
    frame_number: int
    difficulty: str
    timestamp: str


class TrainingDataBatch(BaseModel):
    """Batch of training data from a gameplay session"""
    session_id: str
    username: str
    difficulty: str
    data_points: List[TrainingDataPoint]
    session_metadata: dict = {}


# ===== Helper Functions =====

def get_client_ip(request: Request) -> str:
    """
    Extract real client IP address from request
    Handles proxies and load balancers by checking X-Forwarded-For header
    """
    # Check if behind a proxy (X-Forwarded-For header)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
        # First IP is the original client
        return forwarded_for.split(",")[0].strip()

    # Check X-Real-IP header (used by some proxies)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"


def get_client_info(request: Request) -> dict:
    """
    Extract client information including IP and User Agent
    User Agent provides: browser, OS, device type - helps identify unique devices on shared IPs

    Returns:
        dict with 'ip' and 'user_agent' keys
    """
    return {
        'ip': get_client_ip(request),
        'user_agent': request.headers.get("User-Agent", "unknown")
    }


def log_activity(activity_type: str, details: dict):
    """
    Log suspicious activity to audit log file

    Args:
        activity_type: Type of activity (e.g., "score_submission", "failed_login")
        details: Dict with activity details (ip, username, etc.)
    """
    from datetime import datetime
    import json

    log_dir = DATA_DIR / "audit_logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"activity_{datetime.utcnow().strftime('%Y-%m')}.json"

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": activity_type,
        **details
    }

    # Append to monthly log file
    try:
        logs = []
        if log_file.exists() and log_file.stat().st_size > 0:
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                # File is empty or corrupted, start fresh
                logs = []

        logs.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Failed to log activity: {e}")


# ===== Routes =====

@router.get("/", summary="RL API Info")
def rl_info():
    """Get RL API information"""
    return {
        "project": "RL Platformer",
        "description": "Reinforcement Learning side-scrolling platformer - Human vs AI racing",
        "features": [
            "Procedurally generated levels",
            "Visual-based RL agent",
            "Human vs AI racing",
            "1920x1080 HD gameplay",
            "PyTorch GPU training"
        ],
        "endpoints": [
            "/rl/status - Check if model exists",
            "/rl/training/status - Get training status",
            "/rl/training/start - Start training",
            "/rl/training/stop - Stop training",
            "/rl/model/info - Get model information",
            "/rl/model/export - Export model to TensorFlow.js"
        ]
    }


@router.get("/debug/paths", summary="Debug Path Detection")
def debug_paths():
    """Debug endpoint to show path detection"""
    file_path = Path(__file__)
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "RL" / "backend",
        Path(__file__).parent.parent.parent / "RL" / "backend",
    ]

    return {
        "current_file": str(file_path),
        "parent": str(file_path.parent),
        "parent.parent": str(file_path.parent.parent),
        "parent.parent.parent": str(file_path.parent.parent.parent),
        "parent.parent.parent.parent": str(file_path.parent.parent.parent.parent),
        "RL_BACKEND_PATH": str(RL_BACKEND_PATH) if RL_BACKEND_PATH else None,
        "RL_BACKEND_PATH_exists": RL_BACKEND_PATH.exists() if RL_BACKEND_PATH else False,
        "possible_paths_checked": [
            {
                "path": str(p),
                "exists": p.exists()
            }
            for p in possible_paths
        ]
    }


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/auth/register", summary="Register New User")
def register_user(user_data: UserRegister, request: Request):
    """
    Register a new user account
    Returns JWT token on success
    Logs IP address and User Agent for security monitoring
    """
    client_info = get_client_info(request)
    users = load_users()

    # Check if username already exists
    if any(u['username'].lower() == user_data.username.lower() for u in users):
        log_activity("registration_failed", {
            **client_info,
            "username": user_data.username,
            "reason": "username_already_exists"
        })
        raise HTTPException(status_code=400, detail="Username already exists")

    # Validate username (3-20 chars, alphanumeric + underscore)
    if not user_data.username or len(user_data.username) < 3 or len(user_data.username) > 20:
        raise HTTPException(status_code=400, detail="Username must be 3-20 characters")

    # Validate password (minimum 6 characters)
    if not user_data.password or len(user_data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Create new user
    user_id = str(uuid.uuid4())
    new_user = {
        "user_id": user_id,
        "username": user_data.username,
        "password_hash": hash_password(user_data.password),
        "created_at": datetime.utcnow().isoformat(),
        "player_color": "#4287f5",  # Default blue color
        "total_games": 0,
        "total_jumps": 0,
        "total_points": 0,
        "total_distance": 0,
        "total_playtime": 0.0,
        "registration_ip": client_info['ip'],  # Store registration IP
        "registration_user_agent": client_info['user_agent']  # Store registration device
    }

    users.append(new_user)
    save_users(users)

    # Log successful registration
    log_activity("user_registered", {
        **client_info,
        "username": user_data.username,
        "user_id": user_id
    })

    # Generate JWT token
    token = create_jwt_token(user_id, user_data.username)

    return {
        "status": "success",
        "message": f"User {user_data.username} registered successfully",
        "token": token,
        "user": {
            "user_id": user_id,
            "username": user_data.username,
            "created_at": new_user["created_at"],
            "player_color": new_user["player_color"]
        }
    }


@router.post("/auth/login", summary="User Login")
def login_user(user_data: UserLogin, request: Request):
    """
    Login with username and password
    Returns JWT token on success
    Logs IP address and User Agent for security monitoring
    """
    client_info = get_client_info(request)
    users = load_users()

    # Find user
    user = next((u for u in users if u['username'].lower() == user_data.username.lower()), None)

    if not user:
        # Log failed login attempt
        log_activity("login_failed", {
            **client_info,
            "username": user_data.username,
            "reason": "user_not_found"
        })
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    if not verify_password(user_data.password, user['password_hash']):
        # Log failed login attempt
        log_activity("login_failed", {
            **client_info,
            "username": user_data.username,
            "reason": "invalid_password"
        })
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Log successful login
    log_activity("user_login", {
        **client_info,
        "username": user['username'],
        "user_id": user['user_id']
    })

    # Generate JWT token
    token = create_jwt_token(user['user_id'], user['username'])

    return {
        "status": "success",
        "message": f"Logged in as {user['username']}",
        "token": token,
        "user": {
            "user_id": user['user_id'],
            "username": user['username'],
            "created_at": user['created_at'],
            "player_color": user.get('player_color', '#4287f5')
        }
    }


@router.get("/auth/verify", summary="Verify Token")
def verify_token(authorization: Optional[str] = Header(None)):
    """
    Verify JWT token and get current user info
    """
    user = get_current_user(authorization)

    return {
        "status": "valid",
        "user": {
            "user_id": user['user_id'],
            "username": user['username'],
            "created_at": user['created_at'],
            "player_color": user.get('player_color', '#4287f5'),
            "stats": {
                "total_games": user.get('total_games', 0),
                "total_jumps": user.get('total_jumps', 0),
                "total_points": user.get('total_points', 0),
                "total_distance": user.get('total_distance', 0),
                "total_playtime": user.get('total_playtime', 0.0)
            }
        }
    }


@router.put("/auth/profile", summary="Update User Profile")
def update_profile(profile_data: UserProfileUpdate, authorization: Optional[str] = Header(None)):
    """
    Update user profile (username, password, player color)
    Requires authentication
    """
    user = get_current_user(authorization)
    users = load_users()
    scores = load_scores()

    # Find user in list
    user_index = next((i for i, u in enumerate(users) if u['user_id'] == user['user_id']), None)
    if user_index is None:
        raise HTTPException(status_code=404, detail="User not found")

    updated_fields = []

    # Update password (requires current password verification)
    if profile_data.new_password:
        if not profile_data.current_password:
            raise HTTPException(status_code=400, detail="Current password required to change password")

        if not verify_password(profile_data.current_password, user['password_hash']):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

        if len(profile_data.new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

        users[user_index]['password_hash'] = hash_password(profile_data.new_password)
        updated_fields.append("password")

    # Update username (updates all associated scores)
    if profile_data.new_username:
        # Validate new username
        if len(profile_data.new_username) < 3 or len(profile_data.new_username) > 20:
            raise HTTPException(status_code=400, detail="Username must be 3-20 characters")

        # Check if username already taken by another user
        if any(u['username'].lower() == profile_data.new_username.lower() and u['user_id'] != user['user_id'] for u in users):
            raise HTTPException(status_code=400, detail="Username already taken")

        old_username = users[user_index]['username']
        users[user_index]['username'] = profile_data.new_username

        # Update username in all scores
        for score in scores:
            if score.get('user_id') == user['user_id']:
                score['name'] = profile_data.new_username

        save_scores(scores)
        updated_fields.append(f"username (from '{old_username}' to '{profile_data.new_username}')")

    # Update player color
    if profile_data.player_color:
        # Validate hex color format
        import re
        if not re.match(r'^#[0-9A-Fa-f]{6}$', profile_data.player_color):
            raise HTTPException(status_code=400, detail="Invalid color format. Use hex format like #4287f5")

        users[user_index]['player_color'] = profile_data.player_color
        updated_fields.append("player color")

    # Save updated users
    save_users(users)

    return {
        "status": "success",
        "message": f"Profile updated: {', '.join(updated_fields)}",
        "user": {
            "user_id": users[user_index]['user_id'],
            "username": users[user_index]['username'],
            "player_color": users[user_index].get('player_color', '#4287f5'),
            "created_at": users[user_index]['created_at']
        }
    }


@router.get("/status", summary="Check RL Status")
def check_status():
    """Check if RL backend is available and model exists"""
    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip" if RL_BACKEND_PATH else None
    tfjs_model = RL_BACKEND_PATH / "models" / "tfjs_model" / "model.json" if RL_BACKEND_PATH else None

    # Check for any ONNX models
    onnx_models_exist = False
    if RL_BACKEND_PATH:
        models_dir = RL_BACKEND_PATH / "models"
        if models_dir.exists():
            # Check for any episode_*_onnx directories with model.onnx
            onnx_models_exist = any((models_dir / d / "model.onnx").exists()
                                   for d in models_dir.glob("episode_*_onnx"))

    any_model_exists = (tfjs_model.exists() if tfjs_model else False) or onnx_models_exist

    return {
        "backend_available": RL_BACKEND_PATH is not None,
        "backend_path": str(RL_BACKEND_PATH) if RL_BACKEND_PATH else None,
        "model_exists": model_path.exists() if model_path else False,
        "tfjs_model_exists": tfjs_model.exists() if tfjs_model else False,
        "onnx_model_exists": onnx_models_exist,
        "any_model_exists": any_model_exists,
        "gpu_available": check_gpu_available(),
        "ready_for_training": RL_BACKEND_PATH is not None,
        "ready_for_gameplay": any_model_exists
    }


@router.get("/training/status", summary="Get Training Status")
def get_training_status():
    """Get current training status from status.json file"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    status_file = RL_BACKEND_PATH / "training" / "status.json"

    # Check if process is still running
    global training_process, training_pid
    is_process_alive = False
    if training_process and training_process.poll() is None:
        is_process_alive = True
    elif training_pid:
        # Check if PID exists
        try:
            os.kill(training_pid, 0)
            is_process_alive = True
        except (OSError, ProcessLookupError):
            is_process_alive = False

    # Read status file if it exists
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
                status['process_alive'] = is_process_alive
                status['pid'] = training_pid

                # Safety: ensure best_reward is JSON-safe (convert None to 0.0 for display)
                if status.get('best_reward') is None:
                    status['best_reward'] = 0.0

                return status
        except Exception as e:
            return {
                "is_training": is_process_alive,
                "error": f"Failed to read status file: {str(e)}",
                "process_alive": is_process_alive,
                "pid": training_pid
            }
    else:
        return {
            "is_training": is_process_alive,
            "current_step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "episodes": 0,
            "avg_reward": 0.0,
            "best_reward": 0.0,
            "message": "Training not started or status file not created yet",
            "process_alive": is_process_alive,
            "pid": training_pid
        }


@router.post("/training/start", summary="Start Training")
def start_training(
    training_mode: str = "reinforcement_learning",
    timesteps: int = 1000000,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    bc_epochs: int = 50,
    rl_timesteps: int = 500000
):
    """
    Start agent training

    Args:
        training_mode: "reinforcement_learning", "behavioral_cloning", or "hybrid" (BC + RL)
        timesteps: Number of timesteps for RL training
        epochs: Number of epochs for BC training
        batch_size: Batch size for BC training
        learning_rate: Learning rate for BC training
        val_split: Validation split ratio for BC training
        bc_epochs: Number of BC epochs for hybrid training
        rl_timesteps: Number of RL timesteps for hybrid training
    """
    global training_process, training_pid

    # Check if training is already running
    if training_process and training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training is already running")

    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Select training script based on mode
    if training_mode == "hybrid":
        train_script = RL_BACKEND_PATH / "training" / "train_hybrid.py"
        if not train_script.exists():
            raise HTTPException(status_code=500, detail="Hybrid training script not found")

        # Check if training data exists (needed for BC phase)
        training_data_dir = Path(__file__).parent.parent / "data" / "training_data"
        if not training_data_dir.exists() or not any(training_data_dir.rglob("*.json")):
            raise HTTPException(
                status_code=400,
                detail="No training data found. Hybrid training requires human gameplay data for BC phase."
            )

        # Build command for hybrid training
        cmd_args = [
            sys.executable, str(train_script),
            "--bc-epochs", str(bc_epochs),
            "--rl-timesteps", str(rl_timesteps),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate)
        ]

        message = f"Hybrid training started: {bc_epochs} BC epochs + {rl_timesteps} RL timesteps"
        estimated_time = (bc_epochs / 10) + (rl_timesteps / 50000)  # BC time + RL time

    elif training_mode == "behavioral_cloning":
        train_script = RL_BACKEND_PATH / "training" / "train_behavioral_cloning.py"
        if not train_script.exists():
            raise HTTPException(status_code=500, detail="Behavioral cloning training script not found")

        # Check if training data exists
        training_data_dir = Path(__file__).parent.parent / "data" / "training_data"
        if not training_data_dir.exists() or not any(training_data_dir.rglob("*.json")):
            raise HTTPException(
                status_code=400,
                detail="No training data found. Play the game in solo mode to collect training data first."
            )

        # Build command for behavioral cloning
        cmd_args = [
            sys.executable, str(train_script),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--val-split", str(val_split)
        ]

        message = f"Behavioral cloning training started with {epochs} epochs"
        estimated_time = epochs / 10  # Rough estimate: ~6 minutes per 10 epochs

    else:  # reinforcement_learning
        train_script = RL_BACKEND_PATH / "training" / "train_agent.py"
        if not train_script.exists():
            raise HTTPException(status_code=500, detail="RL training script not found")

        # Build command for RL
        cmd_args = [
            sys.executable, str(train_script),
            "--timesteps", str(timesteps)
        ]

        message = f"RL training started with {timesteps} timesteps"
        estimated_time = timesteps / 50000  # Rough estimate for RL

    try:
        # Clear old status file
        status_file = RL_BACKEND_PATH / "training" / "status.json"
        if status_file.exists():
            status_file.unlink()

        # Start process in background
        # Use DEVNULL instead of PIPE to avoid Windows asyncio connection errors
        training_process = subprocess.Popen(
            cmd_args,
            cwd=str(RL_BACKEND_PATH / "training"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )

        training_pid = training_process.pid

        return {
            "status": "started",
            "pid": training_pid,
            "training_mode": training_mode,
            "message": message,
            "estimated_time_hours": estimated_time,
            "note": "Training is running in the background. Check /training/status or /training/stream for progress."
        }
    except Exception as e:
        training_process = None
        training_pid = None
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/training/stop", summary="Stop Training")
def stop_training():
    """Stop current training process gracefully"""
    global training_process, training_pid

    if not training_process and not training_pid:
        raise HTTPException(status_code=400, detail="No training process is running")

    try:
        # Try to terminate gracefully
        if training_process and training_process.poll() is None:
            # On Windows, send CTRL_BREAK_EVENT
            if os.name == 'nt':
                training_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                training_process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                training_process.wait(timeout=10)
                message = "Training stopped gracefully"
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                training_process.kill()
                training_process.wait()
                message = "Training forcefully terminated (did not stop gracefully)"
        elif training_pid:
            # Try to kill by PID
            try:
                if os.name == 'nt':
                    os.kill(training_pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.kill(training_pid, signal.SIGTERM)
                message = "Training stop signal sent to PID"
            except (OSError, ProcessLookupError):
                message = "Training process not found (may have already stopped)"

        # Clear tracking variables
        training_process = None
        training_pid = None

        return {
            "status": "stopped",
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


@router.get("/training/stream", summary="Stream Training Status (SSE)")
async def stream_training_status():
    """Stream training status updates using Server-Sent Events"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    status_file = RL_BACKEND_PATH / "training" / "status.json"

    async def event_generator():
        """Generate SSE events with training status updates"""
        last_status = None
        no_file_count = 0

        try:
            while True:
                try:
                    # Check if training process is still alive
                    global training_process, training_pid
                    is_process_alive = False
                    if training_process and training_process.poll() is None:
                        is_process_alive = True
                    elif training_pid:
                        try:
                            os.kill(training_pid, 0)
                            is_process_alive = True
                        except (OSError, ProcessLookupError):
                            is_process_alive = False

                    # Read status file
                    if status_file.exists():
                        try:
                            with open(status_file, 'r') as f:
                                status = json.load(f)
                                status['process_alive'] = is_process_alive
                                status['pid'] = training_pid

                                # Safety: ensure best_reward is JSON-safe (convert None to 0.0 for display)
                                if status.get('best_reward') is None:
                                    status['best_reward'] = 0.0

                                # Only send if status changed
                                if status != last_status:
                                    yield f"data: {json.dumps(status)}\n\n"
                                    last_status = status
                                    no_file_count = 0

                        except Exception as e:
                            yield f"data: {json.dumps({'error': f'Failed to read status: {str(e)}', 'process_alive': is_process_alive})}\n\n"
                    else:
                        no_file_count += 1
                        # Send a waiting message every 5 seconds if no status file
                        if no_file_count % 5 == 0:
                            yield f"data: {json.dumps({'is_training': is_process_alive, 'message': 'Waiting for training to start...', 'process_alive': is_process_alive})}\n\n"

                    # If process died and no more updates, break
                    if not is_process_alive and last_status and last_status.get('is_training') == False:
                        yield f"data: {json.dumps({'is_training': False, 'message': 'Training completed', 'process_alive': False})}\n\n"
                        break

                    await asyncio.sleep(1)  # Update every second

                except (asyncio.CancelledError, GeneratorExit):
                    # Client disconnected - exit gracefully
                    break
                except Exception as e:
                    # Log other errors but continue
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    await asyncio.sleep(1)

        except (asyncio.CancelledError, GeneratorExit, ConnectionResetError):
            # Client disconnected - exit gracefully without error
            pass
        except Exception as e:
            # Log unexpected errors
            try:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            except:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@router.get("/training/frames", summary="Get Training Frames")
def get_training_frames(limit: int = 10):
    """Get list of captured training frames"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    frames_dir = RL_BACKEND_PATH / "training" / "frames"

    if not frames_dir.exists():
        return {"frames": [], "message": "No frames directory found"}

    try:
        # Get all frame files
        frame_files = sorted(frames_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)

        # Limit number of frames
        frame_files = frame_files[:limit]

        frames = []
        for frame_file in frame_files:
            frames.append({
                "filename": frame_file.name,
                "url": f"/rl/training/frame/{frame_file.name}",
                "timestamp": frame_file.stat().st_mtime
            })

        return {
            "frames": frames,
            "total_count": len(list(frames_dir.glob("*.png"))),
            "returned_count": len(frames)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list frames: {str(e)}")


@router.get("/training/frame/{filename}", summary="Get Specific Frame")
def get_training_frame(filename: str):
    """Get a specific training frame image"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Security: only allow alphanumeric and underscore/dash/dot in filename
    if not filename.replace("_", "").replace("-", "").replace(".", "").replace("step", "").replace("episode", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid filename")

    frame_path = RL_BACKEND_PATH / "training" / "frames" / filename

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(frame_path, media_type="image/png")


@router.get("/training/logs", summary="Get Training Logs")
def get_training_logs(lines: int = 100):
    """Get recent training logs"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Look for log files in logs directory
    logs_dir = RL_BACKEND_PATH / "logs"

    if not logs_dir.exists():
        return {"logs": "", "message": "No logs directory found"}

    try:
        # Find most recent log file
        log_files = sorted(logs_dir.rglob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)

        if not log_files:
            # Try tensorboard event files
            event_files = sorted(logs_dir.rglob("events.out.tfevents.*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if event_files:
                return {
                    "logs": "",
                    "message": "Training logs are in TensorBoard format. Use TensorBoard to view.",
                    "tensorboard_command": f"tensorboard --logdir {logs_dir}",
                    "event_files": [str(f.name) for f in event_files[:5]]
                }
            return {"logs": "", "message": "No log files found"}

        # Read the most recent log file
        log_file = log_files[0]
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "logs": "".join(recent_lines),
            "log_file": str(log_file.name),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@router.get("/model/info", summary="Get Model Info")
def get_model_info():
    """Get information about trained model"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip"
    tfjs_path = RL_BACKEND_PATH / "models" / "tfjs_model" / "model.json"

    return {
        "pytorch_model": {
            "exists": model_path.exists(),
            "path": str(model_path) if model_path.exists() else None
        },
        "tfjs_model": {
            "exists": tfjs_path.exists(),
            "path": str(tfjs_path) if tfjs_path.exists() else None
        },
        "input_shape": [84, 84, 3],  # Observation shape
        "num_actions": 9,  # 0=idle, 1=left, 2=right, 3=jump, 4=sprint+right, 5=duck, 6=jump+left, 7=jump+right, 8=sprint+jump+right
        "algorithm": "PPO (Proximal Policy Optimization)"
    }


@router.get("/models/available", summary="List Available Models")
def get_available_models():
    """Get list of all available models for gameplay (both TensorFlow.js and ONNX)"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    models_dir = RL_BACKEND_PATH / "models"
    available_models = []

    # Check for main trained model (TensorFlow.js)
    main_tfjs = models_dir / "tfjs_model" / "model.json"
    if main_tfjs.exists():
        available_models.append({
            "id": "main_tfjs",
            "name": "Main Model (TensorFlow.js)",
            "path": "/models/rl/tfjs_model/model.json",
            "description": "Primary trained model",
            "type": "main",
            "format": "tfjs"
        })

    # Check for main trained model (ONNX)
    main_onnx = models_dir / "exported_onnx" / "model.onnx"
    if main_onnx.exists():
        available_models.append({
            "id": "main_onnx",
            "name": "Main Model (ONNX)",
            "path": "/api/rl/models/onnx/main/model.onnx",
            "description": "Primary trained model (ONNX format)",
            "type": "main",
            "format": "onnx"
        })

    # Check for Behavioral Cloning model (ONNX)
    bc_onnx = LAUNCHER_BACKEND_PATH / "models" / "rl" / "bc_onnx" / "model.onnx"
    if bc_onnx.exists():
        available_models.append({
            "id": "bc_onnx",
            "name": "Behavioral Cloning (ONNX)",
            "path": "/models/rl/bc_onnx/model.onnx",
            "description": "Trained from human gameplay (97.1% accuracy)",
            "type": "main",
            "format": "onnx"
        })

    # Check for episode-specific models
    if models_dir.exists():
        # TensorFlow.js episode models
        for episode_dir in models_dir.glob("episode_*_tfjs"):
            tfjs_model = episode_dir / "tfjs_model" / "model.json"
            if tfjs_model.exists():
                try:
                    episode_num = int(episode_dir.name.split("_")[1])
                    available_models.append({
                        "id": f"episode_{episode_num}_tfjs",
                        "name": f"Episode {episode_num} (TensorFlow.js)",
                        "path": f"/models/{episode_dir.name}/tfjs_model/model.json",
                        "description": f"Episode {episode_num} checkpoint",
                        "type": "episode",
                        "episode": episode_num,
                        "format": "tfjs"
                    })
                except (IndexError, ValueError):
                    pass

        # ONNX episode models
        for episode_dir in models_dir.glob("episode_*_onnx"):
            onnx_model = episode_dir / "model.onnx"
            if onnx_model.exists():
                try:
                    episode_num = int(episode_dir.name.split("_")[1])

                    # Try to read reward from checkpoint manifest
                    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
                    reward_info = ""
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r') as f:
                                checkpoints = json.load(f)
                                checkpoint = next((c for c in checkpoints if c['episode'] == episode_num), None)
                                if checkpoint:
                                    reward_info = f" (Reward: {checkpoint['reward']:.1f})"
                        except:
                            pass

                    available_models.append({
                        "id": f"episode_{episode_num}_onnx",
                        "name": f"Episode {episode_num} (ONNX){reward_info}",
                        "path": f"/api/rl/models/onnx/{episode_num}/model.onnx",
                        "description": f"Episode {episode_num} checkpoint",
                        "type": "episode",
                        "episode": episode_num,
                        "format": "onnx"
                    })
                except (IndexError, ValueError):
                    pass

    # Sort by episode number (newest first), with main models at top
    available_models.sort(key=lambda m: (
        0 if m['type'] == 'main' else 1,  # Main models first
        -m.get('episode', 0)  # Then by episode (descending)
    ))

    return {
        "models": available_models,
        "count": len(available_models)
    }


@router.post("/model/export", summary="Export Model")
def export_model():
    """Export PyTorch model to TensorFlow.js format"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    export_script = RL_BACKEND_PATH / "training" / "export_model.py"
    if not export_script.exists():
        raise HTTPException(status_code=500, detail="Export script not found")

    model_path = RL_BACKEND_PATH / "models" / "platformer_agent.zip"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="No trained model found. Train a model first.")

    try:
        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found")
            print(f"[EXPORT] Run setup_export_env.bat in RL/backend to create it")

        # Run export script
        result = subprocess.run(
            [python_cmd, str(export_script)],
            cwd=str(RL_BACKEND_PATH),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Model exported to TensorFlow.js format",
                "output_path": str(RL_BACKEND_PATH / "models" / "tfjs_model"),
                "details": result.stdout
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Export failed: {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/checkpoints/episodes", summary="Get Episode Checkpoints")
def get_episode_checkpoints():
    """Get list of all episode checkpoints from current training session"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"

    if not manifest_path.exists():
        return {
            "checkpoints": [],
            "message": "No episode checkpoints found. Start training to create checkpoints."
        }

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the best model (highest reward) from entire training session
        best_model = max(checkpoints, key=lambda x: x['reward']) if checkpoints else None

        # Get ALL episodes sorted by episode number (newest first)
        # This allows users to see and play against the best model from the entire session
        all_checkpoints = sorted(checkpoints, key=lambda x: x['episode'], reverse=True)

        return {
            "best_model": best_model,
            "recent_checkpoints": all_checkpoints,  # Now includes all episodes, not just last 10
            "total_count": len(checkpoints)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read checkpoints: {str(e)}")


@router.post("/checkpoints/export-onnx/{episode}", summary="Export Episode Checkpoint to ONNX")
def export_episode_checkpoint_onnx(episode: int):
    """
    Export a specific episode checkpoint to ONNX format for ONNX Runtime Web
    This is the recommended export method - avoids TensorFlow conversion issues
    """
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Read manifest to find checkpoint
    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No episode checkpoints found")

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the checkpoint
        checkpoint = next((c for c in checkpoints if c['episode'] == episode), None)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Checkpoint for episode {episode} not found")

        model_path = RL_BACKEND_PATH / "training" / (checkpoint['path'] + '.zip')
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {model_path}")

        # Export to ONNX using new export script
        export_script = RL_BACKEND_PATH / "training" / "export_model_onnx.py"
        if not export_script.exists():
            raise HTTPException(status_code=500, detail="ONNX export script not found")

        # Output directory for this episode
        output_dir = RL_BACKEND_PATH / "models" / f"episode_{episode}_onnx"

        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found")
            print(f"[EXPORT] Using system Python (may have dependency issues)")

        cmd = [
            python_cmd,
            str(export_script),
            "--model-path", str(model_path),
            "--output-dir", str(output_dir)
        ]

        print(f"[EXPORT] Running ONNX export: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        print(f"[EXPORT] Return code: {result.returncode}")
        if result.stdout:
            print(f"[EXPORT] stdout: {result.stdout}")
        if result.stderr:
            print(f"[EXPORT] stderr: {result.stderr}")

        if result.returncode == 0:
            return {
                "status": "success",
                "episode": episode,
                "reward": checkpoint['reward'],
                "model_path": str(output_dir / "model.onnx"),
                "format": "ONNX",
                "runtime": "ONNX Runtime Web",
                "message": f"Episode {episode} exported to ONNX successfully",
                "usage_guide": str(output_dir / "USAGE.md")
            }
        else:
            error_msg = f"ONNX export failed: {result.stderr}"
            print(f"[EXPORT ERROR] {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export to ONNX: {str(e)}")


@router.post("/checkpoints/export/{episode}", summary="Export Episode Checkpoint")
def export_episode_checkpoint(episode: int):
    """
    Export a specific episode checkpoint to TensorFlow.js format
    This allows playing against that specific episode's AI
    NOTE: This endpoint has known issues with Conv layer conversion.
    Use /checkpoints/export-onnx/{episode} instead for ONNX Runtime Web (recommended)
    """
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    # Read manifest to find checkpoint
    manifest_path = RL_BACKEND_PATH / "training" / "episode_checkpoints" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="No episode checkpoints found")

    try:
        with open(manifest_path, 'r') as f:
            checkpoints = json.load(f)

        # Find the checkpoint
        checkpoint = next((c for c in checkpoints if c['episode'] == episode), None)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Checkpoint for episode {episode} not found")

        model_path = RL_BACKEND_PATH / "training" / (checkpoint['path'] + '.zip')
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {model_path}")

        # Export to TensorFlow.js
        export_script = RL_BACKEND_PATH / "training" / "export_model.py"
        if not export_script.exists():
            raise HTTPException(status_code=500, detail="Export script not found")

        # Run export with specific output directory
        output_dir = RL_BACKEND_PATH / "models" / f"episode_{episode}_tfjs"

        # Check if export virtual environment exists
        export_env_python = RL_BACKEND_PATH / "export_env" / "Scripts" / "python.exe"
        if export_env_python.exists():
            python_cmd = str(export_env_python)
            print(f"[EXPORT] Using export environment: {python_cmd}")
        else:
            python_cmd = sys.executable
            print(f"[EXPORT] WARNING: Export environment not found at {export_env_python}")
            print(f"[EXPORT] Using system Python: {python_cmd}")
            print(f"[EXPORT] Run setup_export_env.bat in RL/backend to create export environment")

        cmd = [
            python_cmd,
            str(export_script),
            "--model-path", str(model_path),
            "--output-dir", str(output_dir)
        ]

        print(f"[EXPORT] Running command: {' '.join(cmd)}")
        print(f"[EXPORT] Model path: {model_path}")
        print(f"[EXPORT] Output dir: {output_dir}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        print(f"[EXPORT] Return code: {result.returncode}")
        if result.stdout:
            print(f"[EXPORT] stdout: {result.stdout}")
        if result.stderr:
            print(f"[EXPORT] stderr: {result.stderr}")

        if result.returncode == 0:
            return {
                "status": "success",
                "episode": episode,
                "reward": checkpoint['reward'],
                "model_path": f"/models/episode_{episode}_tfjs/tfjs_model/model.json",
                "message": f"Episode {episode} exported successfully"
            }
        else:
            error_msg = f"Export failed with code {result.returncode}. stderr: {result.stderr}"
            print(f"[EXPORT ERROR] {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Export timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export checkpoint: {str(e)}")


def check_gpu_available() -> bool:
    """Check if CUDA GPU is available for training"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@router.get("/gpu/info", summary="Get GPU Information")
def get_gpu_info():
    """Get GPU information for training"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": torch.cuda.get_device_capability(0),
                "memory_allocated": torch.cuda.memory_allocated(0),
                "memory_reserved": torch.cuda.memory_reserved(0)
            }
        else:
            return {
                "cuda_available": False,
                "message": "No CUDA GPU detected. Training will use CPU (slower)."
            }
    except ImportError:
        return {
            "cuda_available": False,
            "message": "PyTorch not installed"
        }


@router.get("/models/onnx/main/model.onnx", summary="Get Main ONNX Model File")
def get_main_onnx_model():
    """Serve main ONNX model file for browser loading"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    model_path = RL_BACKEND_PATH / "models" / "exported_onnx" / "model.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Main ONNX model not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename="main_model.onnx"
    )


@router.get("/models/onnx/{episode}/model.onnx", summary="Get ONNX Model File")
def get_onnx_model(episode: int):
    """Serve ONNX model file for browser loading"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    model_path = RL_BACKEND_PATH / "models" / f"episode_{episode}_onnx" / "model.onnx"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model for episode {episode} not found")

    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=f"episode_{episode}_model.onnx"
    )


@router.get("/models/onnx/main/model.onnx.data", summary="Get Main ONNX Model Data File")
def get_main_onnx_model_data():
    """Serve main ONNX model data file (external weights)"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    data_path = RL_BACKEND_PATH / "models" / "exported_onnx" / "model.onnx.data"

    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Main ONNX model data file not found")

    return FileResponse(
        data_path,
        media_type="application/octet-stream",
        filename="main_model.onnx.data"
    )


@router.get("/models/onnx/{episode}/model.onnx.data", summary="Get ONNX Model Data File")
def get_onnx_model_data(episode: int):
    """Serve ONNX model data file (external weights)"""
    if not RL_BACKEND_PATH:
        raise HTTPException(status_code=500, detail="RL backend path not found")

    data_path = RL_BACKEND_PATH / "models" / f"episode_{episode}_onnx" / "model.onnx.data"

    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model data file for episode {episode} not found")

    return FileResponse(
        data_path,
        media_type="application/octet-stream",
        filename=f"episode_{episode}_model.onnx.data"
    )


# Database Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SCORES_DB_PATH = DATA_DIR / "rl_scores.json"
USERS_DB_PATH = DATA_DIR / "rl_users.json"
METRICS_DB_PATH = DATA_DIR / "rl_metrics.json"
TOKENS_DB_PATH = DATA_DIR / "rl_completion_tokens.json"  # Track used completion tokens
TRAINING_DATA_DIR = DATA_DIR / "training_data"

# JWT Configuration
# CRITICAL: Set JWT_SECRET in .env file for production!
# Generate a secure secret: python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production-INSECURE")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 30


# ============================================================================
# USER AUTHENTICATION FUNCTIONS
# ============================================================================

def load_users() -> List[Dict[str, Any]]:
    """Load users from JSON file"""
    if not USERS_DB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    try:
        with open(USERS_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading users: {e}")
        return []


def save_users(users: List[Dict[str, Any]]):
    """Save users to JSON file"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(USERS_DB_PATH, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        print(f"Error saving users: {e}")
        raise


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def create_jwt_token(user_id: str, username: str) -> str:
    """Create a JWT token for authentication"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current user from Authorization header"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")
    payload = verify_jwt_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    users = load_users()
    user = next((u for u in users if u['user_id'] == payload['user_id']), None)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ============================================================================
# USER METRICS FUNCTIONS
# ============================================================================

def update_user_metrics(user_id: str, metrics: GameMetrics):
    """Update user's cumulative metrics"""
    users = load_users()

    user = next((u for u in users if u['user_id'] == user_id), None)
    if not user:
        return False

    # Update cumulative stats
    user['total_games'] = user.get('total_games', 0) + 1
    user['total_jumps'] = user.get('total_jumps', 0) + metrics.jumps
    user['total_points'] = user.get('total_points', 0) + metrics.points
    user['total_distance'] = user.get('total_distance', 0) + metrics.distance
    user['total_playtime'] = user.get('total_playtime', 0.0) + metrics.time_played

    save_users(users)
    return True


# ============================================================================
# SCOREBOARD FUNCTIONS
# ============================================================================

def load_scores() -> List[Dict[str, Any]]:
    """Load scores from JSON file"""
    if not SCORES_DB_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []

    try:
        with open(SCORES_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading scores: {e}")
        return []


def save_scores(scores: List[Dict[str, Any]]):
    """Save scores to JSON file"""
    try:
        SCORES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORES_DB_PATH, 'w') as f:
            json.dump(scores, f, indent=2)
    except Exception as e:
        print(f"Error saving scores: {e}")
        raise


def load_used_tokens() -> List[str]:
    """Load list of used completion tokens"""
    if not TOKENS_DB_PATH.exists() or TOKENS_DB_PATH.stat().st_size == 0:
        return []

    try:
        with open(TOKENS_DB_PATH, 'r') as f:
            data = json.load(f)
            # Clean up old tokens (older than 24 hours)
            current_time = time.time()
            data['tokens'] = [t for t in data.get('tokens', [])
                            if current_time - t.get('used_at', 0) < 86400]
            return [t['token'] for t in data['tokens']]
    except (json.JSONDecodeError, Exception):
        return []


def mark_token_used(token: str):
    """Mark a completion token as used (prevents reuse)"""
    try:
        TOKENS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Load existing tokens
        data = {'tokens': []}
        if TOKENS_DB_PATH.exists() and TOKENS_DB_PATH.stat().st_size > 0:
            try:
                with open(TOKENS_DB_PATH, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # File is empty or corrupted, start fresh
                data = {'tokens': []}

        # Add new token with timestamp
        data['tokens'].append({
            'token': token,
            'used_at': time.time()
        })

        # Save back
        with open(TOKENS_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error marking token as used: {e}")
        raise


def count_recent_completion_tokens(user_id: str, minutes: int = 60) -> int:
    """
    Count how many completion tokens a user has requested in the last N minutes.
    Used for rate limiting.

    Args:
        user_id: The user's unique ID
        minutes: Time window to check (default 60 minutes)

    Returns:
        Number of completion tokens requested in the time window
    """
    try:
        log_dir = DATA_DIR / "audit_logs"
        if not log_dir.exists():
            return 0

        # Check current month's log file
        current_month = datetime.utcnow().strftime('%Y-%m')
        log_file = log_dir / f"activity_{current_month}.json"

        if not log_file.exists():
            return 0

        # Calculate cutoff time
        cutoff_time = time.time() - (minutes * 60)

        # Read and parse log file
        count = 0
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    # Check if this is a game_completed event for this user
                    if (entry.get('type') == 'game_completed' and
                        entry.get('user_id') == user_id):
                        # Parse timestamp
                        timestamp_str = entry.get('timestamp', '')
                        entry_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()

                        # Count if within time window
                        if entry_time >= cutoff_time:
                            count += 1
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        return count
    except Exception as e:
        print(f"Error counting recent completion tokens: {e}")
        # Fail open - don't block users if we can't read logs
        return 0


def validate_score_physics(score_entry: ScoreEntry) -> tuple[bool, str]:
    """
    Validate that score is physically possible based on game mechanics.

    Returns:
        (is_valid, reason_if_invalid)
    """
    # Distance bounds (game has finite level length)
    MAX_DISTANCE = 50000  # Maximum possible distance in game
    MIN_DISTANCE = 100    # Must move at least 100 pixels

    if score_entry.distance < MIN_DISTANCE:
        return False, "distance_too_low"
    if score_entry.distance > MAX_DISTANCE:
        return False, "distance_too_high"

    # Time validation - minimum time based on distance
    # Max player speed ~400 pixels/sec with sprint
    # But realistically need time for jumps, platforms, etc.
    min_possible_time = score_entry.distance / 600  # Very generous max speed
    if score_entry.time < min_possible_time:
        return False, "time_too_fast_for_distance"

    # Time validation - maximum reasonable time (prevents absurdly long games)
    max_reasonable_time = 600  # 10 minutes is plenty for any legit run
    if score_entry.time > max_reasonable_time:
        return False, "time_too_long"

    # Score validation - must be reasonable for distance
    # Score formula: distance/100 + collectibles*10
    # Max collectibles per 100 pixels: ~0.5, so max score ~= distance/100 + distance/200
    max_possible_score = (score_entry.distance / 20) * 1.5  # 50% margin
    if score_entry.score > max_possible_score:
        return False, "score_too_high_for_distance"

    # Score should not be negative
    if score_entry.score < 0:
        return False, "negative_score"

    # Difficulty validation
    valid_difficulties = {'easy', 'medium', 'hard'}
    if score_entry.difficulty not in valid_difficulties:
        return False, "invalid_difficulty"

    return True, ""


def validate_gameplay_recording(score_entry: ScoreEntry) -> tuple[bool, str]:
    """
    Validate score against submitted gameplay recording (training data).
    This is the strongest anti-cheat measure - verifying the actual gameplay.

    Returns:
        (is_valid, reason_if_invalid)
    """
    if not score_entry.training_data or len(score_entry.training_data) == 0:
        # No training data provided - REJECT the score
        # Training data is REQUIRED to prove the game was actually played
        return False, "no_gameplay_data_provided"

    try:
        data_points = score_entry.training_data

        # Validation 1: Minimum number of frames
        # A valid game should have at least 30 frames (0.5 seconds at 60fps)
        if len(data_points) < 30:
            return False, "insufficient_gameplay_frames"

        # Validation 2: Sequential frames
        # Frames should be sequential without huge gaps
        for i in range(1, len(data_points)):
            prev_frame = data_points[i-1].get('frame_number', 0)
            curr_frame = data_points[i].get('frame_number', 0)
            frame_gap = curr_frame - prev_frame

            # Allow small gaps (1-5 frames) for dropped frames, but not huge jumps
            if frame_gap > 10:
                return False, "non_sequential_frames"

        # Validation 3: Distance matches final player position
        # The claimed distance should approximately match the final player_x position
        final_data = data_points[-1]
        final_player_x = final_data.get('player_x', 0)

        # Allow 10% margin of error for rounding/camera offset
        distance_lower = score_entry.distance * 0.9
        distance_upper = score_entry.distance * 1.1

        if not (distance_lower <= final_player_x <= distance_upper):
            return False, "distance_mismatch_with_gameplay"

        # Validation 4: Time matches frame count
        # Assuming 60 FPS, time should be approximately frames/60
        first_frame = data_points[0].get('frame_number', 0)
        last_frame = data_points[-1].get('frame_number', 0)
        frame_duration = last_frame - first_frame
        expected_time = frame_duration / 60.0  # 60 FPS

        # Allow 20% margin for variable frame rate
        time_lower = expected_time * 0.8
        time_upper = expected_time * 1.2

        if not (time_lower <= score_entry.time <= time_upper):
            return False, "time_mismatch_with_gameplay"

        # Validation 5: Difficulty matches
        # All data points should have the same difficulty as the score
        for point in data_points:
            if point.get('difficulty') != score_entry.difficulty:
                return False, "difficulty_mismatch_in_gameplay"

        # Validation 6: Player progresses forward
        # Player X position should generally increase over time (moving right)
        # Check first 10% and last 10% of data
        early_avg_x = sum(p.get('player_x', 0) for p in data_points[:max(1, len(data_points)//10)]) / max(1, len(data_points)//10)
        late_avg_x = sum(p.get('player_x', 0) for p in data_points[-max(1, len(data_points)//10):]) / max(1, len(data_points)//10)

        # Player should have moved significantly forward
        if late_avg_x <= early_avg_x + 100:  # At least 100 pixels forward
            return False, "insufficient_forward_progress"

        # All validations passed!
        return True, ""

    except Exception as e:
        print(f"Error validating gameplay recording: {e}")
        # If validation fails due to error, fail safe - reject the score
        return False, "gameplay_validation_error"


def detect_suspicious_patterns(user_id: str, score_entry: ScoreEntry, all_scores: List[Dict]) -> tuple[bool, str]:
    """
    Detect suspicious patterns in score submissions.

    Returns:
        (is_suspicious, reason)
    """
    # Get all scores for this user
    user_scores = [s for s in all_scores if s.get('user_id') == user_id]

    if len(user_scores) == 0:
        return False, ""  # First score, can't detect patterns

    # Pattern 1: Identical scores across ALL difficulties
    # Real players vary, cheaters often submit same score everywhere
    if len(user_scores) >= 2:
        identical_count = 0
        for existing in user_scores:
            if (existing['score'] == score_entry.score and
                existing['distance'] == score_entry.distance and
                existing['difficulty'] != score_entry.difficulty):  # Different difficulty, same stats
                identical_count += 1

        # If user has 2+ identical scores on different difficulties, suspicious
        if identical_count >= 2:
            return True, "identical_scores_across_difficulties"

    # Pattern 2: Impossibly perfect consistency
    # Real players have variance, bots/cheaters often submit perfect rounds
    if len(user_scores) >= 3:
        # Check if all scores are exactly the same
        all_same = all(
            s['score'] == user_scores[0]['score'] and
            s['distance'] == user_scores[0]['distance']
            for s in user_scores
        )
        if all_same:
            return True, "perfect_consistency_all_games"

    # Pattern 3: Impossibly fast improvement
    # Real players improve gradually, cheaters jump from bad to perfect
    user_scores_same_difficulty = [
        s for s in user_scores
        if s.get('difficulty') == score_entry.difficulty
    ]

    if len(user_scores_same_difficulty) >= 1:
        best_existing_time = min(s['time'] for s in user_scores_same_difficulty)
        # If new time is less than 50% of previous best, very suspicious
        # (unless both are very small times < 5 seconds, which might be valid)
        if score_entry.time < best_existing_time * 0.5 and best_existing_time > 5:
            return True, "impossible_improvement_rate"

    # Pattern 4: Submission timing - too fast after token request
    # This is already checked in the endpoint, but we can add more logic here if needed

    return False, ""


@router.post("/game-complete", summary="Generate Completion Token")
def game_complete(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Generate a completion token when player reaches the goal.
    This token must be submitted with the score to prove the game was actually completed.

    SECURITY:
    - Requires JWT authentication
    - Rate limited to 10 requests per hour per user
    - Tokens are single-use and expire in 5 minutes

    Returns:
        completion_token: JWT signed token proving goal was reached
    """
    # Get client info (IP + User Agent)
    client_info = get_client_info(request)

    # Require authentication
    if not authorization:
        log_activity("game_completion_denied", {
            **client_info,
            "reason": "no_authentication"
        })
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Must be logged in to complete game"
        )

    try:
        user = get_current_user(authorization)
        user_id = user['user_id']
    except Exception as e:
        log_activity("game_completion_denied", {
            **client_info,
            "reason": "invalid_token"
        })
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or expired token"
        )

    # Rate limiting: Check recent completion tokens for this user
    recent_count = count_recent_completion_tokens(user_id, minutes=60)
    if recent_count >= 10:  # Max 10 completion tokens per hour
        log_activity("game_completion_denied", {
            **client_info,
            "user_id": user_id,
            "reason": "rate_limit_exceeded",
            "recent_count": recent_count
        })
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Maximum 10 game completions per hour. Please wait before trying again."
        )

    # Generate unique completion token
    token_id = str(uuid.uuid4())
    payload = {
        'type': 'game_completion',
        'user_id': user_id,
        'token_id': token_id,
        'issued_at': time.time(),
        'exp': datetime.utcnow() + timedelta(minutes=5)  # Token expires in 5 minutes
    }

    completion_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    # Log completion
    log_activity("game_completed", {
        **client_info,
        "user_id": user_id,
        "token_id": token_id
    })

    return {
        "completion_token": completion_token,
        "message": "Game completed! You can now submit your score."
    }


@router.get("/scores", summary="Get Leaderboard Scores")
def get_scores(limit: int = 10, difficulty: str = 'easy', include_flagged: bool = False):
    """
    Get top scores from the global leaderboard
    Returns scores sorted by completion time (fastest first)
    Filters by difficulty level (easy, medium, hard)

    By default, excludes flagged/suspicious scores from the leaderboard.
    Set include_flagged=true to see all scores (for admin review).
    """
    scores = load_scores()

    # Filter by difficulty
    filtered_scores = [s for s in scores if s.get('difficulty', 'easy') == difficulty]

    # Exclude flagged scores unless explicitly requested
    if not include_flagged:
        filtered_scores = [s for s in filtered_scores if not s.get('flagged', False)]

    # Sort by time (fastest first) and limit results
    sorted_scores = sorted(filtered_scores, key=lambda x: x['time'])[:limit]

    return {
        "scores": sorted_scores,
        "total_count": len(filtered_scores),
        "returned_count": len(sorted_scores),
        "difficulty": difficulty,
        "flagged_excluded": not include_flagged
    }


@router.post("/scores", summary="Submit Score to Leaderboard")
def submit_score(score_entry: ScoreEntry, request: Request, authorization: Optional[str] = Header(None)):
    """
    Submit a new score to the global leaderboard
    If logged in, score is linked to user_id
    If player already has a score for this difficulty, only updates if new time is better (faster)
    Logs IP addresses and User Agent for security and abuse detection
    """
    from datetime import datetime

    # Get client info (IP + User Agent) for logging
    client_info = get_client_info(request)

    # Require authentication - only registered users can submit scores
    if not authorization:
        log_activity("score_submission_denied", {
            **client_info,
            "reason": "not_authenticated",
            "attempted_name": score_entry.name
        })
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: You must be logged in to submit scores"
        )

    try:
        user = get_current_user(authorization)
        user_id = user['user_id']
        player_name = user['username']  # Always use registered username
    except Exception as e:
        log_activity("score_submission_denied", {
            **client_info,
            "reason": "invalid_token",
            "attempted_name": score_entry.name
        })
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or expired token"
        )

    # Validate completion token (CRITICAL: Prevents fake scores without finishing game)
    try:
        # Decode completion token
        completion_payload = jwt.decode(
            score_entry.completion_token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )

        # Validate token type and user
        if completion_payload.get('type') != 'game_completion':
            raise ValueError("Invalid token type")

        if completion_payload.get('user_id') != user_id:
            raise ValueError("Token user mismatch")

        # Check if token already used (prevent reuse)
        used_tokens = load_used_tokens()
        if score_entry.completion_token in used_tokens:
            log_activity("score_submission_denied", {
                **client_info,
                "username": player_name,
                "user_id": user_id,
                "reason": "token_already_used",
                "difficulty": score_entry.difficulty
            })
            raise HTTPException(
                status_code=400,
                detail="This completion token has already been used. You must finish a new game."
            )

        # Mark token as used
        mark_token_used(score_entry.completion_token)

    except jwt.ExpiredSignatureError:
        log_activity("score_submission_denied", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "reason": "completion_token_expired",
            "difficulty": score_entry.difficulty
        })
        raise HTTPException(
            status_code=400,
            detail="Completion token expired. Please finish the game again to submit score."
        )
    except (jwt.InvalidTokenError, ValueError) as e:
        log_activity("score_submission_denied", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "reason": "invalid_completion_token",
            "difficulty": score_entry.difficulty
        })
        raise HTTPException(
            status_code=400,
            detail="Invalid completion token. You must finish the game to submit a score."
        )

    # Load existing scores
    scores = load_scores()

    # ENHANCED VALIDATION: Check if score is physically possible
    is_valid, validation_reason = validate_score_physics(score_entry)
    if not is_valid:
        log_activity("invalid_score_rejected", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "time": score_entry.time,
            "score": score_entry.score,
            "distance": score_entry.distance,
            "difficulty": score_entry.difficulty,
            "reason": validation_reason
        })
        raise HTTPException(
            status_code=400,
            detail=f"Invalid score: {validation_reason.replace('_', ' ')}"
        )

    # GAMEPLAY RECORDING VALIDATION: Verify score matches submitted gameplay data
    # This is the strongest anti-cheat - validates actual gameplay recording
    is_gameplay_valid, gameplay_reason = validate_gameplay_recording(score_entry)
    if not is_gameplay_valid:
        log_activity("invalid_score_rejected", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "time": score_entry.time,
            "score": score_entry.score,
            "distance": score_entry.distance,
            "difficulty": score_entry.difficulty,
            "reason": gameplay_reason,
            "validation_type": "gameplay_recording"
        })
        raise HTTPException(
            status_code=400,
            detail=f"Score does not match gameplay recording: {gameplay_reason.replace('_', ' ')}"
        )

    # ANOMALY DETECTION: Check for suspicious patterns
    is_suspicious, suspicious_reason = detect_suspicious_patterns(user_id, score_entry, scores)
    flagged = False
    flag_reason = None

    if is_suspicious:
        flagged = True
        flag_reason = suspicious_reason
        log_activity("suspicious_score_flagged", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "time": score_entry.time,
            "score": score_entry.score,
            "distance": score_entry.distance,
            "difficulty": score_entry.difficulty,
            "reason": suspicious_reason
        })

    # Create new score entry
    new_score = {
        "name": player_name,
        "time": round(score_entry.time, 1),
        "score": score_entry.score,
        "distance": score_entry.distance,
        "difficulty": score_entry.difficulty,
        "timestamp": datetime.utcnow().isoformat(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "user_id": user_id,  # Link to user if authenticated
        "ip_address": client_info['ip'],  # Store IP for abuse tracking
        "user_agent": client_info['user_agent'],  # Store User Agent for device identification
        "flagged": flagged,  # Mark if anomaly detected
        "flag_reason": flag_reason if flagged else None  # Reason for flagging
    }

    # Check if player already exists for this difficulty
    existing_index = None
    for i, s in enumerate(scores):
        # Match by user_id if authenticated, otherwise by name
        if user_id:
            if s.get('user_id') == user_id and s.get('difficulty', 'easy') == new_score['difficulty']:
                existing_index = i
                break
        else:
            if (s['name'].lower() == new_score['name'].lower() and
                s.get('difficulty', 'easy') == new_score['difficulty'] and
                not s.get('user_id')):  # Only match unlinked scores
                existing_index = i
                break

    if existing_index is not None:
        # Only update if new time is better (faster)
        if new_score['time'] < scores[existing_index]['time']:
            previous_time = scores[existing_index]['time']
            scores[existing_index] = new_score
            save_scores(scores)

            # Log score update
            log_activity("score_updated", {
                **client_info,
                "username": player_name,
                "user_id": user_id,
                "difficulty": new_score['difficulty'],
                "new_time": new_score['time'],
                "previous_time": previous_time,
                "score": new_score['score'],
                "distance": new_score['distance']
            })

            response = {
                "status": "updated",
                "message": f"New best time for {new_score['name']} on {new_score['difficulty']}!",
                "score": new_score,
                "previous_time": previous_time
            }
            if flagged:
                response["warning"] = f"Score flagged for review: {flag_reason.replace('_', ' ')}"
                response["under_review"] = True
            return response
        else:
            return {
                "status": "not_updated",
                "message": f"Previous time was better",
                "current_best": scores[existing_index]['time'],
                "submitted_time": new_score['time']
            }
    else:
        # Add new player for this difficulty
        scores.append(new_score)
        save_scores(scores)

        # Log new score submission
        log_activity("score_created", {
            **client_info,
            "username": player_name,
            "user_id": user_id,
            "difficulty": new_score['difficulty'],
            "time": new_score['time'],
            "score": new_score['score'],
            "distance": new_score['distance']
        })

        response = {
            "status": "created",
            "message": f"Score added for {new_score['name']} on {new_score['difficulty']}",
            "score": new_score
        }
        if flagged:
            response["warning"] = f"Score flagged for review: {flag_reason.replace('_', ' ')}"
            response["under_review"] = True
        return response


@router.delete("/scores", summary="Clear All Scores (Admin Only)")
def clear_scores(request: Request, auth: AdminAuth = None):
    """
    Clear all scores from the leaderboard
    WARNING: This is irreversible!
    Requires admin password for authorization (sent in request body)
    """
    client_info = get_client_info(request)

    # Admin password check - MUST be set in environment variables
    ADMIN_PASSWORD = os.getenv("LEADERBOARD_ADMIN_PASSWORD")
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: LEADERBOARD_ADMIN_PASSWORD not set"
        )

    if not auth or not auth.admin_password or auth.admin_password != ADMIN_PASSWORD:
        # Log unauthorized attempt
        log_activity("leaderboard_clear_denied", {
            **client_info,
            "reason": "invalid_admin_password"
        })
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Admin password required to clear leaderboard"
        )

    try:
        # Count scores before clearing
        scores_count = len(load_scores()) if SCORES_DB_PATH.exists() else 0

        if SCORES_DB_PATH.exists():
            SCORES_DB_PATH.unlink()

        # Log successful clear
        log_activity("leaderboard_cleared", {
            **client_info,
            "scores_deleted": scores_count
        })

        return {
            "status": "success",
            "message": f"All {scores_count} scores have been cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear scores: {str(e)}")


@router.delete("/scores/{player_name}/{difficulty}", summary="Delete Individual Score")
def delete_score(
    player_name: str,
    difficulty: str,
    request: Request,
    authorization: Optional[str] = Header(None),
    auth: Optional[AdminAuth] = None
):
    """
    Delete a specific player's score from a specific difficulty
    Authorization required:
    - Users can delete their own scores (if authenticated via JWT token)
    - Admin can delete any score with admin password (sent in request body)
    """
    client_info = get_client_info(request)

    # Check admin password first (allows admin to delete any score)
    ADMIN_PASSWORD = os.getenv("LEADERBOARD_ADMIN_PASSWORD")
    if not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: LEADERBOARD_ADMIN_PASSWORD not set"
        )
    is_admin = auth and auth.admin_password and auth.admin_password == ADMIN_PASSWORD

    # If not admin, check if user is authenticated and deleting their own score
    if not is_admin:
        try:
            if not authorization:
                # Log unauthorized deletion attempt
                log_activity("score_delete_denied", {
                    **client_info,
                    "attempted_delete": player_name,
                    "difficulty": difficulty,
                    "reason": "no_authentication"
                })
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: Must be logged in to delete your own score, or use admin password"
                )

            user = get_current_user(authorization)

            # Check if user is trying to delete their own score
            if user['username'].lower() != player_name.lower():
                log_activity("score_delete_denied", {
                    **client_info,
                    "username": user['username'],
                    "attempted_delete": player_name,
                    "difficulty": difficulty,
                    "reason": "not_own_score"
                })
                raise HTTPException(
                    status_code=403,
                    detail="Forbidden: Can only delete your own scores, or use admin password"
                )
        except HTTPException:
            raise
        except Exception:
            # Log invalid token attempt
            log_activity("score_delete_denied", {
                **client_info,
                "attempted_delete": player_name,
                "difficulty": difficulty,
                "reason": "invalid_token"
            })
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid token or admin password required"
            )

    try:
        scores = load_scores()

        # Find and remove the score
        initial_count = len(scores)
        scores = [s for s in scores if not (
            s['name'].lower() == player_name.lower() and
            s.get('difficulty', 'easy') == difficulty
        )]

        if len(scores) == initial_count:
            raise HTTPException(status_code=404, detail=f"Score not found for {player_name} on {difficulty}")

        save_scores(scores)

        # Log successful deletion
        log_activity("score_deleted", {
            **client_info,
            "deleted_player": player_name,
            "difficulty": difficulty,
            "by_admin": is_admin
        })

        return {
            "status": "success",
            "message": f"Deleted score for {player_name} on {difficulty}",
            "remaining_scores": len(scores)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete score: {str(e)}")


# ============================================================================
# USER METRICS ENDPOINTS
# ============================================================================

@router.post("/metrics/submit", summary="Submit Game Metrics")
def submit_metrics(metrics: GameMetrics, authorization: Optional[str] = Header(None)):
    """
    Submit game session metrics (jumps, points, distance, time)
    Updates user's cumulative statistics
    Requires authentication
    """
    user = get_current_user(authorization)

    success = update_user_metrics(user['user_id'], metrics)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update metrics")

    # Get updated user stats
    users = load_users()
    updated_user = next((u for u in users if u['user_id'] == user['user_id']), None)

    return {
        "status": "success",
        "message": "Metrics updated successfully",
        "stats": {
            "total_games": updated_user.get('total_games', 0),
            "total_jumps": updated_user.get('total_jumps', 0),
            "total_points": updated_user.get('total_points', 0),
            "total_distance": updated_user.get('total_distance', 0),
            "total_playtime": updated_user.get('total_playtime', 0.0)
        }
    }


@router.get("/metrics/stats", summary="Get User Stats")
def get_user_stats(authorization: Optional[str] = Header(None)):
    """
    Get current user's statistics
    Requires authentication
    """
    user = get_current_user(authorization)

    return {
        "status": "success",
        "username": user['username'],
        "stats": {
            "total_games": user.get('total_games', 0),
            "total_jumps": user.get('total_jumps', 0),
            "total_points": user.get('total_points', 0),
            "total_distance": user.get('total_distance', 0),
            "total_playtime": user.get('total_playtime', 0.0)
        },
        "created_at": user['created_at']
    }


@router.get("/metrics/stats/{username}", summary="Get Any Player's Stats")
def get_player_stats(username: str):
    """
    Get any player's public statistics by username
    No authentication required - public endpoint

    Returns registered user stats if account exists, otherwise aggregates stats from scores
    """
    from datetime import datetime

    users = load_users()
    scores = load_scores()

    # Find user by username (case-insensitive)
    user = next((u for u in users if u['username'].lower() == username.lower()), None)

    if user:
        # Registered user - return their tracked stats
        return {
            "status": "success",
            "username": user['username'],
            "user_id": user['user_id'],
            "registered": True,
            "stats": {
                "total_games": user.get('total_games', 0),
                "total_jumps": user.get('total_jumps', 0),
                "total_points": user.get('total_points', 0),
                "total_distance": user.get('total_distance', 0),
                "total_playtime": user.get('total_playtime', 0.0)
            },
            "created_at": user['created_at']
        }

    # Not a registered user - check if they have scores
    player_scores = [s for s in scores if s['name'].lower() == username.lower()]

    if not player_scores:
        raise HTTPException(status_code=404, detail=f"Player '{username}' not found")

    # Aggregate stats from scores for non-registered player
    total_scores = len(player_scores)
    total_points = sum(s.get('score', 0) for s in player_scores)
    total_distance = sum(s.get('distance', 0) for s in player_scores)

    # Get earliest score timestamp
    timestamps = [s.get('timestamp') or s.get('date', '') for s in player_scores]
    first_played = min([t for t in timestamps if t]) if timestamps else datetime.utcnow().isoformat()

    return {
        "status": "success",
        "username": username,
        "user_id": None,
        "registered": False,
        "stats": {
            "total_games": total_scores,
            "total_jumps": 0,  # Not tracked for non-registered users
            "total_points": total_points,
            "total_distance": total_distance,
            "total_playtime": 0.0  # Not tracked for non-registered users
        },
        "created_at": first_played,
        "note": "This player is not registered. Stats are estimated from leaderboard scores."
    }


@router.post("/training/data", summary="Submit Training Data")
def submit_training_data(batch: TrainingDataBatch):
    """
    Submit a batch of training data from human gameplay
    Used for behavioral cloning / imitation learning

    Data includes state-action pairs: what the human did given the game state
    """
    from datetime import datetime
    import json

    try:
        # Create training data directory if it doesn't exist
        TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Organize by date and difficulty
        today = datetime.utcnow().strftime("%Y-%m-%d")
        date_dir = TRAINING_DATA_DIR / today
        date_dir.mkdir(exist_ok=True)

        # Create filename with session ID and timestamp
        timestamp = datetime.utcnow().strftime("%H-%M-%S")
        filename = f"{batch.session_id}_{batch.username}_{batch.difficulty}_{timestamp}.json"
        filepath = date_dir / filename

        # Prepare data for storage
        data_to_save = {
            "session_id": batch.session_id,
            "username": batch.username,
            "difficulty": batch.difficulty,
            "num_data_points": len(batch.data_points),
            "session_metadata": batch.session_metadata,
            "collected_at": datetime.utcnow().isoformat(),
            "data_points": [point.dict() for point in batch.data_points]
        }

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        return {
            "status": "success",
            "message": f"Saved {len(batch.data_points)} training data points",
            "filepath": str(filepath.relative_to(DATA_DIR)),
            "num_points": len(batch.data_points)
        }

    except Exception as e:
        print(f"Error saving training data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save training data: {str(e)}")
