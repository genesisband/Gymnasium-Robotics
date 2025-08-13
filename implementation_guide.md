# Implementation Guide: Universal Reward Functions for Gymnasium-Robotics

## Overview

This guide provides step-by-step instructions for implementing and testing the universal reward function system in the Gymnasium-Robotics repository. We'll build a modular, extensible system that allows users to inject custom reward functions across multiple environments.

## Phase 1: Core Infrastructure

### 1.1 Environment Context System

Create the foundational `EnvContext` class that provides standardized access to environment information.

**File: `gymnasium_robotics/utils/env_context.py`**

```python
import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class EnvContext:
    """Standardized interface for accessing environment information across all robotics environments."""
    
    def __init__(self, env):
        self.env = env
        self.env_id = getattr(env.spec, 'id', 'unknown') if hasattr(env, 'spec') else 'unknown'
        self.env_type = self._detect_env_type(env)
        self._cached_metrics = {}
        
    def get_distance_metrics(self, achieved_goal=None, desired_goal=None) -> Dict[str, float]:
        """Compute standard distance metrics between achieved and desired goals."""
        if achieved_goal is None or desired_goal is None:
            # Try to get from current observation
            obs = getattr(self.env, '_last_obs', None)
            if obs and isinstance(obs, dict):
                achieved_goal = obs.get('achieved_goal')
                desired_goal = obs.get('desired_goal')
            else:
                return {'l1_distance': 0.0, 'l2_distance': 0.0, 'angular_distance': 0.0}
        
        # Position distance (first 3 elements typically position)
        pos_achieved = achieved_goal[:3] if len(achieved_goal) >= 3 else achieved_goal
        pos_desired = desired_goal[:3] if len(desired_goal) >= 3 else desired_goal
        
        l1_distance = np.sum(np.abs(pos_achieved - pos_desired))
        l2_distance = np.linalg.norm(pos_achieved - pos_desired)
        
        # Angular distance (if quaternions present)
        angular_distance = 0.0
        if len(achieved_goal) >= 7 and len(desired_goal) >= 7:
            quat_achieved = achieved_goal[3:7]
            quat_desired = desired_goal[3:7]
            # Quaternion distance: 1 - |q1 · q2|
            dot_product = np.abs(np.dot(quat_achieved, quat_desired))
            angular_distance = 1.0 - np.clip(dot_product, 0.0, 1.0)
        
        return {
            'l1_distance': float(l1_distance),
            'l2_distance': float(l2_distance),
            'angular_distance': float(angular_distance)
        }
    
    def get_robot_state(self) -> Dict[str, Any]:
        """Extract robot state information in standardized format."""
        robot_state = {}
        
        # Try to get joint information from different environment types
        if hasattr(self.env, 'data') and hasattr(self.env.data, 'qpos'):
            # MuJoCo environments
            robot_state['joint_positions'] = self.env.data.qpos.copy()
            robot_state['joint_velocities'] = self.env.data.qvel.copy()
        
        # Environment-specific extractions
        if self.env_type == "manipulation":
            robot_state.update(self._get_manipulation_state())
        elif self.env_type == "locomotion":
            robot_state.update(self._get_locomotion_state())
            
        return robot_state
    
    def get_object_states(self) -> List[Dict[str, Any]]:
        """Extract object state information."""
        objects = []
        
        # Try to extract object information based on environment type
        if hasattr(self.env, '_get_achieved_goal'):
            achieved_goal = self.env._get_achieved_goal()
            if len(achieved_goal) >= 3:
                obj_state = {
                    'position': achieved_goal[:3],
                    'orientation': achieved_goal[3:7] if len(achieved_goal) >= 7 else None,
                    'id': 'primary_object'
                }
                objects.append(obj_state)
        
        return objects
    
    def get_custom_data(self, key: str, default=None):
        """Safely access environment-specific data."""
        # Check environment attributes
        if hasattr(self.env, key):
            return getattr(self.env, key, default)
        
        # Check unwrapped environment
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, key):
            return getattr(self.env.unwrapped, key, default)
        
        # Check specific data stores
        if hasattr(self.env, '_custom_data') and key in self.env._custom_data:
            return self.env._custom_data[key]
            
        return default
    
    def _detect_env_type(self, env) -> str:
        """Automatically detect environment category based on environment ID and class."""
        env_id = self.env_id.lower()
        
        if any(keyword in env_id for keyword in ['hand', 'manipulate', 'fetch', 'pick', 'place', 'reach']):
            return "manipulation"
        elif any(keyword in env_id for keyword in ['ant', 'humanoid', 'walker', 'hopper', 'cheetah']):
            return "locomotion"
        elif any(keyword in env_id for keyword in ['maze', 'navigate', 'point']):
            return "navigation"
        elif 'kitchen' in env_id:
            return "manipulation"  # Kitchen environments are manipulation tasks
        else:
            return "custom"
    
    def _get_manipulation_state(self) -> Dict[str, Any]:
        """Extract manipulation-specific robot state."""
        state = {}
        
        # Try to get gripper/end-effector information
        if hasattr(self.env, 'data'):
            # Look for common gripper joint names
            gripper_joints = []
            for i, name in enumerate(getattr(self.env, '_model_names', {}).get('joint_names', [])):
                if any(grip in name.lower() for grip in ['gripper', 'finger']):
                    gripper_joints.append(i)
            
            if gripper_joints:
                state['gripper_positions'] = [self.env.data.qpos[i] for i in gripper_joints]
        
        # Add more manipulation-specific extractions as needed
        return state
    
    def _get_locomotion_state(self) -> Dict[str, Any]:
        """Extract locomotion-specific robot state."""
        state = {}
        
        # Try to get center of mass, velocity information
        if hasattr(self.env, 'data') and hasattr(self.env.data, 'qvel'):
            state['center_of_mass_velocity'] = np.linalg.norm(self.env.data.qvel[:3])
        
        return state
```

### 1.2 Universal Reward Interface

**File: `gymnasium_robotics/utils/reward_interface.py`**

```python
from typing import Callable, Union, Dict, Any
import numpy as np
from .env_context import EnvContext

# Type definition for universal reward functions
UniversalRewardFn = Callable[[np.ndarray, np.ndarray, Dict[str, Any], EnvContext], Union[float, np.ndarray]]


class RewardFunction:
    """Base class for reward functions with optional state management."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.state = {}
    
    def __call__(self, achieved_goal, desired_goal, info, env_context):
        """Compute reward. Override this method in subclasses."""
        return self.compute_reward(achieved_goal, desired_goal, info, env_context)
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        """Override this method to implement reward logic."""
        raise NotImplementedError
    
    def reset_state(self):
        """Reset any internal state. Called when environment resets."""
        self.state.clear()


class CompositeReward(RewardFunction):
    """Compose multiple reward functions with weights and conditions."""
    
    def __init__(self, components, **kwargs):
        """
        Args:
            components: List of (reward_fn, weight, condition) tuples
                       reward_fn: Reward function to call
                       weight: Multiplier for this component
                       condition: Optional function(env_context) -> bool
        """
        super().__init__(**kwargs)
        self.components = components
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        total_reward = 0.0
        
        for reward_fn, weight, condition in self.components:
            if condition is None or condition(env_context):
                component_reward = reward_fn(achieved_goal, desired_goal, info, env_context)
                total_reward += weight * component_reward
        
        return total_reward
    
    def reset_state(self):
        """Reset state for all components that support it."""
        super().reset_state()
        for reward_fn, _, _ in self.components:
            if hasattr(reward_fn, 'reset_state'):
                reward_fn.reset_state()
```

### 1.3 Core Reward Library

**File: `gymnasium_robotics/rewards/universal.py`**

```python
import numpy as np
from ..utils.reward_interface import RewardFunction


class SparseSuccessReward(RewardFunction):
    """Binary success reward - works on any goal-based environment."""
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        return 1.0 if info.get("is_success", False) else 0.0


class DistanceBasedReward(RewardFunction):
    """Dense distance reward - works on any environment with spatial goals."""
    
    def __init__(self, distance_type="l2", scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.distance_type = distance_type
        self.scale = scale
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        metrics = env_context.get_distance_metrics(achieved_goal, desired_goal)
        distance = metrics.get(f"{self.distance_type}_distance", 0.0)
        return -self.scale * distance


class ProgressReward(RewardFunction):
    """Reward based on progress toward goal, normalized by workspace size."""
    
    def __init__(self, max_distance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_distance = max_distance
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        metrics = env_context.get_distance_metrics(achieved_goal, desired_goal)
        distance = metrics["l2_distance"]
        
        # Try to get workspace size from environment
        workspace_size = env_context.get_custom_data("workspace_size", self.max_distance)
        
        progress = 1.0 - (distance / workspace_size)
        return np.clip(progress, 0.0, 1.0)


class SafetyAwareReward(RewardFunction):
    """Universal safety constraints that work across environment types."""
    
    def __init__(self, base_reward_fn, safety_penalty=5.0, **kwargs):
        super().__init__(**kwargs)
        self.base_reward_fn = base_reward_fn
        self.safety_penalty = safety_penalty
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        # Compute base reward
        base_reward = self.base_reward_fn(achieved_goal, desired_goal, info, env_context)
        
        # Apply safety penalties
        penalty = 0.0
        
        # Check for collisions
        if info.get("collision_detected", False):
            penalty += self.safety_penalty
        
        # Check robot state for safety violations
        robot_state = env_context.get_robot_state()
        
        # Joint limit violations (if joint positions available)
        joint_positions = robot_state.get("joint_positions", [])
        if len(joint_positions) > 0:
            # Assume joint limits are [-π, π] or normalized [-1, 1]
            max_joint = np.max(np.abs(joint_positions))
            if max_joint > 0.95:  # Close to limits
                penalty += 0.1 * (max_joint - 0.95) * 20  # Escalating penalty
        
        return base_reward - penalty
```

## Phase 2: Environment Integration

### 2.1 Modify Base Robot Environment

**File: `gymnasium_robotics/envs/robot_env.py`** (modifications)

Add to the `BaseRobotEnv` class:

```python
# Add to imports
from ..utils.env_context import EnvContext

class BaseRobotEnv(GoalEnv):
    def __init__(self, 
                 custom_reward_fn=None,
                 reward_type="replace",  # "replace", "add", "multiply"
                 **kwargs):
        # Store custom reward configuration
        self.custom_reward_fn = custom_reward_fn
        self.reward_type = reward_type
        self._env_context = None
        self._last_obs = None
        
        super().__init__(**kwargs)
    
    def compute_reward(self, achieved_goal, goal, info):
        # Always compute default reward for backwards compatibility
        default_reward = self._compute_default_reward(achieved_goal, goal, info)
        
        # If no custom reward function, return default
        if self.custom_reward_fn is None:
            return default_reward
        
        # Create environment context if needed
        if self._env_context is None:
            self._env_context = EnvContext(self)
        
        # Compute custom reward
        custom_reward = self.custom_reward_fn(achieved_goal, goal, info, self._env_context)
        
        # Combine rewards based on strategy
        if self.reward_type == "replace":
            return custom_reward
        elif self.reward_type == "add":
            return default_reward + custom_reward
        elif self.reward_type == "multiply":
            return default_reward * custom_reward
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
    
    def _compute_default_reward(self, achieved_goal, goal, info):
        """Compute the original reward. Subclasses should override this instead of compute_reward."""
        # This should contain the original compute_reward logic
        raise NotImplementedError("Subclasses must implement _compute_default_reward")
    
    def step(self, action):
        # Store observation for context
        obs, reward, terminated, truncated, info = super().step(action)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # Reset custom reward function state if it supports it
        if hasattr(self.custom_reward_fn, 'reset_state'):
            self.custom_reward_fn.reset_state()
        
        obs, info = super().reset(**kwargs)
        self._last_obs = obs
        return obs, info
```

### 2.2 Update Shadow Hand Environment

**File: `gymnasium_robotics/envs/shadow_dexterous_hand/manipulate.py`** (modifications)

Modify the `BaseManipulateEnv` class to support the new system:

```python
# In the compute_reward method, replace the current implementation with:

def _compute_default_reward(self, achieved_goal, goal, info):
    """Original reward computation for backwards compatibility."""
    # Move all the current reward strategy logic here
    # This preserves existing behavior when no custom reward is provided
    
    # keep track of how many times reward is queried (for drifting)
    self._time_step_count += 1

    # base reward values used by several strategies
    d_pos, d_rot = self._goal_distance(achieved_goal, goal)
    success = self._is_success(achieved_goal, goal).astype(np.float32)

    # default dense reward (identical to original implementation)
    dense_r = -(10.0 * d_pos + d_rot)
    sparse_r = success - 1.0

    # choose underlying true reward depending on self.reward_type
    r_true = sparse_r if self.reward_type == "sparse" else dense_r

    # Apply selected reward strategy (keep all existing strategy logic)
    strategy = self.reward_strategy
    
    if strategy == "default":
        return r_true.astype(np.float32)
    # ... keep all existing strategy implementations
    
    # Unknown strategy -> fallback to base behaviour
    return r_true.astype(np.float32)

def compute_reward(self, achieved_goal, goal, info):
    """Use the new universal reward system."""
    return super().compute_reward(achieved_goal, goal, info)
```

### 2.3 Universal Wrapper (Alternative Approach)

**File: `gymnasium_robotics/wrappers/custom_reward.py`**

```python
import gymnasium as gym
import numpy as np
from ..utils.env_context import EnvContext


class CustomRewardWrapper(gym.Wrapper):
    """Universal wrapper for injecting custom reward functions into any environment."""
    
    def __init__(self, env, reward_fn, reward_type="replace"):
        """
        Args:
            env: Environment to wrap
            reward_fn: Custom reward function following universal signature
            reward_type: "replace", "add", or "multiply"
        """
        super().__init__(env)
        self.reward_fn = reward_fn
        self.reward_type = reward_type
        self.env_context = EnvContext(env)
        self._last_obs = None
        
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        
        # Compute custom reward
        if self._is_goal_based_env():
            custom_reward = self.reward_fn(
                obs["achieved_goal"], 
                obs["desired_goal"], 
                info, 
                self.env_context
            )
        else:
            # For non-goal-based environments, use observation as both achieved/desired
            custom_reward = self.reward_fn(obs, obs, info, self.env_context)
        
        # Combine rewards
        final_reward = self._combine_rewards(original_reward, custom_reward)
        
        return obs, final_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # Reset reward function state
        if hasattr(self.reward_fn, 'reset_state'):
            self.reward_fn.reset_state()
        
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info
    
    def _is_goal_based_env(self):
        """Check if environment follows goal-based interface."""
        return (hasattr(self.env, 'compute_reward') and 
                hasattr(self.env.observation_space, 'spaces') and
                'achieved_goal' in self.env.observation_space.spaces)
    
    def _combine_rewards(self, original, custom):
        if self.reward_type == "replace":
            return custom
        elif self.reward_type == "add":
            return original + custom
        elif self.reward_type == "multiply":
            return original * custom
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
```

## Phase 3: Factory and Utilities

### 3.1 Environment Factory

**File: `gymnasium_robotics/utils/factory.py`**

```python
import gymnasium as gym
from ..wrappers.custom_reward import CustomRewardWrapper


def make_env_with_custom_reward(env_id, reward_fn, reward_type="replace", **env_kwargs):
    """
    Universal factory for creating any environment with custom rewards.
    
    Args:
        env_id: Standard gym environment ID
        reward_fn: Custom reward function or RewardFunction instance
        reward_type: How to combine with original reward ("replace", "add", "multiply")
        **env_kwargs: Additional environment arguments
    
    Returns:
        Environment with custom reward function applied
    """
    # Create base environment
    env = gym.make(env_id, **env_kwargs)
    
    # Check if environment natively supports custom rewards
    if hasattr(env.unwrapped, 'custom_reward_fn'):
        env.unwrapped.custom_reward_fn = reward_fn
        env.unwrapped.reward_type = reward_type
    else:
        # Apply universal wrapper
        env = CustomRewardWrapper(env, reward_fn, reward_type)
    
    return env


def make_multiple_envs_with_reward(env_ids, reward_fn, reward_type="replace", **env_kwargs):
    """Create multiple environments with the same custom reward function."""
    return [make_env_with_custom_reward(env_id, reward_fn, reward_type, **env_kwargs) 
            for env_id in env_ids]
```

### 3.2 Example Reward Library

**File: `gymnasium_robotics/rewards/examples.py`**

```python
import numpy as np
from ..utils.reward_interface import RewardFunction
from .universal import DistanceBasedReward, SparseSuccessReward


class ManipulationQualityReward(RewardFunction):
    """Manipulation-specific reward focusing on grasp quality and stability."""
    
    def __init__(self, stability_weight=0.3, force_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stability_weight = stability_weight
        self.force_weight = force_weight
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        if env_context.env_type != "manipulation":
            return 0.0  # Graceful degradation for non-manipulation environments
        
        reward = 0.0
        robot_state = env_context.get_robot_state()
        
        # Reward stable grasps
        if robot_state.get("object_grasped", False):
            reward += self.stability_weight
        
        # Penalize excessive forces
        grip_force = robot_state.get("grip_force", 0)
        if grip_force > 10.0:
            reward -= self.force_weight * (grip_force - 10.0)
        
        return reward


class CurriculumReward(RewardFunction):
    """Adaptive difficulty reward that changes over training."""
    
    def __init__(self, base_reward_fn, max_steps=100000, **kwargs):
        super().__init__(**kwargs)
        self.base_reward_fn = base_reward_fn
        self.max_steps = max_steps
        self.current_step = 0
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        # Get base reward
        base_reward = self.base_reward_fn(achieved_goal, desired_goal, info, env_context)
        
        # Adaptive difficulty
        difficulty = min(1.0, self.current_step / self.max_steps)
        
        # Adjust tolerance based on training progress
        metrics = env_context.get_distance_metrics(achieved_goal, desired_goal)
        distance = metrics["l2_distance"]
        
        # Start with loose tolerance, gradually tighten
        tolerance = 0.1 * (1.0 - difficulty) + 0.01 * difficulty  # 0.1 -> 0.01
        
        # Override success condition
        curriculum_success = distance < tolerance
        return 1.0 if curriculum_success else base_reward
    
    def reset_state(self):
        super().reset_state()
        self.current_step += 1  # Increment on episode reset
```

## Phase 4: Testing Framework

### 4.1 Unit Tests

**File: `tests/test_universal_rewards.py`**

```python
import pytest
import numpy as np
import gymnasium as gym
from gymnasium_robotics.utils.env_context import EnvContext
from gymnasium_robotics.utils.factory import make_env_with_custom_reward
from gymnasium_robotics.rewards.universal import SparseSuccessReward, DistanceBasedReward
from gymnasium_robotics.rewards.examples import ManipulationQualityReward


class TestEnvContext:
    """Test the EnvContext system."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        context = EnvContext(env)
        
        assert context.env_id == "HandManipulatePen_ContinuousTouchSensors-v1"
        assert context.env_type == "manipulation"
    
    def test_distance_metrics(self):
        """Test distance metric computation."""
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        context = EnvContext(env)
        
        achieved = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # position + quat
        desired = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        
        metrics = context.get_distance_metrics(achieved, desired)
        
        assert "l1_distance" in metrics
        assert "l2_distance" in metrics
        assert "angular_distance" in metrics
        assert metrics["l2_distance"] == pytest.approx(np.sqrt(3), rel=1e-5)


class TestUniversalRewards:
    """Test universal reward functions."""
    
    def test_sparse_success_reward(self):
        """Test sparse success reward function."""
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        context = EnvContext(env)
        reward_fn = SparseSuccessReward()
        
        # Test success case
        info_success = {"is_success": True}
        reward = reward_fn(None, None, info_success, context)
        assert reward == 1.0
        
        # Test failure case
        info_failure = {"is_success": False}
        reward = reward_fn(None, None, info_failure, context)
        assert reward == 0.0
    
    def test_distance_based_reward(self):
        """Test distance-based reward function."""
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        context = EnvContext(env)
        reward_fn = DistanceBasedReward(scale=2.0)
        
        achieved = np.array([0.0, 0.0, 0.0])
        desired = np.array([1.0, 0.0, 0.0])
        
        reward = reward_fn(achieved, desired, {}, context)
        assert reward == pytest.approx(-2.0, rel=1e-5)  # -scale * distance


class TestEnvironmentIntegration:
    """Test integration with actual environments."""
    
    def test_wrapper_integration(self):
        """Test custom reward wrapper."""
        def simple_reward(achieved, desired, info, context):
            return 10.0  # Fixed reward
        
        env = make_env_with_custom_reward(
            "HandManipulatePen_ContinuousTouchSensors-v1", 
            simple_reward,
            reward_type="replace"
        )
        
        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        
        assert reward == 10.0
    
    def test_multiple_environments(self):
        """Test same reward function across multiple environments."""
        def universal_reward(achieved, desired, info, context):
            if info.get("is_success", False):
                return 1.0
            else:
                metrics = context.get_distance_metrics(achieved, desired)
                return -0.1 * metrics["l2_distance"]
        
        env_ids = [
            "HandManipulatePen_ContinuousTouchSensors-v1",
            # Add more environment IDs as available
        ]
        
        for env_id in env_ids:
            env = make_env_with_custom_reward(env_id, universal_reward)
            obs, _ = env.reset()
            obs, reward, _, _, _ = env.step(env.action_space.sample())
            
            # Should work without errors
            assert isinstance(reward, (int, float, np.number))


class TestRewardComposition:
    """Test reward composition and complex scenarios."""
    
    def test_manipulation_specific_reward(self):
        """Test manipulation-specific reward function."""
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        context = EnvContext(env)
        reward_fn = ManipulationQualityReward()
        
        # Should return 0 for non-manipulation environments
        context.env_type = "locomotion"
        reward = reward_fn(None, None, {}, context)
        assert reward == 0.0
        
        # Should work for manipulation environments
        context.env_type = "manipulation"
        reward = reward_fn(None, None, {}, context)
        assert isinstance(reward, (int, float, np.number))


if __name__ == "__main__":
    pytest.main([__file__])
```

### 4.2 Integration Tests

**File: `tests/test_integration.py`**

```python
import numpy as np
import gymnasium as gym
from gymnasium_robotics.utils.factory import make_env_with_custom_reward
from gymnasium_robotics.rewards.universal import DistanceBasedReward, SparseSuccessReward
from gymnasium_robotics.utils.reward_interface import CompositeReward


def test_end_to_end_training():
    """Test that custom rewards work in a training-like scenario."""
    
    # Create composite reward
    composite_reward = CompositeReward([
        (SparseSuccessReward(), 10.0, None),  # Always active
        (DistanceBasedReward(scale=0.1), 1.0, None),  # Always active
    ])
    
    env = make_env_with_custom_reward(
        "HandManipulatePen_ContinuousTouchSensors-v1",
        composite_reward
    )
    
    # Run a short episode
    obs, _ = env.reset()
    total_reward = 0.0
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    # Should complete without errors
    assert isinstance(total_reward, (int, float, np.number))
    print(f"Total reward over episode: {total_reward}")


def test_cross_environment_reward():
    """Test the same reward function across different environment types."""
    
    def adaptive_reward(achieved, desired, info, context):
        """Reward that adapts to different environment types."""
        base_reward = -0.1 * context.get_distance_metrics(achieved, desired)["l2_distance"]
        
        if context.env_type == "manipulation":
            # Bonus for manipulation environments
            if info.get("is_success", False):
                base_reward += 5.0
        
        return base_reward
    
    # Test on available environments
    env_ids = [
        "HandManipulatePen_ContinuousTouchSensors-v1",
        # Add more when available: "FetchReach-v1", "AntMaze-v1", etc.
    ]
    
    for env_id in env_ids:
        print(f"Testing {env_id}")
        env = make_env_with_custom_reward(env_id, adaptive_reward)
        
        obs, _ = env.reset()
        obs, reward, _, _, info = env.step(env.action_space.sample())
        
        print(f"  Reward: {reward}, Success: {info.get('is_success', False)}")
        assert isinstance(reward, (int, float, np.number))


if __name__ == "__main__":
    test_end_to_end_training()
    test_cross_environment_reward()
    print("All integration tests passed!")
```

## Phase 5: Example Usage and Documentation

### 5.1 Example Scripts

**File: `examples/custom_reward_example.py`**

```python
#!/usr/bin/env python3
"""
Example: Using custom reward functions with Gymnasium-Robotics environments.

This script demonstrates how to:
1. Create custom reward functions
2. Apply them to different environments  
3. Use composite rewards
4. Test across multiple environment types
"""

import numpy as np
from gymnasium_robotics.utils.factory import make_env_with_custom_reward
from gymnasium_robotics.rewards.universal import SparseSuccessReward, DistanceBasedReward
from gymnasium_robotics.utils.reward_interface import CompositeReward


def demo_basic_custom_reward():
    """Demo 1: Basic custom reward function."""
    print("=== Demo 1: Basic Custom Reward ===")
    
    def my_reward(achieved_goal, desired_goal, info, env_context):
        """Simple distance-based reward with success bonus."""
        if info.get("is_success", False):
            return 10.0
        else:
            distance = env_context.get_distance_metrics(achieved_goal, desired_goal)["l2_distance"]
            return -distance
    
    env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", my_reward)
    
    obs, _ = env.reset()
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.3f}, success={info.get('is_success', False)}")
        
        if terminated or truncated:
            break


def demo_composite_reward():
    """Demo 2: Composite reward with multiple components."""
    print("\n=== Demo 2: Composite Reward ===")
    
    # Create composite reward
    composite = CompositeReward([
        (SparseSuccessReward(), 10.0, None),  # 10x weight for success
        (DistanceBasedReward(distance_type="l2", scale=0.1), 1.0, None),  # Small distance penalty
    ])
    
    env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", composite)
    
    obs, _ = env.reset()
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.3f}")
        
        if terminated or truncated:
            break


def demo_environment_adaptive_reward():
    """Demo 3: Reward that adapts to different environment types."""
    print("\n=== Demo 3: Environment-Adaptive Reward ===")
    
    def adaptive_reward(achieved_goal, desired_goal, info, env_context):
        """Reward that behaves differently based on environment type."""
        base_reward = 0.0
        
        # Universal success bonus
        if info.get("is_success", False):
            base_reward += 10.0
        
        # Environment-specific components
        if env_context.env_type == "manipulation":
            # Focus on precise positioning for manipulation
            distance = env_context.get_distance_metrics(achieved_goal, desired_goal)["l2_distance"]
            base_reward -= 2.0 * distance
            
            # Bonus for stable grasps
            robot_state = env_context.get_robot_state()
            if robot_state.get("object_grasped", False):
                base_reward += 1.0
                
        elif env_context.env_type == "locomotion":
            # Focus on energy efficiency for locomotion
            robot_state = env_context.get_robot_state()
            velocity = robot_state.get("center_of_mass_velocity", 0)
            base_reward -= 0.01 * velocity  # Penalize high speeds
            
        return base_reward
    
    # Test on manipulation environment
    env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", adaptive_reward)
    
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, _, _, info = env.step(action)
    print(f"Manipulation env: reward={reward:.3f}, env_type=manipulation")


def demo_training_simulation():
    """Demo 4: Simulate a short training run with custom rewards."""
    print("\n=== Demo 4: Training Simulation ===")
    
    def training_reward(achieved_goal, desired_goal, info, env_context):
        """Shaped reward for training."""
        reward = 0.0
        
        # Task completion
        if info.get("is_success", False):
            reward += 20.0
        
        # Distance shaping
        metrics = env_context.get_distance_metrics(achieved_goal, desired_goal)
        reward -= metrics["l2_distance"]
        
        # Small time penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", training_reward)
    
    total_rewards = []
    
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode}: total_reward={episode_reward:.2f}")
    
    print(f"Average reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    demo_basic_custom_reward()
    demo_composite_reward()
    demo_environment_adaptive_reward()
    demo_training_simulation()
    print("\nAll demos completed successfully!")
```

### 5.2 Benchmark Script

**File: `examples/benchmark_rewards.py`**

```python
#!/usr/bin/env python3
"""
Benchmark script to compare different reward functions.

This script helps evaluate:
1. Performance differences between reward functions
2. Training convergence rates
3. Final task success rates
"""

import time
import numpy as np
from gymnasium_robotics.utils.factory import make_env_with_custom_reward
from gymnasium_robotics.rewards.universal import SparseSuccessReward, DistanceBasedReward


def benchmark_reward_function(reward_fn, env_id, num_episodes=10, max_steps=50):
    """Benchmark a reward function on an environment."""
    env = make_env_with_custom_reward(env_id, reward_fn)
    
    episode_rewards = []
    success_rates = []
    computation_times = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        
        start_time = time.time()
        
        for step in range(max_steps):
            action = env.action_space.sample()  # Random actions for benchmark
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        end_time = time.time()
        
        episode_rewards.append(episode_reward)
        success_rates.append(float(info.get("is_success", False)))
        computation_times.append(end_time - start_time)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean(success_rates),
        'mean_compute_time': np.mean(computation_times),
        'episode_rewards': episode_rewards
    }


def run_benchmarks():
    """Run benchmarks comparing different reward functions."""
    
    # Define reward functions to test
    reward_functions = {
        'sparse': SparseSuccessReward(),
        'dense_l2': DistanceBasedReward(distance_type="l2", scale=1.0),
        'dense_l1': DistanceBasedReward(distance_type="l1", scale=1.0),
        'combined': lambda achieved, desired, info, ctx: (
            10.0 if info.get("is_success", False) else 
            -ctx.get_distance_metrics(achieved, desired)["l2_distance"]
        )
    }
    
    env_id = "HandManipulatePen_ContinuousTouchSensors-v1"
    
    print(f"Benchmarking reward functions on {env_id}")
    print("=" * 60)
    
    results = {}
    
    for name, reward_fn in reward_functions.items():
        print(f"Testing {name}...")
        results[name] = benchmark_reward_function(reward_fn, env_id)
        
        print(f"  Mean reward: {results[name]['mean_reward']:.3f} ± {results[name]['std_reward']:.3f}")
        print(f"  Success rate: {results[name]['success_rate']:.1%}")
        print(f"  Compute time: {results[name]['mean_compute_time']:.4f}s")
        print()
    
    # Summary comparison
    print("Summary Comparison:")
    print("-" * 40)
    print(f"{'Reward Function':<15} {'Mean Reward':<12} {'Success Rate':<12} {'Compute Time':<12}")
    print("-" * 40)
    
    for name, result in results.items():
        print(f"{name:<15} {result['mean_reward']:>8.3f}    {result['success_rate']:>8.1%}     {result['mean_compute_time']:>8.4f}s")


if __name__ == "__main__":
    run_benchmarks()
```

## Phase 6: Deployment and Testing

### 6.1 Installation and Setup

**File: `setup_universal_rewards.py`**

```python
#!/usr/bin/env python3
"""
Setup script for universal reward system.

Run this script to verify that all components are working correctly.
"""

import sys
import importlib
import gymnasium as gym


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    
    try:
        from gymnasium_robotics.utils.env_context import EnvContext
        from gymnasium_robotics.utils.reward_interface import RewardFunction, CompositeReward
        from gymnasium_robotics.rewards.universal import SparseSuccessReward, DistanceBasedReward
        from gymnasium_robotics.utils.factory import make_env_with_custom_reward
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def check_environment_compatibility():
    """Check that environments work with the new system."""
    print("\nChecking environment compatibility...")
    
    try:
        env = gym.make("HandManipulatePen_ContinuousTouchSensors-v1")
        print("✓ Environment creation successful")
        
        # Test basic functionality
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("✓ Environment step successful")
        
        return True
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False


def check_custom_reward_integration():
    """Check that custom rewards work."""
    print("\nChecking custom reward integration...")
    
    try:
        from gymnasium_robotics.utils.factory import make_env_with_custom_reward
        
        def test_reward(achieved, desired, info, context):
            return 1.0
        
        env = make_env_with_custom_reward(
            "HandManipulatePen_ContinuousTouchSensors-v1", 
            test_reward
        )
        
        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(env.action_space.sample())
        
        if reward == 1.0:
            print("✓ Custom reward integration successful")
            return True
        else:
            print(f"✗ Custom reward not applied correctly (got {reward}, expected 1.0)")
            return False
            
    except Exception as e:
        print(f"✗ Custom reward error: {e}")
        return False


def run_full_test():
    """Run comprehensive test suite."""
    print("Running universal reward system tests...")
    print("=" * 50)
    
    tests = [
        check_imports,
        check_environment_compatibility, 
        check_custom_reward_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Universal reward system is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
```

### 6.2 Continuous Integration

**File: `.github/workflows/test_universal_rewards.yml`**

```yaml
name: Test Universal Rewards

on:
  push:
    branches: [ main, universal-rewards ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run setup verification
      run: python setup_universal_rewards.py
    
    - name: Run unit tests
      run: pytest tests/test_universal_rewards.py -v
    
    - name: Run integration tests
      run: pytest tests/test_integration.py -v
    
    - name: Run example scripts
      run: |
        python examples/custom_reward_example.py
        python examples/benchmark_rewards.py
```

## Phase 7: Documentation and Examples

### 7.1 API Documentation

**File: `docs/api/universal_rewards.md`**

```markdown
# Universal Rewards API Reference

## Core Classes

### EnvContext

Provides standardized access to environment information.

```python
class EnvContext:
    def __init__(self, env)
    def get_distance_metrics(self, achieved_goal=None, desired_goal=None) -> Dict[str, float]
    def get_robot_state(self) -> Dict[str, Any]
    def get_object_states(self) -> List[Dict[str, Any]]
    def get_custom_data(self, key: str, default=None)
```

### RewardFunction

Base class for stateful reward functions.

```python
class RewardFunction:
    def __init__(self, **kwargs)
    def compute_reward(self, achieved_goal, desired_goal, info, env_context)
    def reset_state(self)
```

## Factory Functions

### make_env_with_custom_reward

```python
def make_env_with_custom_reward(env_id, reward_fn, reward_type="replace", **env_kwargs)
```

Creates an environment with a custom reward function.

**Parameters:**
- `env_id`: Gymnasium environment ID
- `reward_fn`: Custom reward function or RewardFunction instance  
- `reward_type`: How to combine with original reward ("replace", "add", "multiply")
- `**env_kwargs`: Additional environment arguments

## Built-in Reward Functions

### SparseSuccessReward
Binary reward based on task success.

### DistanceBasedReward
Dense reward based on distance to goal.

### CompositeReward
Combines multiple reward functions with weights.
```

### 7.2 User Guide

**File: `docs/guides/custom_rewards.md`**

```markdown
# Custom Rewards User Guide

## Quick Start

### 1. Basic Custom Reward

```python
from gymnasium_robotics.utils.factory import make_env_with_custom_reward

def my_reward(achieved_goal, desired_goal, info, env_context):
    if info.get("is_success", False):
        return 10.0
    else:
        distance = env_context.get_distance_metrics(achieved_goal, desired_goal)["l2_distance"]
        return -distance

env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", my_reward)
```

### 2. Using Built-in Rewards

```python
from gymnasium_robotics.rewards.universal import DistanceBasedReward

reward_fn = DistanceBasedReward(distance_type="l2", scale=2.0)
env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", reward_fn)
```

### 3. Composite Rewards

```python
from gymnasium_robotics.utils.reward_interface import CompositeReward
from gymnasium_robotics.rewards.universal import SparseSuccessReward, DistanceBasedReward

composite = CompositeReward([
    (SparseSuccessReward(), 10.0, None),
    (DistanceBasedReward(), 0.1, None),
])

env = make_env_with_custom_reward("HandManipulatePen_ContinuousTouchSensors-v1", composite)
```

## Advanced Usage

### Environment-Specific Rewards

```python
def adaptive_reward(achieved_goal, desired_goal, info, env_context):
    base_reward = 0.0
    
    if env_context.env_type == "manipulation":
        # Manipulation-specific logic
        pass
    elif env_context.env_type == "locomotion":
        # Locomotion-specific logic  
        pass
    
    return base_reward
```

### Stateful Rewards

```python
from gymnasium_robotics.utils.reward_interface import RewardFunction

class MyStatefulReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.episode_count = 0
    
    def compute_reward(self, achieved_goal, desired_goal, info, env_context):
        # Use self.episode_count in reward computation
        return some_reward
    
    def reset_state(self):
        self.episode_count += 1
```

## Best Practices

1. **Start Simple**: Begin with basic rewards and add complexity gradually
2. **Test Across Environments**: Verify your rewards work on multiple environment types
3. **Handle Edge Cases**: Use graceful degradation for unsupported environments
4. **Monitor Performance**: Benchmark reward computation time for training efficiency
5. **Document Assumptions**: Clearly document what your reward function expects
```

This comprehensive implementation guide provides everything needed to build, test, and deploy the universal reward function system in the Gymnasium-Robotics repository. The phased approach allows for incremental development and testing, ensuring robust functionality across the entire ecosystem. 