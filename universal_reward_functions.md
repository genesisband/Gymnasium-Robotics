# Universal Reward Functions for Multi-Environment Gym Ecosystems

## Overview

This document outlines a design for implementing **universal reward functions** that can work seamlessly across multiple different Gymnasium environments. The goal is to enable researchers and practitioners to write reward functions once and apply them to various robotics environments (manipulation, locomotion, navigation, etc.) without modification.

## Core Concept

The foundation is a **callback/function injection pattern** with a standardized interface that provides enough context for environment-specific logic while maintaining simplicity and flexibility.

## Universal Reward Function Interface

### Standard Signature

All reward functions follow this universal signature:

```python
def universal_reward_fn(achieved_goal, desired_goal, info, env_context):
    """
    Universal reward function signature
    
    Args:
        achieved_goal: What the agent actually achieved (numpy array)
        desired_goal: What the agent was supposed to achieve (numpy array)
        info: Environment-specific info dictionary
        env_context: Standardized context object with environment details
        
    Returns:
        reward: Float or numpy array of reward values
    """
    return reward
```

### Key Design Principles

1. **Environment Agnostic**: Works across different environment types
2. **Context Rich**: Provides access to environment internals when needed
3. **Type Safe**: Clear parameter types and return values
4. **Backwards Compatible**: Doesn't break existing implementations
5. **Performance Conscious**: Minimal overhead for reward computation

## Standardized Environment Context

### EnvContext Class

The `EnvContext` object provides consistent access to environment internals:

```python
class EnvContext:
    """Standardized interface for accessing environment information"""
    
    def __init__(self, env):
        self.env_id = env.spec.id
        self.env_type = self._detect_env_type(env)  # "manipulation", "locomotion", "navigation"
        self.observation = env._last_observation
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.timestep = getattr(env, '_timestep', 0)
        
    def get_distance_metrics(self):
        """Standard distance calculations for any environment"""
        # Returns dict with l1_distance, l2_distance, angular_distance, etc.
        
    def get_robot_state(self):
        """Extract robot state in standardized format"""
        # Returns dict with joint_positions, joint_velocities, end_effector_pose, etc.
        
    def get_object_states(self):
        """Extract object states if applicable"""
        # Returns list of dicts with position, orientation, velocity for each object
        
    def get_custom_data(self, key):
        """Access environment-specific data"""
        # Safe access to environment-specific attributes
        
    def _detect_env_type(self, env):
        """Automatically detect environment category"""
        # Logic to classify environment type
```

### Environment Type Detection

Automatic classification of environments into categories:

- **manipulation**: Hand/arm manipulation tasks (HandManipulate*, Fetch*, etc.)
- **locomotion**: Moving agents (Ant*, Humanoid*, Walker*, etc.)  
- **navigation**: Spatial navigation (AntMaze*, PointMaze*, etc.)
- **multi_agent**: Multi-agent environments
- **custom**: User-defined or unclassified environments

## Environment-Agnostic Reward Functions

### Basic Examples

```python
def sparse_success_reward(achieved_goal, desired_goal, info, env_context):
    """Binary success reward - works on any goal-based environment"""
    return 1.0 if info.get("is_success", False) else 0.0

def distance_based_reward(achieved_goal, desired_goal, info, env_context):
    """Dense distance reward - works on any environment with spatial goals"""
    distance = env_context.get_distance_metrics()["l2_distance"]
    return -distance

def normalized_progress_reward(achieved_goal, desired_goal, info, env_context):
    """Progress-based reward normalized by environment scale"""
    metrics = env_context.get_distance_metrics()
    max_distance = env_context.get_custom_data("workspace_size") or 1.0
    progress = 1.0 - (metrics["l2_distance"] / max_distance)
    return np.clip(progress, 0.0, 1.0)
```

### Environment-Specific Adaptations

```python
def adaptive_manipulation_reward(achieved_goal, desired_goal, info, env_context):
    """Adapts behavior based on environment type"""
    base_reward = -env_context.get_distance_metrics()["l2_distance"]
    
    if env_context.env_type == "manipulation":
        # Add manipulation-specific bonuses
        robot_state = env_context.get_robot_state()
        if robot_state.get("object_grasped", False):
            base_reward += 0.5
            
        # Penalize excessive grip force
        grip_force = robot_state.get("grip_force", 0)
        if grip_force > 10.0:
            base_reward -= 0.1 * (grip_force - 10.0)
            
    elif env_context.env_type == "locomotion":
        # Add locomotion-specific considerations
        robot_state = env_context.get_robot_state()
        energy_cost = robot_state.get("energy_consumption", 0)
        base_reward -= 0.01 * energy_cost
        
    return base_reward

def multi_objective_reward(achieved_goal, desired_goal, info, env_context):
    """Combines multiple reward components with environment-specific weights"""
    components = {}
    
    # Universal components
    components["task_success"] = 10.0 if info.get("is_success") else 0.0
    components["goal_distance"] = -env_context.get_distance_metrics()["l2_distance"]
    
    # Environment-specific components
    if env_context.env_type == "manipulation":
        components["manipulation_quality"] = _compute_manipulation_quality(env_context)
        weights = {"task_success": 1.0, "goal_distance": 0.1, "manipulation_quality": 0.3}
    elif env_context.env_type == "locomotion":
        components["movement_efficiency"] = _compute_movement_efficiency(env_context) 
        weights = {"task_success": 1.0, "goal_distance": 0.2, "movement_efficiency": 0.1}
    else:
        weights = {"task_success": 1.0, "goal_distance": 0.1}
    
    return sum(weights.get(k, 0) * v for k, v in components.items())
```

## Integration Approaches

### 1. Base Environment Modification

Modify existing base classes to support custom reward injection:

```python
class BaseRobotEnv(GoalEnv):
    def __init__(self, custom_reward_fn=None, **kwargs):
        self.custom_reward_fn = custom_reward_fn
        super().__init__(**kwargs)
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.custom_reward_fn:
            env_context = EnvContext(self)
            return self.custom_reward_fn(achieved_goal, desired_goal, info, env_context)
        else:
            return self._default_compute_reward(achieved_goal, desired_goal, info)
            
    def _default_compute_reward(self, achieved_goal, desired_goal, info):
        """Original reward computation - maintains backwards compatibility"""
        # Original implementation
```

### 2. Universal Wrapper Approach

Create a wrapper that can be applied to any environment:

```python
class CustomRewardWrapper(gym.Wrapper):
    """Universal wrapper for injecting custom reward functions"""
    
    def __init__(self, env, reward_fn, reward_type="replace"):
        """
        Args:
            env: The environment to wrap
            reward_fn: Custom reward function following universal signature
            reward_type: "replace", "add", or "multiply" - how to combine with original
        """
        super().__init__(env)
        self.reward_fn = reward_fn
        self.reward_type = reward_type
        
    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Create environment context
        env_context = EnvContext(self.env)
        
        # Compute custom reward
        if self._is_goal_based_env():
            custom_reward = self.reward_fn(
                obs["achieved_goal"], 
                obs["desired_goal"], 
                info, 
                env_context
            )
        else:
            # Adapt interface for non-goal-based environments
            custom_reward = self.reward_fn(
                obs, obs, info, env_context  # Use observation as both achieved/desired
            )
        
        # Combine rewards based on strategy
        final_reward = self._combine_rewards(original_reward, custom_reward)
        
        return obs, final_reward, terminated, truncated, info
        
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

### 3. Environment Factory Pattern

Provide a universal factory for creating environments with custom rewards:

```python
def make_env_with_custom_reward(env_id, reward_fn, reward_type="replace", **env_kwargs):
    """
    Universal factory for creating any environment with custom rewards
    
    Args:
        env_id: Standard gym environment ID
        reward_fn: Custom reward function
        reward_type: How to combine with original reward
        **env_kwargs: Additional environment arguments
    """
    env = gym.make(env_id, **env_kwargs)
    
    # Check if environment natively supports custom rewards
    if hasattr(env.unwrapped, 'custom_reward_fn'):
        env.unwrapped.custom_reward_fn = reward_fn
    else:
        # Apply universal wrapper
        env = CustomRewardWrapper(env, reward_fn, reward_type)
    
    return env

# Usage examples
hand_env = make_env_with_custom_reward("HandManipulatePen-v1", my_reward_fn)
fetch_env = make_env_with_custom_reward("FetchReach-v1", my_reward_fn) 
ant_env = make_env_with_custom_reward("AntMaze-v1", my_reward_fn)
```

## Cross-Environment Reward Libraries

### Organized Library Structure

```python
# universal_rewards/manipulation.py
def grasp_quality_reward(achieved_goal, desired_goal, info, env_context):
    """Reward based on grasp stability and force distribution"""
    if env_context.env_type != "manipulation":
        return 0.0  # Graceful degradation
    
    robot_state = env_context.get_robot_state()
    # Implementation for grasp quality assessment

def object_stability_reward(achieved_goal, desired_goal, info, env_context):
    """Reward for maintaining object stability during manipulation"""
    # Implementation

# universal_rewards/locomotion.py  
def energy_efficiency_reward(achieved_goal, desired_goal, info, env_context):
    """Reward energy-efficient movement patterns"""
    # Implementation
    
def smoothness_reward(achieved_goal, desired_goal, info, env_context):
    """Reward smooth, non-jerky movements"""
    # Implementation

# universal_rewards/navigation.py
def exploration_bonus_reward(achieved_goal, desired_goal, info, env_context):
    """Bonus for exploring new areas"""
    # Implementation

# universal_rewards/universal.py
def curiosity_driven_reward(achieved_goal, desired_goal, info, env_context):
    """Information-theoretic curiosity reward"""
    # Implementation that works across all environment types

def safety_aware_reward(achieved_goal, desired_goal, info, env_context):
    """Penalizes unsafe behaviors across any environment"""
    # Implementation
```

### Reward Composition Framework

```python
class CompositeReward:
    """Compose multiple reward functions with weights"""
    
    def __init__(self, reward_components):
        """
        Args:
            reward_components: List of (reward_fn, weight, condition) tuples
        """
        self.components = reward_components
    
    def __call__(self, achieved_goal, desired_goal, info, env_context):
        total_reward = 0.0
        
        for reward_fn, weight, condition in self.components:
            if condition is None or condition(env_context):
                component_reward = reward_fn(achieved_goal, desired_goal, info, env_context)
                total_reward += weight * component_reward
                
        return total_reward

# Usage example
my_reward = CompositeReward([
    (sparse_success_reward, 10.0, None),  # Always active
    (distance_based_reward, 0.1, None),   # Always active  
    (grasp_quality_reward, 0.5, lambda ctx: ctx.env_type == "manipulation"),  # Conditional
    (energy_efficiency_reward, 0.2, lambda ctx: ctx.env_type == "locomotion")  # Conditional
])
```

## Implementation Examples

### Research Use Case

```python
# Researcher studying curriculum learning across multiple environments
def curriculum_reward(achieved_goal, desired_goal, info, env_context):
    """Reward that adapts difficulty over time"""
    base_reward = sparse_success_reward(achieved_goal, desired_goal, info, env_context)
    
    # Get training progress (injected via info or env_context)
    training_step = env_context.get_custom_data("training_step") or 0
    difficulty = min(1.0, training_step / 100000)  # Ramp up over 100k steps
    
    # Adaptive tolerance based on difficulty
    distance = env_context.get_distance_metrics()["l2_distance"]
    tolerance = 0.1 * (1.0 - difficulty) + 0.01 * difficulty  # 0.1 -> 0.01
    
    success = distance < tolerance
    return 1.0 if success else -0.01 * distance

# Apply to multiple environments
envs = [
    make_env_with_custom_reward("HandManipulatePen-v1", curriculum_reward),
    make_env_with_custom_reward("FetchReach-v1", curriculum_reward),
    make_env_with_custom_reward("AntMaze-v1", curriculum_reward)
]
```

### Production Use Case

```python
# Production system with safety constraints
def safe_manipulation_reward(achieved_goal, desired_goal, info, env_context):
    """Production-ready reward with safety constraints"""
    # Task performance
    task_reward = distance_based_reward(achieved_goal, desired_goal, info, env_context)
    
    # Safety constraints
    safety_penalty = 0.0
    robot_state = env_context.get_robot_state()
    
    # Force limits
    if robot_state.get("max_force", 0) > 50.0:  # Newtons
        safety_penalty += 10.0
    
    # Joint limits
    joint_positions = robot_state.get("joint_positions", [])
    if any(abs(pos) > 0.9 for pos in joint_positions):  # Normalized joint limits
        safety_penalty += 5.0
    
    # Collision detection
    if info.get("collision_detected", False):
        safety_penalty += 20.0
    
    return task_reward - safety_penalty

# Deploy across production environments
production_envs = [
    make_env_with_custom_reward("HandManipulatePen-v1", safe_manipulation_reward),
    make_env_with_custom_reward("FetchPickAndPlace-v1", safe_manipulation_reward)
]
```

## Benefits and Advantages

### For Researchers
- **Reproducibility**: Share exact reward implementations across papers
- **Rapid Prototyping**: Test reward hypotheses across multiple environments quickly
- **Standardized Comparisons**: Compare algorithms using identical reward functions
- **Cross-Domain Transfer**: Study how reward functions transfer between domains

### For Practitioners  
- **Reusable Components**: Build libraries of production-tested reward functions
- **Domain Expertise**: Leverage domain-specific rewards from experts
- **Safety Integration**: Consistently apply safety constraints across systems
- **Rapid Deployment**: Deploy proven reward strategies to new environments

### For the Community
- **Knowledge Sharing**: Central repository of reward function strategies
- **Best Practices**: Established patterns for common reward scenarios
- **Extensibility**: Easy to add support for new environment types
- **Backwards Compatibility**: Existing code continues to work unchanged

## Future Extensions

### Advanced Features
- **Automatic Hyperparameter Tuning**: Environment-specific parameter optimization
- **Reward Learning**: Integration with inverse reinforcement learning
- **Multi-Task Rewards**: Reward functions that handle multiple simultaneous objectives
- **Temporal Rewards**: Rewards that consider action sequences over time

### Integration Possibilities
- **Stable-Baselines3**: Native support in popular RL libraries
- **Weights & Biases**: Automatic logging of reward components
- **Ray/RLLib**: Distributed training with custom rewards
- **Custom Gym Wrappers**: Specialized wrappers for specific research needs

## Conclusion

This universal reward function architecture enables a **write-once, use-everywhere** approach for reward engineering in robotics environments. By providing a standardized interface with rich contextual information, researchers and practitioners can develop sophisticated reward functions that work seamlessly across the entire Gymnasium robotics ecosystem.

The modular design promotes code reuse, facilitates reproducible research, and accelerates the development of robust robotic learning systems. As the community adopts this pattern, we can build a comprehensive library of proven reward strategies that benefit everyone in the field. 