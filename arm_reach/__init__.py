from gymnasium.envs.registration import register

register(
    id="ArmReach-v0",
    entry_point="arm_reach.envs:ArmReachEnv"
)
