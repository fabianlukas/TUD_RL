from gym.envs.registration import register 

register(
    id="MyMountainCar-v0", 
    entry_point="tud_rl.envs:MountainCar",
)

register(
    id="Ski-v0", 
    entry_point="tud_rl.envs:Ski",
)

register(
    id="ObstacleAvoidance-v0", 
    entry_point="tud_rl.envs:ObstacleAvoidance",
)

register(
    id="ObstacleAvoidance_CRstudy-v0", 
    entry_point="tud_rl.envs:ObstacleAvoidance_CRstudy",
)

register(
    id="ObstacleAvoidance_CRstudy_box-v0", 
    entry_point="tud_rl.envs:ObstacleAvoidance_CRstudy_box",
)

register(
    id="FossenEnv-v0", 
    entry_point="tud_rl.envs:FossenEnv",
)

register(
    id="FossenEnvScenarioOne-v0", 
    entry_point="tud_rl.envs:FossenEnvScenarioOne",
)
