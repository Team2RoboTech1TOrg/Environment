from scenarios.SprayingScenario import SprayingScenario
from scenarios.ExplorationScenario import ExplorationScenario
# from scenarios.AnimalsScenario import AnimalsScenario


def get_dict_scenarios(num_agents, grid_size) -> dict:
    """Create dictionary from scenarios"""
    spraying_scenario = SprayingScenario(num_agents, grid_size)
    exploration_scenario = ExplorationScenario(num_agents, grid_size)
    # animal_map = AnimalsScenario(num_agents, grid_size)
    return {
        1: spraying_scenario,
        2: exploration_scenario,
        # 3: animal_map
    }
