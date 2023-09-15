import rlhw.agent as agents
from rlhw.env import env_builder
from rlhw.valueiter import FLVI
import numpy as np


def main():
    map_name = "4x4"
    # map_name = "8x8"
    params = {
        "render": True,
        "discount_factor": 0.95,
        "epsilon": 0.99,
        "epsilon_decay": 1,
        "min_epsilon": 0.01,
        "learning_rate": 0.1,
        "min_learning_rate": 0.01,
        "learning_rate_decay": 1,
    }
    is_slippery = True
    # params = {
    #     "render": True,
    #     "discount_factor": 0.95,
    #     "epsilon": 1,
    #     "epsilon_decay": 0.9999,
    #     "min_epsilon": 0.01,
    #     "learning_rate": 1,
    #     "min_learning_rate": 0.01,
    #     "learning_rate_decay": 0.9999,
    # }
    build_env = env_builder(
        "FrozenLake-v1", {"map_name": map_name, "is_slippery": is_slippery}
    )
    # agent = agents.RandomAgent(build_env, params)
    # agent.run(10000, 100, True)
    # eval_res = agent.run(100, 100, True) ## modified false to true
    # total = sum(list(eval_res.values()))
    # success_rate = total / 100.0
    # print(f"success_rate: {success_rate}")

    # np.set_printoptions(suppress=True)
    # learned = np.max(agent.policy, axis=1)
    # print("Learned state values:\n", learned)
    # computed = FLVI(map_name, params["discount_factor"], is_slippery)
    # print("True state values:\n", computed)
    # print("Value error norm: ", np.linalg.norm(learned - computed))

    agent = agents.QLearningAgent(build_env, params)
    # agent = agents.QLearningAgent(build_env, params)
    agent.run(10000, 100, True)
    eval_res = agent.run(100, 100, False) ## modified false to true
    total = sum(list(eval_res.values()))
    success_rate = total / 100.0
    print(f"success_rate: {success_rate}")
    print("Policy: ", agent.policy)


    np.set_printoptions(suppress=True)
    learned = np.max(agent.policy, axis=1)
    print("Learned state values:\n", learned)
    computed = FLVI(map_name, params["discount_factor"], is_slippery)
    print("True state values:\n", computed)
    print("Value error norm: ", np.linalg.norm(learned - computed))


if __name__ == "__main__":
    main()
