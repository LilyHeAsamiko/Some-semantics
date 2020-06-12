#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:57:15 2020

@author: lilyheasamiko
"""
import argparse
from gym.spaces import Discrete, Tuple
import logging

import ray
from ray import tune
from ray.tune import function
import CorrelatedActionsEnv
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument("--flat", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--stop-reward", type=float, default=0.0)
parser.add_argument("--stop-timesteps", type=int, default=100000)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    #if args.flat:
    #    results = tune.run(
    #        "PPO",
    #        stop=stop,
    #        config={
    #            "env": CorrelatedActionsEnv,
    #            "num_workers": 0,
    #            "framework": "torch" if args.torch else "tf",
    #        },
    #    )
    #else:
    #    maze = CorrelatedActionsEnv(None)

        def policy_mapping_fn(agent_id):
            if agent_id.startswith("low_level_"):
                return "low_level_policy"
            elif agent_id.startswith("mid_level_"):
                return "mid_level_policy"
            else:         
                return "high_level_policy"

        config = {
            "env": HierarchicalWindyMazeEnv,
            "num_workers": 0,
            "log_level": "INFO",
            "entropy_coeff": 0.01,
            "multiagent": {
                "policies": {
                    "high_level_policy": (None, maze.observation_space,
                                          Discrete(4), {
                                              "gamma": 0.9
                                          }),
                    "low_level_policy": (None,
                                         Tuple([
                                             maze.observation_space,
                                             Discrete(4)
                                         ]), maze.action_space, {
                                             "gamma": 0.0
                                         }),
                },
                "policy_mapping_fn": function(policy_mapping_fn),
            },
            "framework": "torch" if args.torch else "tf",
        }

        results = tune.run("PPO", stop=stop, config=config)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()