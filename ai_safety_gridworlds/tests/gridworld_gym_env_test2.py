# Copyright 2023 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import time
import pytest
import numpy as np
import numpy.testing as npt

try:
    from gymnasium.spaces import Discrete, MultiDiscrete
    from gymnasium.utils.env_checker import check_env
    gym_v26 = True
except:
    from gym.spaces import Discrete, MultiDiscrete
    from gym.utils.env_checker import check_env
    gym_v26 = False

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface_ma
from ai_safety_gridworlds.environments.shared.rl import pycolab_interface_mo
from ai_safety_gridworlds.environments.shared.safety_game import Actions
from ai_safety_gridworlds.helpers.gridworld_gym_env import GridworldGymEnv


def demos1():
    _demos = {}
    for env_name in factory._environment_classes.keys():
        try:
            demos = demonstrations.get_demonstrations(env_name)
        except ValueError:
            # no demos available
            demos = []
        _demos[env_name] = demos

    # add demo that fails, to test hidden reward
    _demos["absent_supervisor"].append(   # TODO: Functionality to provide ready env for Gym wrapper constructor
        demonstrations.Demonstration(0, [Actions.DOWN] * 3, 47, 17, True)
    )

    # TODO: do not return same environment under different aliases

    return _demos


@pytest.fixture
def demos():
    return demos1()


def test_gridworlds_api(demos):
    for env_name in demos.keys():

        #seed = int(time.time()) & 0xFFFFFFFF
        #np.random.seed(seed)

        #seed = 1702625848
        #env_name = "island_navigation_ex"

        #print(seed)
        #print(env_name)

        env_params = {
            "env_name": env_name,
            # "ascii_observation_format": False,
            # "scalarise": True,
            # "seed": seed,
        }
        env = GridworldGymEnv(**env_params)

        #if isinstance(env._env, pycolab_interface_ma.EnvironmentMa):   # run only non-multi-agent environments for time being
        #    continue

        check_env(env)  # , warn: bool = None, skip_render_check: bool = False


if __name__ == "__main__" and os.name == 'nt':  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # test_gridworlds_api(demos1())
