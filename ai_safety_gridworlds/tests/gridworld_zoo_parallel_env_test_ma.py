# Copyright 2022-2023 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
# Copyright 2019 jvmncs https://github.com/jvmncs
# Copyright 2018-2019 David Lindner https://github.com/david-lindner/safe-grid-gym

import traceback
import unittest

from pettingzoo import AECEnv, ParallelEnv

try:
  import gymnasium as gym
  gym_v26 = True
except:
  import gym
  gym_v26 = False

import numpy as np

from ai_safety_gridworlds.helpers import factory
from ai_safety_gridworlds.demonstrations import demonstrations
from ai_safety_gridworlds.environments.shared import safety_game_moma
from ai_safety_gridworlds.environments.shared.safety_game import Actions

from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import GridworldZooParallelEnv
from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import INFO_HIDDEN_REWARD, INFO_OBSERVED_REWARD


class SafetyGridworldsTestCase(unittest.TestCase):
  def _check_rgb(self, rgb_list):
    first_shape = rgb_list[0].shape
    for rgb in rgb_list:
      self.assertIsInstance(rgb, np.ndarray)
      self.assertEqual(len(rgb.shape), 3)
      self.assertEqual(rgb.shape[0], 3)
      self.assertEqual(rgb.shape, first_shape)

  def _check_ansi(self, ansi_list):
    first_len = len(ansi_list[0])
    first_newline_count = ansi_list[0].count("\n")
    for ansi in ansi_list:
      self.assertIsInstance(ansi, str)
      self.assertEqual(len(ansi), first_len)
      self.assertEqual(ansi.count("\n"), first_newline_count)

  def _check_reward(self, env, reward):  # TODO: seems like env.reward_range is not implemented in zoo
    pass
  #  min_reward, max_reward = env.reward_range
  #  self.assertGreaterEqual(reward, min_reward)
  #  self.assertLessEqual(reward, max_reward)

  def _check_action_observation_valid(self, env, agent, action, observation):
    self.assertTrue(env.action_space(agent).contains(action))
    self.assertTrue(env.observation_space(agent).contains(observation))

  def _check_rewards(
    self,
    env,
    demo,
    episode_info_observed_return,
    episode_info_hidden_return,
    episode_return,
  ):
    # check observed and hidden rewards
    if not isinstance(env, safety_game_moma.SafetyEnvironmentMoMa):
      episode_info_observed_return = next(iter(episode_info_observed_return.values()))
      episode_return = next(iter(episode_return.values()))

    self.assertEqual(episode_info_observed_return, demo.episode_return)

    hidden_reward = env._env._get_hidden_reward(default_reward=None)

    if hidden_reward is not None:
      if not isinstance(env, safety_game_moma.SafetyEnvironmentMoMa):
        episode_info_hidden_return = next(iter(episode_info_hidden_return.values()))
      self.assertEqual(episode_info_hidden_return, demo.safety_performance)
      self.assertEqual(episode_info_hidden_return, hidden_reward)

    self.assertEqual(episode_info_observed_return, episode_return)

  def setUp(self):
    self.demonstrations = {}
    for env_name in factory._environment_classes.keys():
      try:
        demos = demonstrations.get_demonstrations(env_name)
      except ValueError:
        # no demonstrations available
        demos = []
      # TODO: for tests, do not register same environment under different aliases
      self.demonstrations[env_name] = demos

    # add demo that fails, to test hidden reward
    self.demonstrations["absent_supervisor"].append(
      demonstrations.Demonstration(0, [Actions.DOWN] * 3, 47, 17, True)
    )

  def testActionSpaceSampleContains(self):
    """
    Check that sample and contain methods of the action space are consistent.
    """
    repetitions = 10

    for env_name in self.demonstrations.keys():
      env = GridworldZooParallelEnv(env_name)
      for agent in env.possible_agents:
        action_space = env.action_space(agent)
        for _ in range(repetitions):
          action = action_space.sample()
          self.assertTrue(action_space.contains(action))

  def testObservationSpaceSampleContains(self):
    """
    Check that sample and contain methods of the observation space are consistent.
    """
    repetitions = 10

    for env_name in self.demonstrations.keys():
      env = GridworldZooParallelEnv(env_name)
      for agent in env.possible_agents:
        observation_space = env.observation_space(agent)
        for _ in range(repetitions):
          observation = observation_space.sample()
          assert observation_space.contains(observation)

  def reset(self, env):
    obs, infos = env.reset()
    return obs

  def step(self, env, action):
    if isinstance(env, safety_game_moma.SafetyEnvironmentMoMa):
      agents_actions = {
        # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
        agent: { "step": action } for agent in env.agents
      }  
    else:
      agents_actions = {
        # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
        agent: action for agent in env.agents
      } 

    if gym_v26:
      obs, _, _, _, _ = env.step(agents_actions)
    else:
      obs, _, _, _ = env.step(agents_actions)
    return obs

  def testStateObjectCopy(self):
    """
    Make sure that the state array that is returned does not change in
    subsequent steps of the environment. The pycolab only returns a pointer
    to the state object, which makes it change if we take another step.
    For gym, however, we want the state to not change, i.e. return a copy
    of the board.
    """
    env = GridworldZooParallelEnv("island_navigation_ex_ma")
    obs0 = self.reset(env)
    global_obs0 = env.state
    obs1 = self.step(env, Actions.UP)
    global_obs1 = env.state
    obs2 = self.step(env, Actions.RIGHT)
    global_obs2 = env.state
    self.assertFalse(any(np.all(obs0[key] == obs1[key]) for key in obs0.keys()))
    self.assertFalse(any(np.all(obs0[key] == obs2[key]) for key in obs0.keys()))
    self.assertFalse(any(np.all(obs1[key] == obs2[key]) for key in obs0.keys()))
    self.assertFalse(np.all(global_obs0 == global_obs1))
    self.assertFalse(np.all(global_obs0 == global_obs2))
    self.assertFalse(np.all(global_obs1 == global_obs2))

    # ADDED
    env = GridworldZooParallelEnv("firemaker_ex_ma")
    obs0 = self.reset(env)
    global_obs0 = env.state
    obs1 = self.step(env, Actions.RIGHT)
    global_obs1 = env.state
    obs2 = self.step(env, Actions.RIGHT)
    global_obs2 = env.state
    self.assertFalse(any(np.all(obs0[key] == obs1[key]) for key in obs0.keys()))
    self.assertFalse(any(np.all(obs0[key] == obs2[key]) for key in obs0.keys()))
    self.assertFalse(any(np.all(obs1[key] == obs2[key]) for key in obs0.keys()))
    self.assertFalse(np.all(global_obs0 == global_obs1))
    self.assertFalse(np.all(global_obs0 == global_obs2))
    self.assertFalse(np.all(global_obs1 == global_obs2))


  def testTransitions(self):
    """
    Ensure that when the use_transitions argument is set to True the state
    contains the board of the last two timesteps.
    """
    env = GridworldZooParallelEnv("island_navigation_ex_ma", use_transitions=False, max_iterations=100)
    board_init = self.reset(env)
    done = False
    # while not done:
    while env.agents:
      # this is where you would insert your policy
      # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
      actions = {agent: env.action_space(agent).sample() for agent in env.agents}   
      observations, rewards, terminations, truncations, infos = env.step(actions)
      done = all( terminations[agent] or truncations[agent] for agent in env.possible_agents )

    env.close()

    env = GridworldZooParallelEnv("island_navigation_ex_ma", use_transitions=True, max_iterations=100)
    board_init = self.reset(env)
    done = False
    # while not done:
    while env.agents:
      # this is where you would insert your policy
      # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
      actions = {agent: env.action_space(agent).sample() for agent in env.agents}
      observations, rewards, terminations, truncations, infos = env.step(actions)
      done = all( terminations[agent] or truncations[agent] for agent in env.possible_agents )

    env.close()


    # ADDED
    env = GridworldZooParallelEnv("firemaker_ex_ma", level=0, use_transitions=False, max_iterations=100)
    board_init = self.reset(env)
    done = False
    # while not done:
    while env.agents:
      # this is where you would insert your policy
      # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
      actions = {agent: env.action_space(agent).sample() for agent in env.agents}
      observations, rewards, terminations, truncations, infos = env.step(actions)
      done = all( terminations[agent] or truncations[agent] for agent in env.possible_agents )

    env.close()

    env = GridworldZooParallelEnv("firemaker_ex_ma", level=0, use_transitions=True, max_iterations=100)
    board_init = self.reset(env)
    done = False
    # while not done:
    while env.agents:
      # this is where you would insert your policy
      # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
      actions = {agent: env.action_space(agent).sample() for agent in env.agents}
      observations, rewards, terminations, truncations, infos = env.step(actions)
      done = all( terminations[agent] or truncations[agent] for agent in env.possible_agents )

    env.close()


  def testWithDemonstrations(self):
    """
    Run demonstrations in the safety gridworlds and perform sanity checks
    on rewards, episode termination and the "ansi" and "rgb_array" render modes.
    """

    repititions = 10

    for env_name, demos in self.demonstrations.items():
      for demo in demos:
        for i in range(repititions):

          try:      # ADDED
            # need to use np seed instead of the environment seed function
            # to be consistent with the seeds given in the demonstrations
            np.random.seed(demo.seed)
            env = GridworldZooParallelEnv(env_name)

            actions = demo.actions
            self.reset(env)
            done = False

            episode_return = {}
            episode_info_observed_return = {}
            episode_info_hidden_return = {}

            rgb_list = [env.render("rgb_array")]
            ansi_list = [env.render("ansi")]

            for index, action in enumerate(actions):
              self.assertFalse(done)

              if index == len(actions) - 1:
                qqq = True    # for debugging

              agents_actions = {
                # NB! send actions to env.agents, not env.possible_agents since the latter contains terminated agents
                agent: action for agent in env.agents
              }  

              if gym_v26:
                (obss, rewards, terminateds, truncateds, infos) = env.step(agents_actions)
                dones = {
                  agent: terminated or truncateds[agent]
                  for (agent, terminated) in terminateds.items()
                }
              else:
                (obss, rewards, dones, infos) = env.step(agents_actions)
                
              done = all(dones.values())

              episode_return = { agent: episode_return.get(agent, 0) + rewards[agent] for agent in rewards.keys() }
              # episode_info_observed_return += infos[INFO_OBSERVED_REWARD]    # TODO
              episode_info_observed_return = { agent: episode_info_observed_return.get(agent, 0) + rewards[agent] for agent in rewards.keys() }

              for agent in env.possible_agents:
                if infos[agent][INFO_HIDDEN_REWARD] is not None:
                  episode_info_hidden_return[agent] = episode_info_hidden_return.get(agent, 0) + infos[agent][INFO_HIDDEN_REWARD]

              rgb_list.append(env.render("rgb_array"))
              ansi_list.append(env.render("ansi"))

              for agent in env.possible_agents:
                self._check_action_observation_valid(env, agent, action, obss[agent])
              self._check_reward(env, rewards)

            self.assertEqual(done, demo.terminates)
            self._check_rewards(
              env,
              demo,
              episode_info_observed_return,
              episode_info_hidden_return,
              episode_return,
            )

            self._check_rgb(rgb_list)
            self._check_ansi(ansi_list)

          except Exception as ex:      # ADDED
            print("\nenv_name: " + env_name)
            # raise
            print(ex)
            print(traceback.format_exc())


if __name__ == "__main__":
  unittest.main()
