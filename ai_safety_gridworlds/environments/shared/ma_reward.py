# Copyright 2022-2023 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
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
"""Helpers for creating safety environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ast import literal_eval
import itertools

from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward

# Dependency imports
import numpy as np


class ma_reward(object):

  def __init__(self, reward_agents_dict, immutable=True):

    self._agent_rewards_dict = reward_agents_dict
    self._immutable = immutable


  def copy(self):

    dict_clone = dict(self._agent_rewards_dict) # clone
    return ma_reward(dict_clone, immutable=False)


  def __eq__(self, other):

    if np.isscalar(other):
      return all(value == other for value in self._agent_rewards_dict.values())

    return self._agent_rewards_dict == other._agent_rewards_dict


  def iszero(self):  

    return all(value == 0 for value in self._agent_rewards_dict.values())


  def max(self, other):

    result_dict = dict(self._agent_rewards_dict)  # clone

    if np.isscalar(other):
      return ma_reward({ key: max(value, other) for key, value in self._agent_rewards_dict.items() }, immutable=False)

    elif isinstance(other, ma_reward):
      result_dict = { key: value.max(0) for key, value in result_dict.items() }
      for other_key, other_value in other._agent_rewards_dict.items():
        result_dict[other_key] = other_value.max(result_dict.get(other_key, 0))

      return ma_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.max, expecting a scalar or ma_reward")


  def min(self, other):

    result_dict = dict(self._agent_rewards_dict)  # clone

    if np.isscalar(other):
      return ma_reward({ key: min(value, other) for key, value in self._agent_rewards_dict.items() }, immutable=False)

    elif isinstance(other, ma_reward):
      result_dict = { key: value.min(0) for key, value in result_dict.items() }
      for other_key, other_value in other._agent_rewards_dict.items():
        result_dict[other_key] = other_value.min(result_dict.get(other_key, 0))

      return ma_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.min, expecting a scalar or ma_reward")


  @staticmethod
  def max(rewards_list):

    result = ma_reward({})
    for reward in rewards_list:
      result = result.max(reward)
    return result


  @staticmethod
  def min(rewards_list):

    result = ma_reward({})
    for reward in rewards_list:
      result = result.min(reward)
    return result


  @staticmethod
  def parse(string):

    if string == "":
      return ma_reward({})
    else:
      object = literal_eval(string)
      # object = json.loads(string.replace("'", '"')) # ma_reward input python dictionary string is similar to json
      return ma_reward(object)


  @staticmethod
  def get_enabled_agent_rewards_keys(enabled_ma_rewards):
    """Returns keys of all agents defined in enabled_ma_rewards.

    Args:
      enabled_ma_rewards: a list of ma_reward objects.
    """

    if enabled_ma_rewards is None:

      return [None]

    else: # if enabled_ma_rewards is not None:

      # each reward may contain more than one enabled agent
      enabled_agents_rewards_keys = {}
      for agent_key, enabled_mo_rewards in enabled_ma_rewards.items():

        #keys_per_reward = [{ 
        #                      key for key, unit_value 
        #                      in reward._reward_dimensions_dict.items() if unit_value != 0
        #                    }
        #                    for reward in enabled_mo_rewards]

        ## select distinct keys:
        #enabled_agent_rewards_keys = list(set.union(*keys_per_reward))  # this does not preserve the order of the keys
        ## enabled_agent_rewards_keys = list(dict.fromkeys(itertools.chain.from_iterable(keys_per_reward)).keys())  # this preserves the order of the keys
        #enabled_agent_rewards_keys.sort()  # for some reason the sorting order was still changing

        enabled_agent_rewards_keys = mo_reward.get_enabled_reward_dimension_keys(enabled_mo_rewards)
        enabled_agents_rewards_keys[agent_key] = enabled_agent_rewards_keys

      #/ for agent_key, enabled_mo_rewards in enabled_ma_rewards.items():

      return enabled_agents_rewards_keys


  @staticmethod
  def get_enabled_reward_unit_space(enabled_ma_rewards):  
    """Returns dictionary of per-agent-tuples of min unit reward vector and max unit reward vector for each reward.

    Args:
      enabled_ma_rewards: a list of ma_reward objects.
    """

    if enabled_ma_rewards is None:

      return None

    else: # if enabled_ma_rewards is not None:

      # enabled_agent_rewards_keys = ma_reward.get_enabled_agent_rewards_keys(enabled_ma_rewards)

      #min_unit_value_per_key = {
      #                            key: mo_reward.min([
      #                              reward._agent_rewards_dict.get(key, 0)
      #                              for reward in enabled_ma_rewards
      #                            ])
      #                            for key in enabled_agent_rewards_keys
      #                         }
 
      #max_unit_value_per_key = {
      #                            key: mo_reward.max([
      #                              reward._agent_rewards_dict.get(key, 0)
      #                              for reward in enabled_ma_rewards
      #                            ])
      #                            for key in enabled_agent_rewards_keys
      #                         }

      #return [list(min_unit_value_per_key.values()), list(max_unit_value_per_key.values())]

      enabled_agent_rewards_unit_spaces = {}
      for agent_key, enabled_mo_rewards in enabled_ma_rewards.items():

        # result[agent_key] = self._agent_rewards_dict.get(enabled_agent_rewards_key, mo_reward({})).get_enabled_reward_unit_space(enabled_ma_rewards[enabled_agent_rewards_key])

        enabled_agent_rewards_unit_space = mo_reward.get_enabled_reward_unit_space(enabled_mo_rewards)
        enabled_agent_rewards_unit_spaces[agent_key] = enabled_agent_rewards_unit_space

      return enabled_agent_rewards_unit_spaces


  def tolist(self, enabled_ma_rewards):
    """Converts the ma_reward value to a dictionary of per-agent-lists of all reward values including rewards with zero values."""

    if enabled_ma_rewards is None:  # scalarise

      reward_items = self._agent_rewards_dict.items()
      return { key: reward_value.tolist(None) for key, reward_value in reward_items }

    else: # if enabled_ma_rewards is not None:

      enabled_agent_rewards_keys = ma_reward.get_enabled_agent_rewards_keys(enabled_ma_rewards)

      for key, value in self._agent_rewards_dict.items():
        if key not in enabled_agent_rewards_keys and value != 0:
          raise ValueError("Agent %s is not enabled but is still included in ma_reward with nonzero value" % key)

      result = {}
      for enabled_agent_rewards_key in enabled_agent_rewards_keys:
        result[enabled_agent_rewards_key] = self._agent_rewards_dict.get(enabled_agent_rewards_key, mo_reward({})).tolist(enabled_ma_rewards[enabled_agent_rewards_key])
      return result


  def tofull(self, enabled_ma_rewards):
    """Converts the ma_reward value to dictionary of per-agent-dictionaries containing keys of all rewards including rewards with zero values."""

    if enabled_ma_rewards is None:  # scalarise

      reward_items = self._agent_rewards_dict.items()
      return { key: reward_value.tofull(None) for key, reward_value in reward_items }

    else: # if enabled_ma_rewards is not None:

      enabled_agent_rewards_keys = ma_reward.get_enabled_agent_rewards_keys(enabled_ma_rewards)

      for key, value in self._agent_rewards_dict.items():
        if key not in enabled_agent_rewards_keys and value != 0:
          raise ValueError("Agent %s is not enabled but is still included in ma_reward with nonzero value" % key)

      result = {}
      for enabled_agent_rewards_key in enabled_agent_rewards_keys:
        result[enabled_agent_rewards_key] = self._agent_rewards_dict.get(enabled_agent_rewards_key, mo_reward({})).tofull(enabled_ma_rewards[enabled_agent_rewards_key])
      return result


  def __str__(self, enabled_ma_rewards=None): # tostring

    if enabled_ma_rewards is not None:
      enabled_agent_rewards_keys = ma_reward.get_enabled_agent_rewards_keys(enabled_ma_rewards)
      dict_with_enabled_keys = { key: self._agent_rewards_dict.get(key, mo_reward({})) for key in enabled_agent_rewards_keys }
      return str(dict_with_enabled_keys)
    else:
      return str({ key: value for key, value in self._agent_rewards_dict.items() if value != 0 })


  def __repr__(self, enabled_ma_rewards=None): # tostring

    if enabled_ma_rewards is not None:
      enabled_agent_rewards_keys = ma_reward.get_enabled_agent_rewards_keys(enabled_ma_rewards)
      dict_with_enabled_keys = { key: self._agent_rewards_dict.get(key, mo_reward({})) for key in enabled_agent_rewards_keys }
      return "<" + repr(dict_with_enabled_keys) + ">"
    else:
      return "<" + repr({ key: value for key, value in self._agent_rewards_dict.items() if value != 0 }) + ">"


  def __add__(self, other):

    result_dict = dict(self._agent_rewards_dict)  # clone

    if np.isscalar(other):
      return ma_reward({ key: value + other for key, value in self._agent_rewards_dict.items() }, immutable=False)

    elif isinstance(other, ma_reward):
      for other_key, other_value in other._agent_rewards_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) + other_value
      return ma_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.__add__, expecting a scalar or ma_reward")


  def __iadd__(self, other):  # in-place add

    if self._immutable:
      return self.__add__(other)

    if np.isscalar(other):
      for key, value in self._agent_rewards_dict.items():
        self._agent_rewards_dict[key] = value + other

    elif isinstance(other, ma_reward):
      for other_key, other_value in other._agent_rewards_dict.items():
        self._agent_rewards_dict[other_key] = self._agent_rewards_dict.get(other_key, 0) + other_value

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.__iadd__, expecting a scalar or ma_reward")

    return self


  def __radd__(self, other):  # reflected add (the order of operands is exchanged)

    return self + other


  def __sub__(self, other):

    result_dict = dict(self._agent_rewards_dict)  # clone

    if np.isscalar(other):
      return ma_reward({ key: value - other for key, value in self._agent_rewards_dict.items() }, immutable=False)

    elif isinstance(other, ma_reward):
      for other_key, other_value in other._agent_rewards_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) - other_value
      return ma_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.__sub__, expecting a scalar or ma_reward")


  def __isub__(self, other):  # in-place sub

    if self._immutable:
      return self.__sub__(other)

    if np.isscalar(other):
      for key, value in self._agent_rewards_dict.items():
        self._agent_rewards_dict[key] = value - other

    elif isinstance(other, ma_reward):
      for other_key, other_value in other._agent_rewards_dict.items():
        self._agent_rewards_dict[other_key] = self._agent_rewards_dict.get(other_key, 0) - other_value

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.__isub__, expecting a scalar or ma_reward")

    return self


  def __rsub__(self, other):  # reflected sub (the order of operands is exchanged)

    result_dict = dict(self._agent_rewards_dict)  # clone

    if np.isscalar(other):
      return ma_reward({ key: other - value for key, value in self._agent_rewards_dict.items() }, immutable=False)

    elif isinstance(other, ma_reward):
      for other_key, other_value in other._agent_rewards_dict.items():
        result_dict[other_key] = other_value - result_dict.get(other_key, 0)
      return ma_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for ma_reward.__rsub__, expecting a scalar or ma_reward")


  def __neg__(self):  # unary -

    return ma_reward({ key: -value for key, value in self._agent_rewards_dict.items() }, immutable=False)


  def __mul__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for ma_reward.__mul__, expecting a scalar")

    return ma_reward({ key: value * other for key, value in self._agent_rewards_dict.items() }, immutable=False)


  def __imul__(self, other):  # in-place mul

    if self._immutable:
      return self.__mul__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for ma_reward.__imul__, expecting a scalar")

    for key, value in self._agent_rewards_dict.items():
      self._agent_rewards_dict[key] = value * other

    return self


  def __rmul__(self, other):  # reflected mul (the order of operands is exchanged)

    return self * other


  def __truediv__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for ma_reward.__truediv__, expecting a scalar")

    return ma_reward({ key: value / other for key, value in self._agent_rewards_dict.items() }, immutable=False)


  def __itruediv__(self, other):  # in-place div

    if self._immutable:
      return self.__truediv__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for ma_reward.__itruediv__, expecting a scalar")

    for key, value in self._agent_rewards_dict.items():
      self._agent_rewards_dict[key] = value / other

    return self


  def __rtruediv__(self, other):  # reflected div (the order of operands is exchanged)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for ma_reward.__rtruediv__, expecting a scalar")

    return ma_reward({ key: other / value for key, value in self._agent_rewards_dict.items() }, immutable=False)


qqq = True  # for debugging






