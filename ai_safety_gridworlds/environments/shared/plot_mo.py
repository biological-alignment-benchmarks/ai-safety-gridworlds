# Copyright 2022 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
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

from pycolab import plot


class PlotMo(plot.Plot):

  def add_reward(self, reward):
    """Add a value to the reward the `Engine` will return to the player(s).

    During a game iteration, any `Backdrop`, `Sprite`, or `Drape` can call this
    method to add value to the reward that the `Engine` will return to the
    player (or players) for having taken the action (or actions) supplied in the
    `actions` argument to `Engine`'s `play` method.

    This value need not be a number, but can be any kind of value appropriate to
    the game.  If there's ever any chance that more than one `Sprite`, `Drape`,
    or `Backdrop` would supply a reward during a game iteration, the value
    should probably be of a type that supports the `+=` operator in a relevant
    way, since this method uses addition to accumulate reward. (For custom
    classes, this typically means implementing the `__iadd__` method.)

    If this method is never called during a game iteration, the `Engine` will
    supply None to the player (or players) as the reward.

    Args:
      reward: reward value to accumulate into the current game iteration's
          reward for the player(s). See discussion for details.
    """
    if self._engine_directives.summed_reward is None:
      self._engine_directives.summed_reward = reward.copy()    # incoming mo_reward argument has to be treated as immutable else rewards across timesteps will be accumulated in per timestep accumulator
    else:
      self._engine_directives.summed_reward += reward
