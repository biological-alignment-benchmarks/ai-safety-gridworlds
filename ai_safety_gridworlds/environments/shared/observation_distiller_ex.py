# Copyright 2023-2024 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
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
"""Pycolab rendering wrapper for enabling video recording.

This module contains wrappers that allow for simultaneous transformation of
environment observations into agent view (a numpy 2-D array) and human RGB view
(a numpy 3-D array).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from ai_safety_gridworlds.environments.shared import observation_distiller
from ai_safety_gridworlds.environments.shared.rl.pycolab_interface_mo import INFO_LAYERS

import numpy as np

import six

from pycolab import rendering


AGENT_SPRITE = 'agent_sprite'   # TODO: use safety_game_moma.AGENT_SPRITE instead
Z_ORDER = 'z_order'   # TODO: use safety_game_moma.Z_ORDER instead


class ObservationToArrayWithRGBEx(observation_distiller.ObservationToArrayWithRGB):
  """Convert an `Observation` to a 2-D `board` and 3-D `RGB` numpy array.

  This class is a general utility for converting `Observation`s into 2-D
  `board` representation and 3-D `RGB` numpy arrays. They are returned as a
  dictionary containing the aforementioned keys.

  Additonally, `layers` key is returned exactly with same value as in the 
  `Observation.layers`, `ascii_codes` key, containing the `Observation.board` 
  value, and `ascii` key, containing the char array derived from 
  `Observation.board` value.
  """

  def __init__(self, value_mapping, colour_mapping, 
               environment_data=None,      # ADDED  
               observe_gaps_only_where_other_layers_are_blank=False,     # ADDED
               observable_attribute_categories=[],      # ADDED   
               observable_attribute_value_mapping:dict[str, dict[str, float]]={},      # ADDED  
              ):
    """Construct an `ObservationToArrayWithRGBEx`.

    Builds a callable that will take `Observation`s and emit a dictionary
    containing a 2-D and 3-D numpy array. The rows and columns of the 2-D array
    contain the values obtained after mapping the characters of the original
    `Observation` through `value_mapping`. The rows and columns of the 3-D array
    contain RGB values of the previous 2-D mapping in the [0,1] range.

    Args:
      value_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a scalar or 1-D vector value. All values
          in this dict must be the same type and dimension. Note that strings
          are considered 1-D vectors, not scalar values.
      colour_mapping: a dict mapping any characters that might appear in the
          original `Observation`s to a 3-tuple of RGB values in the range
          [0,999].

    """
    super(ObservationToArrayWithRGBEx, self).__init__(value_mapping, colour_mapping)

    self._environment_data = environment_data    # ADDED
    self._observable_attribute_categories = observable_attribute_categories    # ADDED
    self._observable_attribute_value_mapping = observable_attribute_value_mapping    # ADDED
    self._observe_gaps_only_where_other_layers_are_blank = observe_gaps_only_where_other_layers_are_blank   # ADDED

    self._board_to_ascii_vectorize = np.vectorize(chr)
    
    self._renderers.update({
        INFO_LAYERS: lambda observation: observation.layers, # "Rendering" function for the `layers` value.
        'ascii_codes': lambda observation: observation.board,   # "Rendering" function for the `ascii_ord` value.
        'ascii': lambda observation: self._board_to_ascii_vectorize(observation.board),   # Rendering function for the `ascii` value. converting ordinals to chars.
    })

    if self._environment_data is not None:
      self._renderers['agent_attribute_board_ascii_codes'] = self.compute_agent_attribute_board
      self._renderers['agent_attribute_layers'] = self.compute_agent_attribute_layers


  def set_observable_attribute_categories(self, observable_attribute_categories=[], observable_attribute_value_mapping:dict[str, dict[str, float]]={}):   # ADDED   
    self._observable_attribute_categories = observable_attribute_categories
    self._observable_attribute_value_mapping = observable_attribute_value_mapping


  def compute_agent_attribute_board(self, observation):

    layers = {}
    for attribute in self._observable_attribute_categories:
      layer = np.zeros_like(observation.board)
      layers[attribute] = layer

      # for character, entity in six.iteritems(self._env._sprites_and_drapes):  # process agents in the reverse z-order just as in _render() method in engine.py
      for character in self._environment_data[Z_ORDER]:
        entity = self._environment_data[AGENT_SPRITE].get(character)
        if entity is not None:  # does that character represent an agent?
          if entity.visible:    # TODO: does this check need to be applied elsewhere too?
            
            value = entity.observable_attributes.get(attribute)
            if value is not None:
              layer[entity.position] = value

    return layers


  def compute_agent_attribute_layers(self, observation):

    layers = {}
    for attribute in self._observable_attribute_categories:
      layers[attribute] = {}

      for character, entity in self._environment_data[AGENT_SPRITE].items():
      # for character, entity in six.iteritems(self._env._sprites_and_drapes):  # process agents in the reverse z-order just as in _render() method in engine.py
        # if isinstance(entity, safety_game_ma.SafetySprite) and entity.visible:
        if entity.visible:  

          value = entity.observable_attributes.get(attribute)
          if value is not None:

            # need separate layer for each agent and attribute since we need to run them through agent perspectives transformation and then later extract the perspective-filtered attribute values and coordinates per attribute and owner agent combination again
            layer = np.zeros_like(next(iter(observation.layers)))    # TODO: create a special sparse class for processing these layers in agent perspective transformation
            layers[attribute][entity.character] = layer

            layer[entity.position] = value

    return layers
        

  def __call__(self, observation):
    """Derives `board` and `RGB` arrays from an `Observation`.

    Returns a dict with 2-D `board` and 3-D `RGB` numpy arrays as described in
    the constructor.

    Args:
      observation: an `Observation` from which this method derives numpy arrays.

    Returns:
      a dict containing 'board' and 'RGB' keys as described.

    """
    # Perform observation rendering for agent and for video recording.
    result = {}
    for key, renderer in self._renderers.items():
      result[key] = renderer(observation)

    if self._observe_gaps_only_where_other_layers_are_blank:

      layers = dict(result[INFO_LAYERS])     # shallow clone before the gap_chr entry of layers is overwritten below
      # observation.layers = layers   # comment-out: cannot replace attribute in tuple. So you just need to be careful to not user observation.layers anymore anywhere
      result[INFO_LAYERS] = layers

      gap_chr = self._environment_data.get("what_lies_beneath", ' ')
      gaps_layer = layers[gap_chr].copy()   # NB! make copy in order to not change the gaps layer in pycolab
      layers[gap_chr] = gaps_layer

      for chr, layer in layers.items(): 
        if chr == gap_chr:
          continue
        gaps_layer &= np.logical_not(layer)
      

    if self._environment_data is not None:
      result['agent_attribute_board_ascii'] = {    # ADDED
        key: self._board_to_ascii_vectorize(value) 
        for key, value in result['agent_attribute_board_ascii_codes'].items() 
      } 

    # Convert to [0, 255] RGB values.
    result['RGB'] = (result['RGB'] / 999.0 * 255.0).astype(np.uint8)
    return result

