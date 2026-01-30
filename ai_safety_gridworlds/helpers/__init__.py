# Copyright 2026 Roland Pihlakas. https://github.com/biological-alignment-benchmarks/ai-safety-gridworlds
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


def get_project_root():
    project_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    return project_root

def read_code_version():
    project_root = get_project_root()

    pyproject_toml = os.path.join(project_root, "pyproject.toml")
    if os.path.exists(pyproject_toml):
        import toml

        toml_data = toml.load(pyproject_toml)
        toml_version = str(toml_data.get("project", {}).get("version", "0.0.0.0"))
    else:
        toml_version = "0.0.0.0"

    citation_cff = os.path.join(project_root, "CITATION.cff")
    if os.path.exists(citation_cff):
        import yaml

        with open(citation_cff, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh)
        cff_version = str(yaml_data.get("version", "0.0.0.0"))
    else:
        cff_version = "0.0.0.0"

    from packaging.version import parse as parse_version

    toml_version = parse_version(toml_version)
    cff_version = parse_version(cff_version)
    result = str(
        max(toml_version, cff_version)
    )  # take latest version in case one of the files is forgotten to be updated

    return result


# / def read_code_version():

code_version = (
    read_code_version()
)  # read code version at the start of the program so that if the files are updated while program is running then that will not mess up the version info


def get_gridworlds_code_version():
    return code_version
