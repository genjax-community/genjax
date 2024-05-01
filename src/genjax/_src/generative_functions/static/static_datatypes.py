# Copyright 2024 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import jax  # noqa: I001
import jax.numpy as jnp  # noqa: I001
import rich.console
import rich.tree


from genjax._src.core.datatypes.generative import (
    GenerativeFunction,
    HierarchicalChoiceMap,
    Selection,
    Trace,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.serialization.pickle import (
    PickleDataFormat,
    PickleSerializationBackend,
    SupportsPickleSerialization,
)
from genjax._src.core.typing import Any, FloatArray, Tuple, dispatch, PRNGKey

#########
# Trace #
#########


class StaticTrace(
    Trace,
    SupportsPickleSerialization,
):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    address_choices: Trie
    cache: Trie
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return HierarchicalChoiceMap(self.address_choices).strip()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def get_subtrace(self, addr):
        return self.address_choices[addr]

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        weight = jnp.array(0.0)
        for k, subtrace in self.address_choices.get_submaps_shallow():
            key, sub_key = jax.random.split(key)
            remaining = selection.step(k)
            weight += subtrace.project(sub_key, remaining)
        return weight

    def has_cached_value(self, addr):
        return self.cache.has_submap(addr)

    def get_cached_value(self, addr):
        return self.cache.get_submap(addr)

    def get_aux(self):
        return (
            self.address_choices,
            self.cache,
        )

    #################
    # Serialization #
    #################

    @dispatch
    def dumps(
        self,
        backend: PickleSerializationBackend,
    ) -> PickleDataFormat:
        args, retval, score = self.args, self.retval, self.score
        choices_payload = []
        addr_payload = []
        for addr, subtrace in self.address_choices.get_submaps_shallow():
            inner_payload = subtrace.dumps(backend)
            choices_payload.append(inner_payload)
            addr_payload.append(addr)
        payload = [
            backend.dumps(args),
            backend.dumps(retval),
            backend.dumps(score),
            backend.dumps(addr_payload),
            backend.dumps(choices_payload),
        ]
        return PickleDataFormat(payload)

    def __rich__(self):
        tree = rich.tree.Tree(self.__class__.__name__)
        tree.add("gen_fn").add(rich.console.Pretty(self.gen_fn))
        tree.add("args").add(rich.console.Pretty(self.args))
        tree.add("retval").add(rich.console.Pretty(self.retval))
        tree.add("score").add(rich.console.Pretty(self.score))
        self.address_choices._append_to_rich_tree(tree.add("choices"))
        return tree
