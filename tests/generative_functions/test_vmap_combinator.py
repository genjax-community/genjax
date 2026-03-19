# Copyright 2023 MIT Probabilistic Computing Project
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

import re

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import IndexRequest, Regenerate, Selection, StaticRequest, Update
from genjax import Selection as S
from genjax._src.core.typing import IntArray


class TestVmap:
    def test_vmap_combinator_simple_normal(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_vmap_simple_normal_project(self):
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        vmapped = model.vmap(in_axes=(0,))

        key = jax.random.key(314159)
        means = jnp.arange(0, 10, dtype=float)

        tr = jax.jit(vmapped.simulate)(key, (means,))

        vmapped_score = tr.get_score()

        assert tr.project(key, Selection.all()) == vmapped_score
        assert tr.project(key, Selection.none()) == 0.0

    def test_vmap_combinator_vector_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(
            jnp.arange(3), jnp.array([3.0, 2.0, 3.0])
        )

        (_, w) = jax.jit(kernel.importance)(key, chm, (map_over,))
        assert (
            w
            == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]
            + genjax.normal.assess(C.v(2.0), (1.0, 1.0))[0]
            + genjax.normal.assess(C.v(3.0), (2.0, 1.0))[0]
        )

    def test_vmap_combinator_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = C[0, "z"].set(3.0)
        key, sub_key = jax.random.split(key)
        (_, w) = jax.jit(kernel.importance)(sub_key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]

        key, sub_key = jax.random.split(key)
        zv = jnp.array([3.0, -1.0, 2.0])
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(jnp.arange(3), zv)
        (tr, _) = kernel.importance(sub_key, chm, (map_over,))
        for i in range(0, 3):
            v = tr.get_choices()[i, "z"]
            assert v == zv[i]

    def test_vmap_combinator_nested_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def higher_model(x):
            return model(x) @ "outer"

        key = jax.random.key(314159)
        map_over = jnp.ones((3, 3), dtype=float)
        chm = C[0, "outer", 1, "z"].set(1.0)
        (_, w) = jax.jit(higher_model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]

    def test_vmap_combinator_vmap_pytree(self):
        @genjax.gen
        def model2(x):
            _ = genjax.normal(x, 1.0) @ "y"
            return x

        model_mv2 = model2.mask().vmap()
        masks = jnp.array([
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ])
        xs = jnp.arange(0.0, 10.0, 1.0)

        key = jax.random.key(314159)

        # show that we don't error if we map along multiple axes via the default.
        tr = jax.jit(model_mv2.simulate)(key, (masks, xs))
        assert jnp.array_equal(tr.get_retval().value, xs)
        assert jnp.array_equal(tr.get_retval().flag, masks)

        @genjax.vmap(in_axes=(None, (0, None)))
        @genjax.gen
        def foo(y, args):
            loc, (scale, _) = args
            x = genjax.normal(loc, scale) @ "x"
            return x + y

        _ = jax.jit(foo.simulate)(key, (10.0, (jnp.arange(3.0), (1.0, jnp.arange(3)))))

    def test_vmap_combinator_assess(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        sample = tr.get_choices()
        map_score = tr.get_score()
        assert model.assess(sample, (map_over,))[0] == map_score

    def test_vmap_validation(self):
        @genjax.gen
        def foo(loc: float, scale: float):
            return genjax.normal(loc, scale) @ "x"

        key = jax.random.key(314159)

        with pytest.raises(
            ValueError,
            match="vmap was requested to map its argument along axis 0, which implies that its rank should be at least 1, but is only 0",
        ):
            jax.jit(foo.vmap(in_axes=(0, None)).simulate)(key, (10.0, jnp.arange(3.0)))

        # in_axes doesn't match args
        with pytest.raises(
            ValueError,
            match="vmap in_axes specification must be a tree prefix of the corresponding value",
        ):
            jax.jit(foo.vmap(in_axes=(0, (0, None))).simulate)(
                key, (10.0, jnp.arange(3.0))
            )

        with pytest.raises(
            (IndexError, ValueError),
        ):
            jax.jit(foo.vmap(in_axes=0).simulate)(key, (jnp.arange(2), jnp.arange(3)))

        # in_axes doesn't match args
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Found incompatible dtypes, <class 'numpy.float32'> and <class 'numpy.int32'>"
            ),
        ):
            jax.jit(foo.vmap(in_axes=(None, 0)).simulate)(key, (10.0, jnp.arange(3)))

    def test_vmap_key_vmap(self):
        @genjax.gen
        def model(x):
            y = genjax.normal(x, 1.0) @ "y"
            return y

        vmapped = model.vmap(in_axes=(0,))

        key = jax.random.key(314159)
        keys = jax.random.split(key, 10)
        xs = jnp.arange(5, dtype=float)

        results = jax.vmap(lambda k: vmapped.simulate(k, (xs,)))(jnp.array(keys))

        chm = results.get_choices()

        # the inner vmap aggregates a score, while the outer vmap does not accumulate anything
        assert results.get_score().shape == (10,)

        # the inner vmap has vmap'd over the y's
        assert chm[:, "y"].shape == (10, 5)

    def test_zero_length_vmap(self):
        @genjax.gen
        def step(state, sigma):
            new_x = genjax.normal(state, sigma) @ "x"
            return (new_x, new_x + 1)

        trace = step.vmap(in_axes=(None, 0)).simulate(
            jax.random.key(20), (2.0, jnp.arange(0, dtype=float))
        )

        assert trace.get_choices().static_is_empty(), (
            "zero-length vmap produces empty choicemaps."
        )


@genjax.Pytree.dataclass
class MyClass(genjax.PythonicPytree):
    x: IntArray


class TestVmapPytree:
    def test_vmap_pytree(self):
        batched_val = jax.vmap(lambda x: MyClass(x))(jnp.arange(5))

        def regular_function(mc: MyClass):
            return mc.x + 5

        assert jnp.array_equal(
            jax.vmap(regular_function)(batched_val), jnp.arange(5) + 5
        )

        @genjax.gen
        def generative_function(mc: MyClass):
            return mc.x + 5

        key = jax.random.key(0)

        # check that we can vmap over a vectorized pytree.
        assert jnp.array_equal(
            generative_function.vmap(in_axes=0)(batched_val)(key), jnp.arange(5) + 5
        )


class TestVmapIndexRequest:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    def test_vmap_regenerate(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal.vmap()(jnp.zeros(1000), jnp.ones(1000)) @ "a"
            return x

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        for idx in range(10):
            old_a = tr.get_choices()["a", idx]
            old_target_density = genjax.normal.logpdf(old_a, 0.0, 1.0)

            request = StaticRequest({
                "a": IndexRequest(jnp.array(idx), Regenerate(S.all()))
            })

            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_a = new_tr.get_choices()["a", idx]
            new_target_density = genjax.normal.logpdf(new_a, 0.0, 1.0)

            assert fwd_w == new_target_density - old_target_density

    def test_vmap_update(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal.vmap()(jnp.zeros(1000), jnp.ones(1000)) @ "a"
            return x

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        for idx in range(10):
            old_a = tr.get_choices()["a", idx]
            old_target_density = genjax.normal.logpdf(old_a, 0.0, 1.0)

            request = StaticRequest({
                "a": IndexRequest(jnp.array(idx), Update(C.v(idx + 7.0)))
            })

            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_a = new_tr.get_choices()["a", idx]
            new_target_density = genjax.normal.logpdf(new_a, 0.0, 1.0)

            assert new_a == idx + 7.0
            assert fwd_w == new_target_density - old_target_density
