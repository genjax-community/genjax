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


import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Selection
from genjax import SelectionBuilder as S
from genjax._src.generative_functions.gen import DefTrace


class TestTupleAddr:
    def test_tupled_address(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ ("x", "x0")
            y = genjax.normal(x, 1.0) @ "y"
            return y

        tr = f.simulate(())
        chm = tr.get_choices()
        x_score, _ = genjax.normal.assess(C.v(chm["x", "x0"]), (0.0, 1.0))
        assert x_score == tr.project(
            Selection.at["x", "x0"],
        )

    @pytest.mark.skip(reason="this check is not yet implemented")
    def test_tupled_address_conflict(self):
        @genjax.gen
        def submodel():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.gen
        def model():
            _ = genjax.normal(0.0, 1.0) @ ("x", "y")
            return submodel() @ "x"

        with pytest.raises(Exception):
            tr = model.simulate(())
            tr.get_choices()


class TestProject:
    def test_project(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(0.0, 1.0) @ "y"
            return x, y

        # get a trace
        tr = f.simulate(())
        # evaluations
        x_score = tr.project(S["x"])
        with pytest.deprecated_call():
            assert x_score == tr.get_subtrace(("x",)).get_score()
        assert x_score == tr.get_subtrace("x").get_score()

        y_score = tr.project(S["y"])
        with pytest.deprecated_call():
            assert y_score == tr.get_subtrace(("y",)).get_score()
        assert y_score == tr.get_subtrace("y").get_score()

        assert tr.get_score() == x_score + y_score


class TestGetSubtrace:
    def test_get_subtrace(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(0.0, 1.0) @ "y"
            return x, y

        @genjax.gen
        def g():
            x, y = f() @ "f"
            return x + y

        @genjax.gen
        def h():
            return g() @ "g"

        tr = g.simulate(())
        f_tr = tr.get_subtrace("f")
        assert isinstance(f_tr, DefTrace)
        assert (
            tr.get_subtrace("f", "x").get_score() == f_tr.get_subtrace("x").get_score()
        )
        assert (
            tr.get_subtrace("f", "y").get_score() == f_tr.get_subtrace("y").get_score()
        )

        tr = h.simulate(())
        assert (
            tr.get_subtrace("g").get_subtrace("f").get_subtrace("x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )
        assert (
            tr.get_subtrace("g").get_subtrace("f", "x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )
        assert (
            tr.get_subtrace("g", "f").get_subtrace("x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )

    def test_get_subtrace_switch(self):
        @genjax.gen
        def f():
            return genjax.normal(0.0, 0.01) @ "x"

        @genjax.gen
        def g():
            return genjax.uniform(10.0, 11.0) @ "y"

        @genjax.gen
        def h():
            flip = genjax.flip(0.5) @ "flip"
            return f.or_else(g)(flip, (), ()) @ "z"

        tr = h.simulate(())
        flip_tr = tr.get_subtrace("flip")
        flip = flip_tr.get_retval()
        if flip:
            assert (
                tr.get_subtrace("z", "x").get_score()
                == tr.get_score() - flip_tr.get_score()
            )
        else:
            assert (
                tr.get_subtrace("z", "y").get_score()
                == tr.get_score() - flip_tr.get_score()
            )

    def test_get_subtrace_vmap(self):
        @genjax.vmap()
        @genjax.gen
        def f(x):
            return genjax.normal(x, 0.01) @ "y"

        tr = f.simulate((jnp.arange(5.0),))
        assert tr.get_subtrace("y").get_score().shape == ()
        assert tr.get_score() == jnp.sum(tr.get_subtrace("y").get_score())

    def test_get_subtrace_scan(self):
        @genjax.gen
        def f(state, step):
            return state + genjax.normal(step, 0.01) @ "y", None

        tr = f.scan().simulate((5.0, jnp.arange(3.0)))
        assert tr.get_subtrace("y").get_score().shape == ()
        assert tr.get_score() == pytest.approx(
            jnp.sum(tr.get_subtrace("y").get_score()), 1e-4
        )


class TestCombinators:
    """Tests for the generative function combinator methods."""

    def test_vmap(self):
        @genjax.gen
        def model(x):
            v = genjax.normal(x, 1.0) @ "v"
            return (v, genjax.normal(v, 0.01) @ "q")

        vmapped_model = model.vmap()

        jit_fn = jax.jit(vmapped_model.simulate)

        tr = jit_fn((jnp.array([10.0, 20.0, 30.0]),))
        chm = tr.get_choices()
        varr, qarr = tr.get_retval()

        # The : syntax groups everything under a sub-key:
        assert jnp.array_equal(chm[:, "v"], varr)
        assert jnp.array_equal(chm[:, "q"], qarr)

    def test_repeat(self):
        @genjax.gen
        def model(x):
            return genjax.normal(x, 1.0) @ "x"

        vmap_model = model.vmap()
        repeat_model = model.repeat(n=3)

        vmap_tr = jax.jit(vmap_model.simulate)((jnp.zeros(3),))
        repeat_tr = jax.jit(repeat_model.simulate)((0.0,))

        repeatarr = repeat_tr.get_retval()
        varr = vmap_tr.get_retval()

        # Check that we get 3 repeated values:
        assert jnp.array_equal(repeat_tr.get_choices()[:, "x"], repeatarr)

        # check that the return value matches the traced values (in this case)
        assert jnp.array_equal(repeat_tr.get_retval(), repeatarr)

        # vmap does as well, but they are different due to internal seed splitting:
        assert jnp.array_equal(vmap_tr.get_choices()[:, "x"], varr)

    def test_or_else(self):
        @genjax.gen
        def if_model(x):
            return genjax.normal(x, 1.0) @ "if_value"

        @genjax.gen
        def else_model(x):
            return genjax.normal(x, 5.0) @ "else_value"

        @genjax.gen
        def switch_model(toss: bool):
            return if_model.or_else(else_model)(toss, (1.0,), (10.0,)) @ "tossed"

        jit_fn = jax.jit(switch_model.simulate)
        if_tr = jit_fn((True,))
        assert "if_value" in if_tr.get_choices()("tossed")

        else_tr = jit_fn((False,))
        assert "else_value" in else_tr.get_choices()("tossed")
