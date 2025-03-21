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
"""This module contains the `Distribution` abstract base class."""

import textwrap
import warnings
from abc import abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp

from genjax._src.checkify import optional_check
from genjax._src.core.compiler.interpreters.incremental import Diff
from genjax._src.core.compiler.pjax import seed
from genjax._src.core.compiler.staging import FlagOp
from genjax._src.core.generative import (
    GFI,
    Argdiffs,
    ChoiceMap,
    EditRequest,
    Mask,
    NotSupportedEditRequest,
    R,
    Regenerate,
    Retdiff,
    Score,
    Selection,
    Trace,
    Update,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    PRNGKey,
)

tfd = tfp.distributions

#####
# DistributionTrace
#####


@Pytree.dataclass
class DistributionTrace(Generic[R], Trace[R]):
    distribution: "Distribution[R]"
    args: tuple[Any, ...]
    value: R

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> R:
        return self.value

    def get_gen_fn(self) -> "Distribution[R]":
        return self.distribution

    def get_score(self) -> Score:
        gen_fn = self.get_gen_fn()
        w = gen_fn.logpdf(self.value, *self.args)
        if jnp.shape(w):
            return jnp.sum(w)
        else:
            return w

    def get_choices(self) -> ChoiceMap:
        return ChoiceMap.choice(self.value)


################
# Distribution #
################


class Distribution(Generic[R], GFI[R]):
    @abstractmethod
    def sample(
        self,
        *args,
        **kwargs,
    ) -> R:
        pass

    @abstractmethod
    def logpdf(
        self,
        v: R,
        *args,
        **kwargs,
    ) -> Score:
        pass

    def sample_and_logpdf(
        self,
        *args,
        **kwargs,
    ) -> tuple[Score, R]:
        v = self.sample(*args)
        w = self.logpdf(v, *args)
        return w, v

    def sample_with_key(
        self,
        key: PRNGKey,
        *args,
        **kwargs,
    ) -> R:
        return seed(self.sample)(key, *args)

    def simulate(
        self,
        args: tuple[Any, ...],
    ) -> Trace[R]:
        v = self.sample(*args)
        tr = DistributionTrace(self, args, v)
        return tr

    def generate_choice_map(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight]:
        v = chm.get_value()
        match v:
            case None:
                tr = self.simulate(args)
                return tr, jnp.array(0.0)

            case Mask(value, flag):

                def _simulate(v):
                    new_v = self.sample(*args)
                    w = 0.0
                    return (w, new_v)

                def _importance(v):
                    w = self.logpdf(v, *args)
                    return (w, v)

                w, new_v = jax.lax.cond(flag, _importance, _simulate, value)
                tr = DistributionTrace(self, args, new_v)
                return tr, w

            case _:
                tr = DistributionTrace(self, args, v)
                w = self.logpdf(v, *args)
                return tr, w

    def generate(
        self,
        constraint: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[R], Weight]:
        match constraint:
            case ChoiceMap():
                tr, w = self.generate_choice_map(constraint, args)

            case _:
                raise Exception("Unhandled type.")
        return tr, w

    def edit_empty(
        self,
        trace: Trace[R],
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        chm = trace.get_choices()
        primals = Diff.tree_primal(argdiffs)
        v = chm.get_value()
        new_trace = DistributionTrace(self, primals, v)
        new_score, _ = self.assess(chm, primals)
        return (
            new_trace,
            new_score - trace.get_score(),
            Diff.no_change(trace.get_retval()),
            Update(ChoiceMap.empty()),
        )

    def edit_update_with_constraint(
        self,
        trace: Trace[R],
        constraint: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        primals = Diff.tree_primal(argdiffs)
        match constraint:
            case ChoiceMap():
                match constraint.get_value():
                    case Mask() as masked_value:

                        def _true_branch(new_value: R, _):
                            fwd = self.logpdf(new_value, *primals)
                            bwd = trace.get_score()
                            w = fwd - bwd
                            return (new_value, w)

                        def _false_branch(_, old_value: R):
                            fwd = self.logpdf(old_value, *primals)
                            bwd = trace.get_score()
                            w = fwd - bwd
                            return (old_value, w)

                        flag = masked_value.primal_flag()
                        new_value: R = masked_value.value
                        old_choices = trace.get_choices()
                        old_value: R = old_choices.get_value()

                        new_value, w = FlagOp.cond(
                            flag,
                            _true_branch,
                            _false_branch,
                            new_value,
                            old_value,
                        )
                        return (
                            DistributionTrace(self, primals, new_value),
                            w,
                            Diff.unknown_change(new_value),
                            Update(
                                old_choices.mask(flag),
                            ),
                        )
                    case None:
                        value_chm = trace.get_choices()
                        v = value_chm.get_value()
                        fwd = self.logpdf(v, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        new_tr = DistributionTrace(self, primals, v)
                        retval_diff = Diff.no_change(v)
                        return (new_tr, w, retval_diff, Update(ChoiceMap.empty()))

                    case v:
                        fwd = self.logpdf(v, *primals)
                        bwd = trace.get_score()
                        w = fwd - bwd
                        new_tr = DistributionTrace(self, primals, v)
                        discard = trace.get_choices()
                        retval_diff = Diff.unknown_change(v)
                        return (new_tr, w, retval_diff, Update(discard))
            case _:
                raise Exception(f"Unhandled constraint in edit: {type(constraint)}.")

    def project(
        self,
        trace: Trace[R],
        selection: Selection,
    ) -> Weight:
        return jnp.where(
            selection.check(),
            trace.get_score(),
            jnp.array(0.0),
        )

    def edit_regenerate(
        self,
        trace: Trace[R],
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        check = () in selection
        if FlagOp.concrete_true(check):
            primals = Diff.tree_primal(argdiffs)
            w, new_v = self.sample_and_logpdf(*primals)
            incremental_w = w - trace.get_score()
            old_v = trace.get_retval()
            new_trace = DistributionTrace(self, primals, new_v)
            return (
                new_trace,
                incremental_w,
                Diff.unknown_change(new_v),
                Update(ChoiceMap.choice(old_v)),
            )
        elif FlagOp.concrete_false(check):
            if Diff.static_check_no_change(argdiffs):
                return (
                    trace,
                    jnp.array(0.0),
                    Diff.no_change(trace.get_retval()),
                    Update(ChoiceMap.empty()),
                )
            else:
                chm = trace.get_choices()
                primals = Diff.tree_primal(argdiffs)
                new_score, _ = self.assess(chm, primals)
                new_trace = DistributionTrace(self, primals, chm.get_value())
                return (
                    new_trace,
                    new_score - trace.get_score(),
                    Diff.no_change(trace.get_retval()),
                    Update(
                        ChoiceMap.empty(),
                    ),
                )
        else:
            raise NotImplementedError

    def edit_update(
        self,
        trace: Trace[R],
        constraint: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], Update]:
        match constraint:
            case ChoiceMap():
                return self.edit_update_with_constraint(trace, constraint, argdiffs)

            case _:
                raise Exception(f"Not implement fwd problem: {constraint}.")

    def edit(
        self,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[R], Weight, Retdiff[R], EditRequest]:
        match edit_request:
            case Update(chm):
                return self.edit_update(
                    trace,
                    chm,
                    argdiffs,
                )
            case Regenerate(selection):
                return self.edit_regenerate(
                    trace,
                    selection,
                    argdiffs,
                )

            case _:
                raise NotSupportedEditRequest(edit_request)

    def assess(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Weight, R]:
        v = chm.get_value()
        match v:
            case Mask(value, flag):

                def _check():
                    checkify.check(
                        bool(flag),
                        "Attempted to unmask when a mask flag is False: the masked value is invalid.\n",
                    )

                optional_check(_check)
                w = self.logpdf(value, *args)
                return w, value
            case _:
                w = self.logpdf(v, *args)
                return w, v


def canonicalize_distribution_name(s: str) -> str:
    """Converts underlying distribution name from CamelCase to snake_case
    and prepends `genjax.`"""
    t = []
    for c in s:
        if c.isupper():
            if t:
                t.append("_")
            t.append(c.lower())
        else:
            t.append(c)
    return "genjax." + "".join(t)


def distribution(
    sample: Callable[..., R],
    logpdf: Callable[..., Score],
    name: str | None = None,
) -> Distribution[R]:
    """Construct a new type, a subclass of Distribution, with the given name,
    (with `genjax.` prepended, to avoid confusion with the underlying object,
    which may not share the same interface) and attach the supplied functions
    as the `sample` and `logpdf` methods. The return value is an instance of
    this new type, and should be treated as a singleton."""
    if name is None:
        warnings.warn("You should supply a name argument to exact_density")
        name = "unknown"

    def kwargle(f, args, kwargs):
        """Keyword arguments currently get unusual treatment in GenJAX: when
        a keyword argument is provided to a generative function, the function
        is asked to provide a new version of itself which receives a different
        signature: `(args, kwargs)` instead of `(*args, **kwargs)`. The
        replacement of the GF with a new object may cause JAX to believe that
        the implementations are materially different. To avoid this, we
        reply to the handle_kwargs request with self and infer kwargs handling
        by seeing whether we were passed a 2-tuple with a dict in the [1] slot.
        We are assuming that this will not represent a useful argument package
        to any of the TF distributions."""
        if len(args) == 2 and isinstance(args[1], dict):
            return f(*args[0], **args[1])
        else:
            return f(*args, **kwargs)

    def kwargle_v(f, a0, args, kwargs):
        if len(args) == 2 and isinstance(args[1], dict):
            return f(a0, *args[0], **args[1])
        else:
            return f(a0, *args, **kwargs)

    T = type(
        canonicalize_distribution_name(name),
        (Distribution,),
        {
            "sample": lambda self, *args, **kwargs: kwargle(sample, args, kwargs),
            "logpdf": lambda self, v, *args, **kwargs: kwargle_v(
                logpdf, v, args, kwargs
            ),
            "handle_kwargs": lambda self: self,
        },
    )

    return Pytree.dataclass(T)()


def implicit_logit_warning(dist):
    """Early versions of GenJAX interpreted bare parameters to certain distributions
    in logit scale, but many newcomers to probabilistic programming may be used to thinking
    in probability scale and expect this to be the default. To conserve the meaning of
    GenJAX programs, use of a bare parameter in these cases now provokes a warning
    requesting that the caller make an explicit choice."""

    def wrapper(implicit_logits=None, **kwargs):
        if implicit_logits is not None:
            warnings.warn(
                textwrap.dedent(
                    f"""
                    The use of a bare argument to {canonicalize_distribution_name(dist.__name__)}
                    is deprecated. Please specify `logits=` or `probs=` for the parameters.
                    The default, which will be used in this case, is logits."""
                ),
                DeprecationWarning,
            )
            return dist(logits=implicit_logits, **kwargs)
        return dist(**kwargs)

    return wrapper
