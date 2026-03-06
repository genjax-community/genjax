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

import functools
from abc import abstractmethod
from dataclasses import dataclass
from operator import or_
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.tree_util as jtu
import treescope.repr_lib as trl
from deprecated import deprecated

from genjax._src.core.generative.functional_types import DynamicAddressComponent, Mask
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Array,
    Callable,
    EllipsisType,
    Final,
    Flag,
    Generic,
    IntArray,
    Iterable,
    TypeVar,
    nobeartype,
)

if TYPE_CHECKING:
    import genjax

#################
# Address types #
#################

# Address components
StaticAddressComponent = str
AddressComponent = StaticAddressComponent | DynamicAddressComponent
ExtendedStaticAddressComponent = StaticAddressComponent | EllipsisType
ExtendedAddressComponent = ExtendedStaticAddressComponent | DynamicAddressComponent

# Addresses
Address = AddressComponent | tuple[AddressComponent, ...]

StaticAddress = StaticAddressComponent | tuple[StaticAddressComponent, ...]
ExtendedStaticAddress = (
    ExtendedStaticAddressComponent | tuple[ExtendedStaticAddressComponent, ...]
)
ExtendedAddress = ExtendedAddressComponent | tuple[ExtendedAddressComponent, ...]

T = TypeVar("T")
K_addr = TypeVar("K_addr", bound=Address)

_full_slice = slice(None, None, None)

##############
# Selections #
##############

###############################
# Selection builder interface #
###############################


class _SelectionBuilder:
    @property
    def all(self) -> "Selection":
        """
        Returns a Selection that selects all addresses.

        Returns:
            A Selection that selects everything.
        """
        return Selection.all()

    @property
    def none(self) -> "Selection":
        """
        Returns a Selection that selects no addresses.

        Returns:
            A Selection that selects nothing.
        """
        return Selection.none()

    @property
    def leaf(self) -> "Selection":
        """
        Returns a Selection that selects only leaf addresses.

        A leaf address is an address that doesn't have any sub-addresses.
        This selection is useful when you want to target only the final elements in a nested structure.

        Returns:
            A Selection that selects only leaf addresses.
        """
        return Selection.leaf()

    def __getitem__(self, addr: ExtendedStaticAddress) -> "Selection":
        addr = addr if isinstance(addr, tuple) else (addr,)
        if addr == ():
            return Selection.leaf()
        else:
            return Selection.all().extend(*addr)


SelectionBuilder = _SelectionBuilder()
"""Deprecated! please use `Selection.at`."""


class Selection(Pytree):
    """
    A class representing a selection of addresses in a ChoiceMap.

    Selection objects are used to filter and manipulate ChoiceMaps by specifying which addresses should be included or excluded.

    Selection instances support various operations such as union (via `&`), intersection (via `|`), and complement (via `~`), allowing for complex selection criteria to be constructed.

    Methods:
        all(): Creates a Selection that includes all addresses.
        none(): Creates a Selection that includes no addresses.
        at: A builder instance for creating Selection objects using indexing syntax.

    Examples:
        Creating selections:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        import genjax
        from genjax import Selection

        # Select all addresses
        all_sel = Selection.all()

        # Select no addresses
        none_sel = Selection.none()

        # Select specific addresses
        specific_sel = Selection.at["x", "y"]

        # Match (<wildcard>, "y")
        wildcard_sel = Selection.at[..., "y"]

        # Combine selections
        combined_sel = specific_sel | Selection.at["z"]
        ```

        Querying selections:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        # Create a selection
        sel = Selection.at["x", "y"]

        # Querying the selection using () returns a sub-selection
        assert sel("x") == Selection.at["y"]
        assert sel("z") == Selection.none()

        # Querying the selection using [] returns a `bool` representing whether or not the input matches:
        assert sel["x"] == False
        assert sel["x", "y"] == True

        # Querying the selection using "in" acts the same:
        assert not "x" in sel
        assert ("x", "y") in sel

        # Nested querying
        nested_sel = Selection.at["a", "b", "c"]
        assert nested_sel("a")("b") == Selection.at["c"]
        ```

    Selection objects can passed to a `ChoiceMap` via the `filter` method to filter and manipulate data based on address patterns.
    """

    #################################################
    # Convenient syntax for constructing selections #
    #################################################

    at: Final[_SelectionBuilder] = _SelectionBuilder()
    """A builder instance for creating Selection objects.

    `at` provides a convenient interface for constructing Selection objects
    using a familiar indexing syntax. It allows for the creation of complex
    selections by chaining multiple address components.

    Examples:
        Creating a selection:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import Selection
        Selection.at["x", "y"]
        ```
    """

    @staticmethod
    def all() -> "Selection":
        """
        Returns a Selection that selects all addresses.

        Returns:
            A Selection that selects everything.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            from genjax import Selection

            all_selection = Selection.all()
            assert all_selection["any_address"] == True
            ```
        """
        return AllSel()

    @staticmethod
    def none() -> "Selection":
        """
        Returns a Selection that selects no addresses.

        Returns:
            A Selection that selects nothing.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            none_selection = Selection.none()
            assert none_selection["any_address"] == False
            ```
        """
        return NoneSel()

    @staticmethod
    def leaf() -> "Selection":
        """
        Returns a Selection that selects only leaf addresses.

        A leaf address is an address that doesn't have any sub-addresses.
        This selection is useful when you want to target only the final elements in a nested structure.

        Returns:
            A Selection that selects only leaf addresses.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            leaf_selection = Selection.leaf().extend("a", "b")
            assert leaf_selection["a", "b"]
            assert not leaf_selection["a", "b", "anything"]
            ```
        """
        return LeafSel()

    ######################
    # Combinator methods #
    ######################

    def __or__(self, other: "Selection") -> "Selection":
        return OrSel.build(self, other)

    def __and__(self, other: "Selection") -> "Selection":
        return AndSel.build(self, other)

    def __invert__(self) -> "Selection":
        return ComplementSel.build(self)

    def complement(self) -> "Selection":
        return ~self

    def filter(self, sample: "ChoiceMap") -> "ChoiceMap":
        """
        Returns a new ChoiceMap filtered with this Selection.

        This method applies the current Selection to the given ChoiceMap, effectively filtering out addresses that are not matched.

        Args:
            sample: The ChoiceMap to be filtered.

        Returns:
            A new ChoiceMap containing only the addresses selected by this Selection.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            selection = Selection.at["x"]

            chm = ChoiceMap.kw(x=1, y=2)
            filtered_chm = selection.filter(chm)

            assert "x" in filtered_chm
            assert "y" not in filtered_chm
            ```
        """
        return sample.filter(self)

    def extend(self, *addrs: ExtendedStaticAddressComponent) -> "Selection":
        """
        Returns a new Selection that is prefixed by the given address components.

        This method creates a new Selection that applies the current selection
        to the specified address components. It handles both static and dynamic
        address components.

        Note that `...` as an address component will match any supplied address.

        Args:
            addrs: The address components under which to nest the selection.

        Returns:
            A new Selection extended by the given address component.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            base_selection = Selection.all()
            indexed_selection = base_selection.extend("x")
            assert indexed_selection["x", "any_subaddress"] == True
            assert indexed_selection["y"] == False
            ```
        """
        acc = self
        for addr in reversed(addrs):
            acc = StaticSel.build(acc, addr)
        return acc

    def __call__(
        self,
        addr: StaticAddress,
    ) -> "Selection":
        addr = addr if isinstance(addr, tuple) else (addr,)
        subselection = self
        for comp in addr:
            subselection = subselection.get_subselection(comp)
        return subselection

    def __getitem__(
        self,
        addr: StaticAddress,
    ) -> bool:
        return self(addr).check()

    def __contains__(
        self,
        addr: StaticAddress,
    ) -> bool:
        return self[addr]

    @abstractmethod
    def check(self) -> bool:
        pass

    @abstractmethod
    def get_subselection(self, addr: StaticAddressComponent) -> "Selection":
        pass


#######################
# Selection functions #
#######################


@Pytree.dataclass(match_args=True)
class AllSel(Selection):
    """Represents a selection that includes all addresses.

    This selection always returns True for any address check and returns itself
    for any subselection, effectively representing a selection of all possible
    addresses in a choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        all_sel = Selection.all()
        assert all_sel["any_address"] == True
        ```
    """

    def check(self) -> bool:
        return True

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        return self


@Pytree.dataclass(match_args=True)
class NoneSel(Selection):
    """Represents a selection that includes no addresses.

    This selection always returns False for any address check and returns itself
    for any subselection, effectively representing an empty selection that
    matches no addresses in a choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        none_sel = Selection.none()
        assert none_sel["any_address"] == False
        assert none_sel.get_subselection("any_address") == none_sel
        ```
    """

    def check(self) -> bool:
        return False

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        return self


@Pytree.dataclass
class LeafSel(Selection):
    """Represents a selection that matches only at the current address level.

    This selection returns True for a check at the current level but returns an
    empty selection (`Selection.none()`) for any subselection, effectively representing a
    leaf node in the selection hierarchy.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        leaf_sel = LeafSel()
        assert leaf_sel.check()
        assert isinstance(leaf_sel.get_subselection("any_address"), NoneSel)
        ```
    """

    def check(self) -> bool:
        return True

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        return Selection.none()


@Pytree.dataclass(match_args=True)
class ComplementSel(Selection):
    """Represents the complement of a selection.

    This selection inverts the behavior of another selection. It checks for the
    opposite of what the wrapped selection checks for, and returns the complement
    of its subselections.

    Attributes:
        s: The selection to be complemented.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        base_sel = Selection.all()
        comp_sel = ~base_sel
        assert comp_sel.check() == False

        specific_sel = Selection.at["x", "y"]
        comp_specific = ~specific_sel
        assert comp_specific["x", "y"] == False
        assert comp_specific["z"] == True
        ```
    """

    s: Selection

    @staticmethod
    def build(s: Selection) -> Selection:
        match s:
            case AllSel():
                return Selection.none()
            case NoneSel():
                return Selection.all()
            case ComplementSel():
                return s.s
            case _:
                return ComplementSel(s)

    def check(self) -> bool:
        return not self.s.check()

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        remaining = self.s(addr)
        return ~remaining


@Pytree.dataclass(match_args=True)
class StaticSel(Selection):
    """Represents a static selection based on a specific address component.

    This selection is used to filter choices based on a static address component.
    It always returns False for the check method, as it's meant to be used in
    combination with other selections or as part of a larger selection structure.

    Attributes:
        s: The underlying selection to be applied if the address matches.
        addr: The static address component to match against.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        static_sel = Selection.at["x"]
        assert static_sel.check() == False
        assert static_sel.get_subselection("x").check() == True
        assert static_sel.get_subselection("y").check() == False
        ```
    """

    s: Selection = Pytree.field()
    addr: ExtendedStaticAddressComponent = Pytree.static()

    @staticmethod
    def build(
        s: Selection,
        addr: ExtendedStaticAddressComponent,
    ) -> Selection:
        match s:
            case NoneSel():
                return s
            case _:
                return StaticSel(s, addr)

    def check(self) -> bool:
        return False

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        if isinstance(self.addr, EllipsisType):
            return self.s

        if addr == self.addr:
            return self.s
        else:
            return Selection.none()


@Pytree.dataclass(match_args=True)
class AndSel(Selection):
    """Represents a selection that combines two other selections using a logical AND operation.

    This selection is true only if both of its constituent selections are true. It allows for the combination of multiple selection criteria.

    Attributes:
        s1: The first selection to be combined.
        s2: The second selection to be combined.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel1 = Selection.at["y"] | Selection.at["x"]
        sel2 = Selection.at["y"] | Selection.at["z"]
        and_sel = sel1 & sel2

        assert and_sel["x"] == False
        assert and_sel["y"] == True
        assert and_sel["z"] == False
        ```
    """

    s1: Selection
    s2: Selection

    @staticmethod
    def build(a: Selection, b: Selection) -> Selection:
        match (a, b):
            case (AllSel(), _):
                return b
            case (_, AllSel()):
                return a
            case (NoneSel(), _):
                return a
            case (_, NoneSel()):
                return b
            case (a, b) if a == b:
                return a
            case _:
                return AndSel(a, b)

    def check(self) -> bool:
        return self.s1.check() and self.s2.check()

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return remaining1 & remaining2


@Pytree.dataclass(match_args=True)
class OrSel(Selection):
    """Represents a selection that combines two other selections using a logical OR operation.

    This selection is true if either of its constituent selections is true.
    It allows for the combination of multiple selection criteria using an inclusive OR.

    Attributes:
        s1: The first selection to be combined.
        s2: The second selection to be combined.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        sel1 = Selection.at["x"]
        sel2 = Selection.at["y"]
        or_sel = sel1 | sel2

        assert or_sel["x", "y"] == True
        assert or_sel["x"] == True
        assert or_sel["y"] == True
        assert or_sel["z"] == False
        ```
    """

    s1: Selection
    s2: Selection

    @staticmethod
    def build(a: Selection, b: Selection) -> Selection:
        match (a, b):
            case (AllSel(), _):
                return a
            case (_, AllSel()):
                return b
            case (NoneSel(), _):
                return b
            case (_, NoneSel()):
                return a
            case (a, b) if a == b:
                return a
            case _:
                return OrSel(a, b)

    def check(self) -> bool:
        return self.s1.check() or self.s2.check()

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        remaining1 = self.s1(addr)
        remaining2 = self.s2(addr)
        return remaining1 | remaining2


@Pytree.dataclass(match_args=True)
class ChmSel(Selection):
    """Represents a selection based on a ChoiceMap.

    This selection is True for addresses that have a value in the associated ChoiceMap, False otherwise.
    It allows for creating selections that match the structure of existing ChoiceMaps.

    Attributes:
        c: The ChoiceMap on which this selection is based.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0) ^ C["z"].set(5.0)
        sel = chm.get_selection()
        assert sel["x", "y"] == True
        assert sel["z"] == True
        assert sel["w"] == False
        ```
    """

    c: "ChoiceMap"

    @staticmethod
    def build(chm: "ChoiceMap") -> Selection:
        if chm.static_is_empty():
            return Selection.none()
        else:
            return ChmSel(chm)

    def check(self) -> bool:
        return self.c.has_value()

    def get_subselection(self, addr: StaticAddressComponent) -> Selection:
        submap = self.c.get_inner_map(addr)
        return submap.get_selection()


###############
# Choice maps #
###############


@dataclass(frozen=True)
class ChoiceMapNoValueAtAddress(Exception):
    """Exception raised when a value is not found at a specified address in a ChoiceMap.

    This exception is thrown when attempting to access a value in a ChoiceMap at an address
    where no value exists.

    Attributes:
        subaddr: The address or sub-address where the value was not found.
    """

    subaddr: Address


def _drop_prefix(
    dynamic_components: list[DynamicAddressComponent],
) -> list[DynamicAddressComponent]:
    # Check for prefix of int or scalar Array instances
    prefix_end = 0
    for comp in dynamic_components:
        if isinstance(comp, int) or (isinstance(comp, Array) and comp.shape == ()):
            prefix_end += 1
        else:
            break

    return dynamic_components[prefix_end:]


def _validate_addr(
    addr: tuple[AddressComponent, ...], allow_partial_slice: bool = False
) -> tuple[AddressComponent, ...]:
    """
    Validates the structure of an address tuple.

    This function checks if the given address adheres to the following structure:

    1. A prefix consisting of only scalar addresses (int or an IntArray with shape == ())
    2. Optionally (if `allow_partial_slice` is True), a single non-full-slice or non-scalar array
    3. A tail of full slices (: or slice(None,None,None))

    Args:
        addr: The address (or address component) to validate.
        allow_partial_slice: If True, allows a single partial slice or non-scalar array. Defaults to False.

    Returns:
        The validated address tuple.

    Raises:
        ValueError: If the address structure is invalid.
    """
    dynamic_components = [
        comp for comp in addr if isinstance(comp, (slice, int, Array))
    ]

    if dynamic_components:
        remaining = _drop_prefix(dynamic_components)

        if len(remaining) > 0:
            first = remaining[0]
            if isinstance(first, Array) and first.shape != ():
                remaining = remaining[1:]
            elif (
                allow_partial_slice
                and isinstance(first, slice)
                and first != _full_slice
            ):
                remaining = remaining[1:]

        if not all(s == _full_slice for s in remaining):
            if allow_partial_slice:
                caveat = "an optional partial slice or Array, and then only full slices"
            else:
                caveat = "full slices"

            raise ValueError(
                f"Address must consist of scalar components, followed by {caveat}. Found: {dynamic_components}"
            )

    return addr


class _ChoiceMapBuilder:
    choice_map: "ChoiceMap | None"
    addrs: list[AddressComponent]

    def __init__(self, choice_map: "ChoiceMap | None", addrs: list[AddressComponent]):
        self.choice_map = choice_map
        self.addrs = addrs

    def __getitem__(self, addr: Address) -> "_ChoiceMapBuilder":
        addr = addr if isinstance(addr, tuple) else (addr,)
        return _ChoiceMapBuilder(
            self.choice_map,
            [*self.addrs, *addr],
        )

    def set(self, v) -> "ChoiceMap":
        addrs = _validate_addr(tuple(self.addrs), allow_partial_slice=False)
        chm = ChoiceMap.entry(v, *addrs)
        if self.choice_map is None:
            return chm
        else:
            return chm + self.choice_map

    def update(
        self, f: Callable[..., "dict[K_addr, Any] | ChoiceMap | Any"]
    ) -> "ChoiceMap":
        """
        Updates an existing value or ChoiceMap at the current address.
        This method allows updating a value or ChoiceMap at the address specified by the builder.
        The provided function `f` is called with the current value or ChoiceMap at that address.
        Args:
            f: A callable that takes the current value (or None) and returns a new value,
               dict, or ChoiceMap to be set at the current address.
        Returns:
            A new ChoiceMap with the updated value at the specified address.
        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMap.d({"x": 5, "y": {"z": 10}})
            updated = chm.at["y", "z"].update(lambda v: v * 2)
            assert updated["y", "z"] == 20
            # Updating a non-existent address
            new_chm = chm.at["w"].update(lambda _: 42)
            assert new_chm["w"] == 42
            ```
        """
        if self.choice_map is None:
            return self.set(f(_empty))
        else:
            submap = self.choice_map(tuple(self.addrs))
            if submap.has_value():
                return self.set(f(submap.get_value()))
            else:
                return self.set(f(submap))

    def n(self) -> "ChoiceMap":
        """
        Returns an empty ChoiceMap. Alias for `ChoiceMap.none()`.

        Returns:
            An empty ChoiceMap.
            ```
        """
        return _empty

    def v(self, v) -> "ChoiceMap":
        """
        Nests a call to `ChoiceMap.value` under the current address held by the builder.
        """
        return self.set(ChoiceMap.choice(v))

    def from_mapping(self, mapping: Iterable[tuple[K_addr, Any]]) -> "ChoiceMap":
        """
        Nests a call to `ChoiceMap.from_mapping` under the current address held by the builder.
        """
        return self.set(ChoiceMap.from_mapping(mapping))

    def d(self, d: dict[K_addr, Any]) -> "ChoiceMap":
        """
        Nests a call to `ChoiceMap.d` under the current address held by the builder.
        """
        return self.set(ChoiceMap.d(d))

    def kw(self, **kwargs) -> "ChoiceMap":
        """
        Nests a call to `ChoiceMap.kw` under the current address held by the builder.
        """
        return self.set(ChoiceMap.kw(**kwargs))

    def switch(self, idx: int | IntArray, chms: Iterable["ChoiceMap"]) -> "ChoiceMap":
        """
        Nests a call to `ChoiceMap.switch` under the current address held by the builder.
        """
        return self.set(ChoiceMap.switch(idx, chms))


class ChoiceMap(Pytree):
    """The type `ChoiceMap` denotes a map-like value which can be sampled from
    generative functions.

    Generative functions which utilize `ChoiceMap` as their sample representation typically support a notion of _addressing_ for the random choices they make. `ChoiceMap` stores addressed random choices, and provides a data language for querying and manipulating these choices.

    Examples:
        (**Making choice maps**) Choice maps can be constructed using the `ChoiceMapBuilder` interface
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x"].set(3.0)
        print(chm.render_html())
        ```

        (**Getting submaps**) Hierarchical choice maps support `__call__`, which allows for the retrieval of _submaps_ at addresses:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        submap = chm("x")
        print(submap.render_html())
        ```

        (**Getting values**) Choice maps support `__getitem__`, which allows for the retrieval of _values_ at addresses:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C

        chm = C["x", "y"].set(3.0)
        value = chm["x", "y"]
        print(value)
        ```

        (**Making vectorized choice maps**) Choice maps can be constructed using `jax.vmap`:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        from genjax import ChoiceMapBuilder as C
        from jax import vmap
        import jax.numpy as jnp

        vec_chm = vmap(lambda idx, v: C["x", idx].set(v))(jnp.arange(10), jnp.ones(10))
        print(vec_chm.render_html())
        ```
    """

    #######################
    # Map-like interfaces #
    #######################

    @abstractmethod
    def filter(self, selection: Selection | Flag) -> "ChoiceMap":
        """
        Filter the choice map on the `Selection`. The resulting choice map only contains the addresses that return True when presented to the selection.

        Args:
            selection: The Selection to filter the choice map with.

        Returns:
            A new ChoiceMap containing only the addresses selected by the given Selection.

        Examples:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            import jax
            import genjax
            from genjax import bernoulli
            from genjax import SelectionBuilder as S


            @genjax.gen
            def model():
                x = bernoulli(0.3) @ "x"
                y = bernoulli(0.3) @ "y"
                return x


            key = jax.random.key(314159)
            tr = model.simulate(key, ())
            chm = tr.get_choices()
            selection = S["x"]
            filtered = chm.filter(selection)
            assert "y" not in filtered
            ```
        """

    @abstractmethod
    def get_value(self) -> Any:
        pass

    @abstractmethod
    def get_inner_map(
        self,
        addr: AddressComponent,
    ) -> "ChoiceMap":
        pass

    def get_submap(self, *addresses: Address) -> "ChoiceMap":
        addr = tuple(
            label for a in addresses for label in (a if isinstance(a, tuple) else (a,))
        )
        addr: tuple[AddressComponent, ...] = _validate_addr(
            addr, allow_partial_slice=True
        )
        return functools.reduce(lambda chm, addr: chm.get_inner_map(addr), addr, self)

    def has_value(self) -> bool:
        return self.get_value() is not None

    ######################################
    # Convenient syntax for construction #
    ######################################

    builder: Final[_ChoiceMapBuilder] = _ChoiceMapBuilder(None, [])

    @staticmethod
    def empty() -> "ChoiceMap":
        """
        Returns a ChoiceMap with no values or submaps.

        Returns:
            An empty ChoiceMap.
        """
        return _empty

    @staticmethod
    def choice(v: Any) -> "ChoiceMap":
        """
        Creates a ChoiceMap containing a single value.

        This method creates and returns an instance of Choice, which represents
        a ChoiceMap with a single value at the root level.

        Args:
            v: The value to be stored in the ChoiceMap.

        Returns:
            A ChoiceMap containing the single value.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            from genjax import ChoiceMap

            value_chm = ChoiceMap.value(42)
            assert value_chm.get_value() == 42
            ```
        """
        return Choice.build(v)

    @staticmethod
    @nobeartype
    @deprecated("Use ChoiceMap.choice() instead.")
    def value(v: Any) -> "ChoiceMap":
        return ChoiceMap.choice(v)

    @staticmethod
    def entry(
        v: "dict[K_addr, Any] | ChoiceMap | Any", *addrs: AddressComponent
    ) -> "ChoiceMap":
        """
        Creates a ChoiceMap with a single value at a specified address.

        This method creates and returns a ChoiceMap with a new ChoiceMap stored at
        the given address.

        - if the provided value is already a ChoiceMap, it will be used directly;
        - `dict` values will be passed to `ChoiceMap.d`;
        - any other value will be passed to `ChoiceMap.value`.

        Args:
            v: The value to be stored in the ChoiceMap. Can be any value, a dict or a ChoiceMap.
            addrs: The address at which to store the value. Can be a static or dynamic address component.

        Returns:
            A ChoiceMap with the value stored at the specified address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            import genjax
            import jax.numpy as jnp

            # Using an existing ChoiceMap
            nested_chm = ChoiceMap.entry(ChoiceMap.value(42), "x")
            assert nested_chm["x"] == 42

            # Using a dict generates a new `ChoiceMap.d` call
            nested_chm = ChoiceMap.entry({"y": 42}, "x")
            assert nested_chm["x", "y"] == 42

            # Static address
            static_chm = ChoiceMap.entry(42, "x")
            assert static_chm["x"] == 42

            # Dynamic address
            dynamic_chm = ChoiceMap.entry(
                jnp.array([1.1, 2.2, 3.3]), jnp.array([1, 2, 3])
            )
            assert dynamic_chm[1] == genjax.Mask(1.1, True)
            ```
        """
        if isinstance(v, ChoiceMap):
            chm = v
        elif isinstance(v, dict):
            chm = ChoiceMap.d(v)
        else:
            chm = ChoiceMap.choice(v)

        return chm.extend(*addrs)

    @staticmethod
    def from_mapping(pairs: Iterable[tuple[K_addr, Any]]) -> "ChoiceMap":
        """
        Creates a ChoiceMap from an iterable of address-value pairs.

        This method constructs a ChoiceMap by iterating through the provided pairs,
        where each pair consists of an address (or address component) and a corresponding value.
        The resulting ChoiceMap will contain all the values at their respective addresses.

        Args:
            pairs: An iterable of tuples, where each tuple contains an address (or address component) and its corresponding value. The address can be a single component or a tuple of components.

        Returns:
            A ChoiceMap containing all the address-value pairs from the input.

        Example:
            ```python
            pairs = [("x", 42), (("y", "z"), 10), ("w", [1, 2, 3])]
            chm = ChoiceMap.from_mapping(pairs)
            assert chm["x"] == 42
            assert chm["y", "z"] == 10
            assert chm["w"] == [1, 2, 3]
            ```

        Note:
            If multiple pairs have the same address, later pairs will overwrite earlier ones.
        """
        acc = ChoiceMap.empty()

        for addr, v in pairs:
            addr = addr if isinstance(addr, tuple) else (addr,)
            acc |= ChoiceMap.entry(v, *addr)

        return acc

    @staticmethod
    def d(d: dict[K_addr, Any]) -> "ChoiceMap":
        """
        Creates a ChoiceMap from a dictionary.

        This method creates and returns a ChoiceMap based on the key-value pairs in the provided dictionary. Each key in the dictionary becomes an address in the ChoiceMap, and the corresponding value is stored at that address.

        Dict-shaped values are recursively converted to ChoiceMap instances.

        Args:
            d: A dictionary where keys are addresses and values are the corresponding data to be stored in the ChoiceMap.

        Returns:
            A ChoiceMap containing the key-value pairs from the input dictionary.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            from genjax import ChoiceMap

            dict_chm = ChoiceMap.d({"x": 42, "y": {"z": [1, 2, 3]}})
            assert dict_chm["x"] == 42
            assert dict_chm["y", "z"] == [1, 2, 3]
            ```
        """
        return ChoiceMap.from_mapping(d.items())

    @staticmethod
    def kw(**kwargs) -> "ChoiceMap":
        """
        Creates a ChoiceMap from keyword arguments.

        This method creates and returns a ChoiceMap based on the provided keyword arguments.
        Each keyword argument becomes an address in the ChoiceMap, and its value is stored at that address.

        Dict-shaped values are recursively converted to ChoiceMap instances with calls to `ChoiceMap.d`.

        Returns:
            A ChoiceMap containing the key-value pairs from the input keyword arguments.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            kw_chm = ChoiceMap.kw(x=42, y=[1, 2, 3], z={"w": 10.0})
            assert kw_chm["x"] == 42
            assert kw_chm["y"] == [1, 2, 3]
            assert kw_chm["z", "w"] == 10.0
            ```
        """
        return ChoiceMap.d(kwargs)

    @staticmethod
    def switch(idx: int | IntArray, chms: Iterable["ChoiceMap"]) -> "ChoiceMap":
        """
        Creates a ChoiceMap that switches between multiple ChoiceMaps based on an index.

        This method creates a new ChoiceMap that selectively includes values from a sequence of
        input ChoiceMaps based on the provided index. The resulting ChoiceMap will contain
        values from the ChoiceMap at the position specified by the index, while masking out
        values from all other ChoiceMaps.

        Args:
            idx: An index or array of indices specifying which ChoiceMap(s) to select from.
            chms: An iterable of ChoiceMaps to switch between.

        Returns:
            A new ChoiceMap containing values from the selected ChoiceMap(s).

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm1 = ChoiceMap.d({"x": 1, "y": 2})
            chm2 = ChoiceMap.d({"x": 3, "y": 4})
            chm3 = ChoiceMap.d({"x": 5, "y": 6})

            switched = ChoiceMap.switch(jnp.array(1), [chm1, chm2, chm3])
            assert switched["x"].unmask() == 3
            assert switched["y"].unmask() == 4
            ```
        """
        return Switch.build(idx, chms)

    ######################
    # Combinator methods #
    ######################

    def mask(self, flag: Flag) -> "ChoiceMap":
        """
        Returns a new ChoiceMap with values masked by a boolean flag.

        This method creates a new ChoiceMap where the values are conditionally
        included based on the provided flag. If the flag is True, the original
        values are retained; if False, the ChoiceMap behaves as if it's empty.

        Args:
            flag: A boolean flag determining whether to include the values.

        Returns:
            A new ChoiceMap with values conditionally masked.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            masked_chm = original_chm.mask(True)
            assert masked_chm.get_value() == 42

            masked_chm = original_chm.mask(False)
            assert masked_chm.get_value() is None
            ```
        """
        return self.filter(flag)

    def extend(self, *addrs: AddressComponent) -> "ChoiceMap":
        """
        Returns a new ChoiceMap with the given address component as its root.

        This method creates a new ChoiceMap where the current ChoiceMap becomes a submap
        under the specified address component. It effectively adds a new level of hierarchy
        to the ChoiceMap structure.

        Args:
            addrs: The address components to use as the new root.

        Returns:
            A new ChoiceMap with the current ChoiceMap nested under the given address.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            original_chm = ChoiceMap.value(42)
            indexed_chm = original_chm.extend("x")
            assert indexed_chm["x"] == 42
            ```
        """
        acc = self
        for addr in reversed(addrs):
            if isinstance(addr, StaticAddressComponent):
                acc = Static.build({addr: acc})
            else:
                acc = Indexed.build(acc, addr)

        return acc

    def merge(self, other: "ChoiceMap") -> "ChoiceMap":
        """
        Merges this ChoiceMap with another ChoiceMap.

        This method combines the current ChoiceMap with another ChoiceMap using the XOR operation (^). It creates a new ChoiceMap that contains all addresses from both input ChoiceMaps; any overlapping addresses will trigger an error on access at the address via `[<addr>]` or `get_value()`. Use `|` if you don't want this behavior.

        Args:
            other: The ChoiceMap to merge with the current one.

        Returns:
            A new ChoiceMap resulting from the merge operation.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm1 = ChoiceMap.value(5).extend("x")
            chm2 = ChoiceMap.value(10).extend("y")
            merged_chm = chm1.merge(chm2)
            assert merged_chm["x"] == 5
            assert merged_chm["y"] == 10
            ```

        Note:
            This method is equivalent to using the | operator between two ChoiceMaps.
        """
        return self | other

    def get_selection(self) -> Selection:
        """
        Returns a Selection representing the structure of this ChoiceMap.

        This method creates a Selection that matches the hierarchical structure
        of the current ChoiceMap. The resulting Selection can be used to filter
        or query other ChoiceMaps with the same structure.

        Returns:
            A Selection object representing the structure of this ChoiceMap.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            chm = ChoiceMap.value(5).extend("x")
            sel = chm.get_selection()
            assert sel["x"] == True
            assert sel["y"] == False
            ```
        """
        return ChmSel.build(self)

    def static_is_empty(self) -> bool:
        """
        Returns True if this ChoiceMap is equal to `ChoiceMap.empty()`, False otherwise.
        """
        return False

    ###########
    # Dunders #
    ###########

    @nobeartype
    @deprecated(
        reason="^ is deprecated, please use | or _.merge(...) instead.",
        version="0.8.0",
    )
    def __xor__(self, other: "ChoiceMap") -> "ChoiceMap":
        return self | other

    def __or__(self, other: "ChoiceMap") -> "ChoiceMap":
        return Or.build(self, other)

    def __and__(self, other: "ChoiceMap") -> "ChoiceMap":
        return other.filter(self.get_selection())

    def __add__(self, other: "ChoiceMap") -> "ChoiceMap":
        return self | other

    def __call__(
        self,
        *addresses: Address,
    ) -> "ChoiceMap":
        """Alias for `get_submap(*addresses)`."""
        return self.get_submap(*addresses)

    def __getitem__(
        self,
        addr: Address,
    ):
        submap = self.get_submap(addr)
        v = submap.get_value()
        if v is None:
            raise ChoiceMapNoValueAtAddress(addr)
        else:
            return v

    def __contains__(
        self,
        addr: Address,
    ) -> bool:
        return self.get_submap(addr).has_value()

    @property
    def at(self) -> _ChoiceMapBuilder:
        """
        Returns a _ChoiceMapBuilder instance for constructing nested ChoiceMaps.

        This property allows for a fluent interface to build complex ChoiceMaps
        by chaining address components and setting values.

        Returns:
            A builder object for constructing ChoiceMaps.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            from genjax import ChoiceMap

            chm = ChoiceMap.d({("x", "y"): 3.0, "z": 12.0})
            updated = chm.at["x", "y"].set(4.0)

            assert updated["x", "y"] == 4.0
            assert updated["z"] == chm["z"]
            ```
        """
        return _ChoiceMapBuilder(self, [])

    @nobeartype
    @deprecated(
        reason="Acts as identity; filters are now automatically pushed down.",
        version="0.8.0",
    )
    def simplify(self) -> "ChoiceMap":
        """Previously pushed down filters, now acts as identity."""
        return self

    def invalid_subset(
        self,
        gen_fn: "genjax.GenerativeFunction[Any]",
        args: tuple[Any, ...],
    ) -> "ChoiceMap | None":
        """
        Identifies the subset of choices that are invalid for a given generative function and its arguments.

        This method checks if all choices in the current ChoiceMap are valid for the given
        generative function and its arguments.

        Args:
            gen_fn: The generative function to check against.
            args: The arguments to the generative function.

        Returns:
            A ChoiceMap containing any extra choices not reachable in the course of `gen_fn`'s execution, or None if no extra choices are found.

        Example:
            ```python exec="yes" html="true" source="material-block" session="choicemap"
            @genjax.gen
            def model(x):
                y = bernoulli(0.5) @ "y"
                return x + y


            chm = ChoiceMap.d({"y": 1, "z": 2})
            extras = chm.invalid_subset(model, (1,))
            assert "z" in extras  # "z" is an extra choice not in the model
            ```
        """
        shape_chm = gen_fn.get_zero_trace(*args).get_choices()
        shape_sel = _shape_selection(shape_chm)
        extras = self.filter(~shape_sel)
        if not extras.static_is_empty():
            return extras


@Pytree.dataclass(match_args=True)
class Choice(Generic[T], ChoiceMap):
    """Represents a choice map with a single value.

    This class represents a choice map that contains a single value at the root level.
    It is used to store individual choices in a hierarchical choice map structure.

    Attributes:
        v: The value stored in this choice map.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        value_chm = ChoiceMap.value(3.14)
        assert value_chm.get_value() == 3.14
        assert value_chm.get_submap("any_address").static_is_empty() == True
        ```
    """

    v: T

    @staticmethod
    def build(v: T) -> ChoiceMap:
        if isinstance(v, Array) and v.shape == (0,):
            return ChoiceMap.empty()
        elif isinstance(v, Mask):
            match v.primal_flag():
                case False:
                    return ChoiceMap.empty()
                case True:
                    return Choice(v.value)
                case _:
                    return Choice(v)
        else:
            return Choice(v)

    def filter(self, selection: Selection | Flag) -> ChoiceMap:
        if isinstance(selection, Selection):
            if selection.check():
                return self
            else:
                return ChoiceMap.empty()
        else:
            masked = Mask.build(self.v, selection)
            return Choice.build(masked)

    def get_value(self) -> T:
        return self.v

    def get_inner_map(self, addr: AddressComponent) -> ChoiceMap:
        if isinstance(addr, StaticAddressComponent):
            return ChoiceMap.empty()
        else:
            return jtu.tree_map(
                lambda v: v[addr], self, is_leaf=lambda x: isinstance(x, Mask)
            )


@Pytree.dataclass(match_args=True)
class Indexed(ChoiceMap):
    """Represents a choice map with dynamic indexing.

    This class represents a choice map that uses dynamic (array-based) addressing.
    It allows for indexing into the choice map using array-like address components.

    Attributes:
        c: The underlying choice map.
        addr: The dynamic address component used for indexing.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        import jax.numpy as jnp

        base_chm = ChoiceMap.value(jnp.array([1, 2, 3]))
        idx_chm = base_chm.extend(jnp.array([0, 1, 2]))

        assert idx_chm.get_submap(1).get_value() == genjax.Mask(2, True)
        ```
    """

    c: ChoiceMap
    addr: int | IntArray

    @staticmethod
    def build(chm: ChoiceMap, addr: DynamicAddressComponent) -> ChoiceMap:
        if chm.static_is_empty():
            return chm

        elif isinstance(addr, slice):
            if addr == _full_slice:
                return chm
            else:
                raise ValueError(f"Partial slices not supported: {addr}")

        elif isinstance(addr, Array) and addr.shape == (0,):
            return ChoiceMap.empty()

        else:
            return Indexed(chm, addr)

    def filter(self, selection: Selection | Flag) -> ChoiceMap:
        return self.c.filter(selection).extend(self.addr)

    def get_value(self) -> Any:
        return None

    def get_inner_map(self, addr: AddressComponent) -> ChoiceMap:
        if isinstance(addr, StaticAddressComponent):
            return ChoiceMap.empty()

        else:
            if not isinstance(addr, slice):
                # If we allowed non-scalar addresses, the `get_submap` call would not reduce the leaf by a dimension, and further get_submap calls would target the same dimension.
                assert not jnp.asarray(addr, copy=False).shape, (
                    "Only scalar dynamic addresses are supported by get_submap."
                )

            if isinstance(self.addr, Array) and self.addr.shape:
                # We can't allow slices, as self.addr might look like, e.g. `[2,5,6]`, and we don't have any way to combine this "sparse array selector" with an incoming slice.
                assert not isinstance(addr, slice), (
                    f"Slices are not allowed against array-shaped dynamic addresses. Tried to apply {addr} to {self.addr}."
                )

                check = self.addr == addr

                # If `check` contains a match (we know it will be a single match, since we constrain addr to be scalar), then `idx` is the index of the match in `self.addr`.
                # Else, idx == 0 (selecting "junk data" of the right shape at the leaf) and check_array[idx] == False (masking the junk data).
                idx = jnp.argwhere(check, size=1, fill_value=0)[0, 0]

                return jtu.tree_map(
                    lambda v: Mask.build(v[idx], check[idx]),
                    self.c,
                    is_leaf=lambda x: isinstance(x, Mask),
                )

            else:
                return self.c.mask(self.addr == addr)


@Pytree.dataclass(match_args=True)
class Static(ChoiceMap):
    """
    Represents a static choice map with a dictionary of address-choicemap pairs.

    This class implements a ChoiceMap where the addresses are static (non-dynamic)  components and the values are other ChoiceMaps. It provides an efficient way to  represent and manipulate hierarchical structures of choices.

    Attributes:
        mapping: A dictionary mapping static address components to ChoiceMaps.
    """

    mapping: dict[StaticAddressComponent, ChoiceMap | dict[StaticAddressComponent, Any]]

    @staticmethod
    def build(d: dict[StaticAddressComponent, ChoiceMap]) -> "Static":
        def unwrap(d: ChoiceMap) -> ChoiceMap | dict[StaticAddressComponent, Any]:
            if isinstance(d, Static):
                return d.mapping
            else:
                return d

        return Static(
            # Filter out empty choice maps
            {k: unwrap(v) for k, v in d.items() if not v.static_is_empty()}
        )

    @staticmethod
    def merge_with(
        merge: Callable[[ChoiceMap, ChoiceMap], ChoiceMap],
        c1: "Static",
        c2: "Static",
    ) -> ChoiceMap:
        """
        Returns a new ChoiceMap generated by merging two Static instances by applying a given merge function to values with overlapping keys and including non-overlapping kv-pairs from both.

        Args:
            merge: A function that defines how to merge two ChoiceMaps when they share the same key.
            c1: The first Static to merge.
            c2: The second Static to merge.

        Returns:
            ChoiceMap: A new ChoiceMap resulting from merging c1 and c2 using the
            provided merge function.
        """
        merged_dict = {}
        for key in set(c1.mapping.keys()) | set(c2.mapping.keys()):
            if key in c1.mapping and key in c2.mapping:
                merged_dict[key] = merge(c1.get_submap(key), c2.get_submap(key))
            elif key in c1.mapping:
                merged_dict[key] = c1.get_submap(key)
            else:
                merged_dict[key] = c2.get_submap(key)
        return Static.build(merged_dict)

    def filter(self, selection: Selection | Flag) -> ChoiceMap:
        def to_subsel(addr: StaticAddressComponent) -> Selection | Flag:
            if isinstance(selection, Selection):
                return selection(addr)
            else:
                return selection

        return Static.build({
            addr: self.get_submap(addr).filter(to_subsel(addr))
            for addr in self.mapping.keys()
        })

    def get_value(self) -> Any:
        return None

    def get_inner_map(self, addr: AddressComponent) -> ChoiceMap:
        if isinstance(addr, StaticAddressComponent):
            v = self.mapping.get(addr, {})
            return Static(v) if isinstance(v, dict) else v
        else:
            return jtu.tree_map(
                lambda v: v[addr], self, is_leaf=lambda x: isinstance(x, Mask)
            )

    def static_is_empty(self) -> bool:
        return len(self.mapping) == 0

    def __treescope_repr__(self, path, subtree_renderer):
        return trl.render_dictionary_wrapper(
            object_type=Static,
            wrapped_dict=self.mapping,
            path=path,
            subtree_renderer=subtree_renderer,
            roundtrippable=False,
            color=self.treescope_color(),
        )


@Pytree.dataclass(match_args=True)
class Switch(ChoiceMap):
    """Represents a choice map that selects between multiple choice maps based on an index.

    This class implements a switch statement over choice maps, where an index determines which
    choice map is active. When used with a concrete integer index, it directly returns the
    selected choice map. When used with a traced/symbolic index (IntArray), it returns a
    Switch instance that masks each choice map based on the index value.

    The get_submap method returns a new Switch instance with the submap operation applied to each
    choice map. The get_value method uses Mask.or_n to combine all non-None values from the choice
    maps. By design, only one branch will be masked as true, ensuring a single valid value.

    Attributes:
        idx: An integer or IntArray that determines which choice map is active.
        chms: A list of choice maps to select from.
    """

    idx: IntArray
    chms: list[ChoiceMap]

    @staticmethod
    def build(
        idx: int | IntArray,
        chm_iter: Iterable[ChoiceMap],
    ) -> ChoiceMap:
        if isinstance(idx, int):
            return list(chm_iter)[idx]
        else:
            chms = [_chm.mask(_idx == idx) for _idx, _chm in enumerate(chm_iter)]
            return Switch(idx, chms)

    def filter(self, selection: Selection | Flag) -> ChoiceMap:
        return Switch.build(self.idx, [chm.filter(selection) for chm in self.chms])

    def get_value(self) -> Any:
        vs = [chm.get_value() for chm in self.chms]
        entries = [Mask.build(v) for v in vs if v is not None]

        return Mask.or_n(*entries) if entries else None

    def get_inner_map(self, addr: AddressComponent) -> ChoiceMap:
        return Switch(self.idx, [chm.get_inner_map(addr) for chm in self.chms])


@Pytree.dataclass(match_args=True)
class Or(ChoiceMap):
    """Represents a choice map that combines two choice maps using an OR operation.

    This class combines two choice maps, prioritizing the first choice map (c1) over the second (c2)
    when there are overlapping addresses. It returns values from c1 if present, otherwise from c2.

    Attributes:
        c1: The first choice map (higher priority).
        c2: The second choice map (lower priority).

    Examples:
        ```python exec="yes" html="true" source="material-block" session="choicemap"
        chm1 = ChoiceMap.value(5)
        chm2 = ChoiceMap.value(10)
        or_chm = chm1 | chm2
        assert or_chm.get_value() == 5  # c1 takes priority

        chm3 = ChoiceMap.empty()
        chm4 = ChoiceMap.value(15)
        or_chm2 = chm3 | chm4
        assert or_chm2.get_value() == 15  # c2 used when c1 is empty
        ```
    """

    c1: ChoiceMap
    c2: ChoiceMap

    @staticmethod
    def build(
        c1: ChoiceMap,
        c2: ChoiceMap,
    ) -> ChoiceMap:
        if c2.static_is_empty():
            return c1
        elif c1.static_is_empty():
            return c2
        else:
            match (c1, c2):
                case (Static(), Static()):
                    return Static.merge_with(or_, c1, c2)

                case (Choice(a), Choice(b)):
                    a = Mask.build(a)
                    b = Mask.build(b)
                    return Choice.build(a | b)

                case (Switch(), Switch()):
                    raise Exception(
                        f"We can't currently handle two switches in an Or: {c1}, {c2}"
                    )

                case (Switch(idx, chms), _):
                    return Switch.build(idx, [c1 | c2 for c1 in chms])

                case (_, Switch(idx, chms)):
                    return Switch.build(idx, [c1 | c2 for c2 in chms])

                case (Choice(), _) | (_, Choice()):
                    raise Exception(f"Choice and non-Choice in Or: {c1}, {c2}")

                case _:
                    return Or(c1, c2)

    def filter(self, selection: Selection | Flag) -> ChoiceMap:
        return self.c1.filter(selection) | self.c2.filter(selection)

    def get_value(self) -> Any:
        return None

    def get_inner_map(self, addr: AddressComponent) -> ChoiceMap:
        submap1 = self.c1.get_inner_map(addr)
        submap2 = self.c2.get_inner_map(addr)
        return submap1 | submap2


def _shape_selection(chm: ChoiceMap) -> Selection:
    def loop(inner: ChoiceMap, selection: Selection) -> Selection:
        match inner:
            case Static(mapping):
                acc = Selection.none()
                for addr in mapping.keys():
                    sub_chm = inner.get_submap(addr)
                    sub_sel = selection(addr)
                    acc |= loop(sub_chm, sub_sel).extend(addr)
                return acc

            case Indexed(c, addr):
                return loop(c, selection).extend(...)

            case Choice():
                return LeafSel()

            case Or(c1, c2):
                return loop(c1, selection) | loop(c2, selection)

            case Switch(_, chms):
                head, tail = chms[0], chms[1:]
                acc = loop(head, selection)
                for chm in tail:
                    acc |= loop(chm, selection)

                return acc

            case _:
                raise ValueError(f"Unknown ChoiceMap type: {type(inner)}")

    return loop(chm, Selection.all())


_empty = Static({})
ChoiceMapBuilder = _ChoiceMapBuilder(_empty, [])

################################
# Choice map specialized types #
################################
