::: genjax._src.generative_functions.static
    options:
      show_root_heading: false

## Usage

The `Static` language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions. 

Below, we illustrate a simple example:
    
```python
from genjax import beta 
from genjax import bernoulli 
from genjax import uniform 
from genjax import gen

@genjax.gen
def beta_bernoulli_process(u):
    p = beta(0, u) @ "p"
    v = bernoulli(p) @ "v"
    return v

@genjax.gen
def joint():
    u = uniform() @ "u"
    v = beta_bernoulli_process(u) @ "bbp"
    return v
```

## Language primitives

The static language exposes custom primitives, which are handled by JAX interpreters to support the semantics of the generative function interface.

### `trace`

The `trace` primitive provides access to the ability to invoke another generative function as a callee. 

::: genjax.generative_functions.static.trace

Returning to our example above:


```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta 
from genjax import bernoulli 
from genjax import gen

@gen
def beta_bernoulli_process(u):
    # Invoking `trace` can be sweetened, or unsweetened.
    p = genjax.trace("p", beta)(0, u) # not sweet
    v = bernoulli(p) @ "v" # sweet
    return v
```

Now, programs written in the DSL which utilize `trace` have generative function interface method implementations which store callee choice data in the trace:

```python exec="yes" source="tabbed-left" session="ex-trace"
import jax
console = genjax.pretty()

key = jax.random.PRNGKey(314159)
tr = beta_bernoulli_process.simulate(key, (2, ))

print(console.render(tr))
```

Notice how the rendered result `Trace` has addresses in its choice trie for `"p"` and `"v"` - corresponding to the invocation of the beta and Bernoulli distribution generative functions.

The `trace` primitive is a critical element of structuring hierarchical generative computation in the static language.

### `cache`

The `cache` primitive is designed to expose a space vs. time trade-off for incremental computation in Gen's `update` interface.

::: genjax.generative_functions.static.cache
