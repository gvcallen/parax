import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.distribution import Distribution

from parax.field import field
from parax.parameter import Parameter

@dataclass
class ParameterGroup:
    r"""
    A metadata class that groups a set of named flat parameters and defines any relationships between them.

    Attributes
    ----------
    parameter_names : list[str]
        The names of the parameters included in this group.
    distribution : dist.Distribution or None
        An optional joint distribution over the flattened parameters.
    """
    param_names: list[str]
    distribution: dist.Distribution | None = field(default=None)
    
    def __init__(self, param_names: list[str] | dict[str, Parameter], distribution: dist.Distribution | None = None):
        r"""
        Construct a :class:`ParameterGroup`.

        Parameters
        ----------
        param_names : list[str] | dict[str, Parameter]
            The names of the flattened parameters (or a mapping to parameters).
        dist : numpyro.distributions.Distribution, optional
            An optional joint distribution over the flattened parameters.
        """
        self.param_names = param_names
        self.distribution = distribution
        
    @property
    def num_params(self):
        r"""
        Number of flattened parameters in the group.

        Returns
        -------
        int
            The count of names in ``parameter_names``.
        """
        return len(self.param_names)        
    
    def with_distribution(self, distribution: Distribution) -> 'Parameter':
        r"""
        Return a copy of the parameter group with a new distribution.

        Parameters
        ----------
        distribution : numpyro.distributions.Distribution
            The distribution to associate with this parameter.

        Returns
        -------
        Parameter
            A copy of this object with ``distribution`` replaced.

        Raises
        ------
        Exception
            If ``dist`` is not a numpyro Distribution.
        """
        if not isinstance(distribution, Distribution):
            raise Exception('Only numpyro distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)