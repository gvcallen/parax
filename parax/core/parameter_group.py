import dataclasses
from dataclasses import dataclass

from distreqx.distributions import AbstractDistribution
from distreqx.bijectors import AbstractBijector, Chain
from parax.core.field import field

@dataclass
class ParameterGroup:
    """
    A metadata class that groups a set of named flat parameters and defines 
    any joint relationships, distributions, or transforms between them.

    """
    #: The names of the parameters included in this group.
    param_names: list[str]
    #: An optional identifier for the group itself (e.g., 'covariance_matrix').
    name: str | None = None
    #: An optional joint distribution over the grouped parameters.
    distribution: AbstractDistribution | None = None
    #: An optional joint bijector applied to the grouped parameters.
    bijector: AbstractBijector | None = None
    #: Arbitrary user-defined metadata associated with the group.
    info: dict = field(default_factory=dict, static=True)
    
    @property
    def num_params(self) -> int:
        """
        Number of flattened parameters in the group.

        Returns
        -------
        int
            The count of names in `param_names`.
        """
        return len(self.param_names)        
    
    def with_distribution(self, distribution: AbstractDistribution) -> 'ParameterGroup':
        """
        Return a copy of the parameter group with a new joint distribution.

        Parameters
        ----------
        distribution : distreqx.distributions.AbstractDistribution
            The joint distribution to associate with this parameter group.

        Returns
        -------
        ParameterGroup
            A copy of this object with `distribution` replaced.

        Raises
        ------
        TypeError
            If `distribution` is not a distreqx AbstractDistribution.
        """
        if not isinstance(distribution, AbstractDistribution):
            raise TypeError('Only distreqx distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)
    
    def transformed(self, bijector: AbstractBijector) -> 'ParameterGroup':
        """
        Return a copy of this parameter group transformed by an additional joint bijector.

        Chains the new bijector with any existing group-level bijector.

        Parameters
        ----------
        bijector : distreqx.bijectors.AbstractBijector
            The bijector to apply to the group.

        Returns
        -------
        ParameterGroup
            A dynamically transformed ParameterGroup object.
            
        Raises
        ------
        TypeError
            If the provided bijector is not a distreqx AbstractBijector.
        """
        if not isinstance(bijector, AbstractBijector):
            raise TypeError("The provided transformation must be a distreqx AbstractBijector.")

        new_bij = self.bijector
        if new_bij is not None:
            new_bij = Chain([bijector, new_bij])
        else:
            new_bij = bijector
            
        return dataclasses.replace(self, bijector=new_bij)