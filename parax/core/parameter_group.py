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

    Attributes
    ----------
    param_names : list of str
        The names of the parameters included in this group.
    name : str or None, optional
        An optional identifier for the group itself (e.g., 'covariance_matrix').
    distribution : AbstractDistribution or None, optional
        An optional joint distribution over the grouped parameters.
    transform : AbstractBijector or None, optional
        An optional joint transform applied to the grouped parameters.
        This is provided for future compatibility and is not yet used.
    info : dict
        Arbitrary user-defined metadata associated with the group. Marked as static.
    """
    param_names: list[str]
    name: str | None = None
    distribution: AbstractDistribution | None = None
    transform: AbstractBijector | None = None
    info: dict = field(default_factory=dict, static=True)
    
    @property
    def num_params(self) -> int:
        """
        Get the number of flattened parameters in the group.

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
            A copy of this object with the `distribution` replaced.

        Raises
        ------
        TypeError
            If `distribution` is not a distreqx AbstractDistribution.
        """
        if not isinstance(distribution, AbstractDistribution):
            raise TypeError('Only distreqx distributions are supported as parameter distributions')
        
        return dataclasses.replace(self, distribution=distribution)
    
    def transformed(self, transform: AbstractBijector) -> 'ParameterGroup':
        """
        Return a copy of this parameter group transformed by an additional joint bijector.

        This method chains the new bijector with any existing group-level bijector,
        applying the transformations sequentially.

        Parameters
        ----------
        transform : distreqx.bijectors.AbstractBijector
            The transform to apply to the group.

        Returns
        -------
        ParameterGroup
            A dynamically transformed ParameterGroup object.
            
        Raises
        ------
        TypeError
            If the provided bijector is not a distreqx AbstractBijector.
        """
        if not isinstance(transform, AbstractBijector):
            raise TypeError("The provided transformation must be a distreqx AbstractBijector.")

        new_transform = self.transform
        if new_transform is not None:
            new_transform = Chain([transform, new_transform])
        else:
            new_transform = transform
            
        return dataclasses.replace(self, bijector=new_transform)