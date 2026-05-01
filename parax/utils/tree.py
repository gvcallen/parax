from typing import Any, Tuple, List, Type
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, fields

import jax
from collections import defaultdict
from typing import Callable, Any

def nodes_by_type(tree: Any, match_type: Type) -> List[Tuple[Tuple[Any, ...], Any]]:
    """
    Recursively search a PyTree for nodes matching a specific type.

    Parameters
    ----------
    tree : Any
        The PyTree to search.
    match_type : Type
        The class type to match.

    Returns
    -------
    List[Any]
        A list of matching node instances.
    """
    matches = []

    if isinstance(tree, match_type):
        matches.append(tree)

    # Handle dataclasses
    if is_dataclass(tree) and not isinstance(tree, type):
        for f in fields(tree):
            value = getattr(tree, f.name)
            matches.extend(nodes_by_type(value, match_type))

    # Handle dicts
    elif isinstance(tree, Mapping):
        for k, v in tree.items():
            matches.extend(nodes_by_type(v, match_type))

    # Handle lists, tuples, etc.
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for i, v in enumerate(tree):
            matches.extend(nodes_by_type(v, match_type))

    return matches

def tree_map_grouped(
    f: Callable, 
    *trees: Any, 
    group_tree: Any, 
    is_leaf: Callable | None = None
) -> Any:
    """
    Like jax.tree.map, but applies `f` to groups of leaves simultaneously.
    
    `group_tree` must be a PyTree structurally identical to `trees`. 
    Leaves in `trees` that share the same value in `group_tree` are gathered 
    into a list and passed to `f` together.
    """
    # Flatten the group tree to get our identifiers
    group_leaves, treedef = jax.tree_util.tree_flatten(group_tree, is_leaf=is_leaf)
    
    # Flatten all target trees
    flat_trees = [jax.tree_util.tree_flatten(t, is_leaf=is_leaf)[0] for t in trees]
    
    # Map group identifiers to their flat list indices
    group_indices = defaultdict(list)
    for i, group_id in enumerate(group_leaves):
        # We only group if group_id is not None. None implies independent/skip.
        if group_id is not None:
            group_indices[group_id].append(i)
        
    # Prepare an output list matching the flat leaves
    out_leaves = [None] * len(group_leaves)
    
    # Apply f to each group
    for group_id, indices in group_indices.items():
        # Gather the arguments for this group across all trees
        group_args = [
            [flat_tree[i] for i in indices] for flat_tree in flat_trees
        ]
        
        # f signature: f(group_id, tree1_group, tree2_group, ...)
        # It must return a tuple/list of results matching the group size
        results = f(group_id, *group_args)
        
        # Scatter the results back into the flat output list
        for i, res in zip(indices, results):
            out_leaves[i] = res
            
    # Reconstruct the original PyTree
    return jax.tree_util.tree_unflatten(treedef, out_leaves)