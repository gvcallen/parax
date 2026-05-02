from typing import Any, Tuple, List, Type
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, fields

import jax
from jaxtyping import PyTree
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey
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


def path_to_name(path: list, separator: str = '.') -> str:
    """Convert a PyTree path to a fully-qualified name."""
    name_fields = []

    for item in path:
        if isinstance(item, GetAttrKey):
            k = item.name
        elif isinstance(item, DictKey):
            k = item.key
        elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
            k = item.idx if hasattr(item, 'idx') else item.key
        else:
            raise Exception(f"Unsupported key type in path: {type(item)}")
        name_fields.append(str(k))
            
    return separator.join(name_fields)
    

def path_to_pseudoname(tree: PyTree, path: list, separator: str = '_') -> str:
    """Convert a PyTree path to a fully-qualified name."""
    name_fields = []
    node = tree

    for item in path:
        if isinstance(item, GetAttrKey):
            k = item.name
            next_node = getattr(node, k)
            
            # 1. Determine transparency
            is_transparent = getattr(node, '_transparent', False)
            if not is_transparent and is_dataclass(node):
                field_obj = next((f for f in fields(node) if f.name == k), None)
                if field_obj is not None:
                    is_transparent = field_obj.metadata.get('transparent', False)

            # 2. Extract user override
            explicit_name = getattr(next_node, 'name', None)
            
            # 3. Rule application
            if is_transparent:
                if explicit_name is not None:
                    name_fields.append(explicit_name)
            else:
                name_fields.append(explicit_name if explicit_name is not None else k)
                    
            node = next_node
            
        elif isinstance(item, DictKey):
            k = item.key
            node = node[k]
            name_fields.append(str(k))
                
        elif isinstance(item, (SequenceKey, FlattenedIndexKey)):
            idx = item.idx if hasattr(item, 'idx') else item.key
            node = node[idx]
            explicit_name = getattr(node, 'name', None)
            if explicit_name is not None:
                name_fields.append(explicit_name)
            else:
                name_fields.append(str(idx))
                
        else:
            raise Exception(f"Unsupported key type in path: {type(item)}")
            
    return separator.join(name_fields)
    