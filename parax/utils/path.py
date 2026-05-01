from dataclasses import is_dataclass, fields

from jaxtyping import PyTree
from jax.tree_util import GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey

def path_to_name(tree: PyTree, path: list, separator: str = '_') -> str:
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
    