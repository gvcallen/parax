from typing import Union, get_origin
from types import GenericAlias, UnionType

def is_convertible_to_float(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False         
           
def get_first_underlying_type(tp: type) -> type | None:
    # The annotations could be unions - in this case we just take the first one TODO upgrade this to do more in-depth inspection?
    if isinstance(tp, UnionType):
        return get_first_underlying_type(tp.__args__[0])
    if isinstance(tp, (type,)) and not isinstance(tp, (GenericAlias, UnionType)):
        return tp

    origin = get_origin(tp)
    if origin is None:
        return None
    if origin is Union:
        return None
    return get_first_underlying_type(origin)