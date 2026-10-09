import pathlib
import yaml

_CFG = yaml.safe_load((pathlib.Path(__file__).parents[2] / "configs" / "labels.yaml").read_text())
UNIFIED = set(_CFG["unified"])
IGNORE = "IGNORE"


def map_label(source, label, identifier_type=None):
    m = _CFG["maps"][source]
    if label not in m:
        raise KeyError(f"unmapped {source} label: {label!r}")
    v = m[label]
    if isinstance(v, dict):
        if identifier_type not in v:
            raise KeyError(f"unmapped {source} label: {label!r}/{identifier_type!r}")
        v = v[identifier_type]
    if v != IGNORE and v not in UNIFIED:
        raise KeyError(f"{source} {label!r} maps to non-unified label {v!r}")
    return v
