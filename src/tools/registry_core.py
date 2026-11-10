# src/tools/registry_core.py
REGISTRY = {}

def tool(args, slow=False, name=None):
    """Register a function as an agent tool."""
    def deco(fn):
        key = name or fn.__name__
        if key in REGISTRY:
            raise RuntimeError(f"Duplicate tool name: {key}")
        REGISTRY[key] = dict(fn=fn, args=args, slow=slow)
        return fn
    return deco
