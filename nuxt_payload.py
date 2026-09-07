import json

_WRAPPER_TAGS = {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "EmptyRef", "EmptyShallowRef"}


def resolve_payload(raw_json_text):
    """Deserialize a Nuxt 3 SSR payload (the devalue-style flat reference
    array served at e.g. /leaderboard/_payload.json) into a plain dict.

    The array's slot 0 is always {"data": <ref>, "prerenderedAt": ...}.
    `data` resolves to a dict keyed by an opaque per-build hash whose sole
    value is the actual page data we want, so we unwrap that automatically.
    """
    arr = json.loads(raw_json_text)
    cache = {}
    root = _resolve(0, arr, cache)
    data = root["data"]
    if isinstance(data, dict) and len(data) == 1:
        return next(iter(data.values()))
    return data


def _resolve(idx, arr, cache):
    if idx in cache:
        return cache[idx]
    val = arr[idx]
    if isinstance(val, list):
        if len(val) == 2 and isinstance(val[0], str) and val[0] in _WRAPPER_TAGS:
            result = _resolve(val[1], arr, cache)
        elif len(val) == 2 and isinstance(val[0], str) and val[0] == "Date":
            result = val[1]
        else:
            result = [None] * len(val)
            cache[idx] = result
            for i, ref in enumerate(val):
                result[i] = _resolve(ref, arr, cache) if isinstance(ref, int) else ref
            return result
    elif isinstance(val, dict):
        result = {}
        cache[idx] = result
        for k, ref in val.items():
            result[k] = _resolve(ref, arr, cache) if isinstance(ref, int) else ref
        return result
    else:
        result = val
    cache[idx] = result
    return result
