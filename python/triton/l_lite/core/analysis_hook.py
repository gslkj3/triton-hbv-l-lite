"""Optional external adviser registration. No predictor or coefficients here."""
_advisers = {}


def register_adviser(identity, callback):
    if not isinstance(identity, str) or not identity or not callable(callback):
        raise ValueError('adviser requires immutable identity and callable')
    if identity in _advisers and _advisers[identity] is not callback:
        raise ValueError('adviser identity already registered; use a new configuration identity')
    _advisers[identity] = callback


def advise(module, metadata, options, capability):
    callback = _advisers.get(options.l_analysis_ref)
    if callback is None:
        raise ValueError('prediction requested without registered external adviser')
    return callback(module, metadata, options, capability)


def mapped_launcher(launcher, original_grid, divisors):
    original_grid, divisors = tuple(original_grid), tuple(divisors)
    if (len(original_grid) != 3 or len(divisors) != 3 or
            any(type(g) is not int or type(d) is not int or g < 1 or d < 1 or g % d
                for g, d in zip(original_grid, divisors))):
        raise ValueError('invalid prediction launch mapping')
    def launch(x, y, z, *args, **kwargs):
        if (x, y, z) != original_grid:
            raise ValueError('compiled prediction belongs to a different original grid')
        return launcher(*(g//d for g, d in zip(original_grid, divisors)), *args, **kwargs)
    return launch
