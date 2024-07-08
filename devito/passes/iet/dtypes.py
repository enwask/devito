import numpy as np

from devito.ir import FindSymbols, Uxreplace

__all__ = ['lower_dtypes']

def lower_dtypes(iet, lang, compiler):
    """
    Add headers for complex arithmetic and lower language-specific dtypes
    """
    # Check for complex numbers that always take dtype precedence
    types = {f.dtype for f in FindSymbols().visit(iet) 
             if issubclass(f.dtype, np.generic)}

    metadata = {}
    if any(np.issubdtype(d, np.complexfloating) for d in types):
        metadata = _complex_includes(lang, compiler)

    # Map dtypes to language-specific types
    mapper = {}
    for s in FindSymbols('indexeds|symbolics').visit(iet):
        if s.dtype in lang['types']:
            mapper[s] = s._rebuild(dtype=lang['types'][s.dtype])

    body = Uxreplace(mapper).visit(iet.body)
    params = Uxreplace(mapper).visit(iet.parameters)
    iet = iet._rebuild(body=body, parameters=params)

    return iet, metadata


def _complex_includes(lang, compiler):
    """
    Add headers for complex arithmetic
    """
    lib = (lang['header-complex'],)

    metadata = {}
    if lang.get('complex-namespace') is not None:
        metadata['namespaces'] = lang['complex-namespace']

    # Some languges such as c++11 need some extra arithmetic definitions
    if lang.get('def-complex'):
        dest = compiler.get_jit_dir()
        hfile = dest.joinpath('complex_arith.h')
        with open(str(hfile), 'w') as ff:
            ff.write(str(lang['def-complex']))
        lib += (str(hfile),)

    metadata['includes'] = lib
    return metadata
