import sympy as sp
class _SymLib:  # numpy-like facade over sympy so one preprocessing definition works numerically and symbolically
    log = staticmethod(sp.log); exp = staticmethod(sp.exp); sin = staticmethod(sp.sin); cos = staticmethod(sp.cos); sqrt = staticmethod(sp.sqrt); pi = sp.pi
    log1p = staticmethod(lambda x: sp.log(1 + x)); maximum = staticmethod(sp.Max)
