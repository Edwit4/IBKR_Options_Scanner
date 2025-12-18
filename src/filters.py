FILTER_VERBOSE = True


def set_filter_verbose(flag: bool):
    global FILTER_VERBOSE
    FILTER_VERBOSE = bool(flag)


def log_filter(reason, symbol=None, expiry=None, spread=None):
    if not FILTER_VERBOSE:
        return
    ctx_parts = []
    if symbol:
        ctx_parts.append(str(symbol))
    if expiry:
        ctx_parts.append(str(expiry))
    if spread:
        ctx_parts.append(str(spread))
    ctx = " | ".join(ctx_parts)
    if ctx:
        print(f"  Filter[{ctx}]: {reason}")
    else:
        print(f"  Filter: {reason}")

def log_pass(msg, symbol=None, expiry=None, spread=None):
    if not FILTER_VERBOSE:
        return
    ctx_parts = []
    if symbol:
        ctx_parts.append(str(symbol))
    if expiry:
        ctx_parts.append(str(expiry))
    if spread:
        ctx_parts.append(str(spread))
    ctx = " | ".join(ctx_parts)
    if ctx:
        print(f"  Pass[{ctx}]: {msg}")
    else:
        print(f"  Pass: {msg}")

