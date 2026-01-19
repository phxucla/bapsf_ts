import acquire_Nshots
import epics
from p4p.client.thread import Context

RED = "\033[1;31m"
RESET = "\033[0m"

def check_ca_connected(pvname: str, timeout: float = 5.0) -> bool:
    pv = epics.PV(pvname)
    pv.wait_for_connection(timeout=timeout)
    return bool(pv.connected)

def get_pva_image_dims(ctxt: Context, pvname: str, timeout: float = 2.0):
    """
    Robustly try to read image dimensions from a PVA PV.

    Strategy:
      1) Try requesting field(dimension) and read dimension[0].size, dimension[1].size (NTNDArray-like)
      2) Fallback to requesting common alternatives: field(sizes), field(shape), field(width,height)
      3) If the PV is just a numpy array, infer dims from its shape when possible
    """
    # 1) NTNDArray-like: dimension[].size
    try:
        v = ctxt.get(pvname, request="field(dimension)", timeout=timeout)
        dim = v["dimension"]
        w = int(dim[0]["size"])
        h = int(dim[1]["size"])
        return w, h
    except Exception:
        pass

    # 2a) Sometimes sizes exists
    try:
        v = ctxt.get(pvname, request="field(sizes)", timeout=timeout)
        sizes = v["sizes"]
        # common ordering: [width, height, ...]
        if len(sizes) >= 2:
            return int(sizes[0]), int(sizes[1])
    except Exception:
        pass

    # 2b) Sometimes shape exists
    try:
        v = ctxt.get(pvname, request="field(shape)", timeout=timeout)
        shape = v["shape"]
        if len(shape) >= 2:
            # could be [height, width] or [width, height]; try to guess
            a, b = int(shape[0]), int(shape[1])
            # heuristic: width often >= height for typical images, but not guaranteed
            return (b, a) if a <= b else (a, b)
    except Exception:
        pass

    # 2c) width/height fields
    try:
        v = ctxt.get(pvname, request="field(width,height)", timeout=timeout)
        return int(v["width"]), int(v["height"])
    except Exception:
        pass

    # 3) PV might just be the pixel array (numpy-like) - try inferring from .shape
    v = ctxt.get(pvname, timeout=timeout)
    # If v is a numpy array, it will have .shape
    if hasattr(v, "shape") and len(getattr(v, "shape", ())) >= 2:
        # numpy shape is typically (height, width)
        h, w = int(v.shape[0]), int(v.shape[1])
        return w, h

    raise RuntimeError("Could not determine image dimensions (no dimension/sizes/shape/width/height fields and not array-like).")

if __name__ == "__main__":
    # One shared PVA context
    ctxt = Context("pva")

    # SCALARS (CA)
    print("SCALARS\n=======")
    for name in acquire_Nshots.scalars:
        if check_ca_connected(name):
            print(f"{name} is connected.")
        else:
            print(f"{RED}{name} is not connected.{RESET}")

    # ARRAYS (CA)
    print("\nARRAYS\n======")
    for name in acquire_Nshots.arrays:
        if check_ca_connected(name):
            print(f"{name} is connected.")
        else:
            print(f"{RED}{name} is not connected.{RESET}")

    # IMAGES (PVA)
    print("\nIMAGES\n======")
    for name in acquire_Nshots.images:
        try:
            w, h = get_pva_image_dims(ctxt, name)
            print(f"{name}: ({w},{h})")
        except Exception as e:
            print(f"{RED}{name}: Error - {e}{RESET}")

