import openmm as mm


def get_fastest_platform() -> mm.Platform:
    """Returns the fastest available OpenMM platform."""
    preferred_platforms = ["CUDA", "CPU", "Reference"]  # Ordered by speed, we ignore OpenCL because we don't have that
    for name in preferred_platforms:
        try:
            return mm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No valid OpenMM platform found!")
