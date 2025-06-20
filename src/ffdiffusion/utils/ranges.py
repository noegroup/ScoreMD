import os
import logging

log = logging.getLogger(__name__)

eps = 1e-2


def merge_intervals(intervals):
    # Sort intervals by their start value.
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []

    for interval in intervals:
        if not merged:
            merged.append(interval)
        else:
            last_start, last_end = merged[-1]
            curr_start, curr_end = interval
            # Merge intervals if they overlap or touch.
            if curr_start <= last_end + eps:
                merged[-1] = (last_start, max(last_end, curr_end))
            else:
                merged.append(interval)
    return merged


def union_contains_range(intervals, target_range):
    """
    Checks if the union of the given intervals covers the target_range.

    Parameters:
      intervals: List of tuples (start, end) for the intervals.
      target_range: A tuple (target_start, target_end).

    Returns:
      True if the union of intervals covers target_range, False otherwise.
    """
    merged = merge_intervals(intervals)
    target_start, target_end = target_range
    # Check if there exists an interval in the merged list that covers the target range.
    for start, end in merged:
        if start <= target_start + eps and end >= target_end - eps:
            return True
    return False


def assert_range_continuity(ranges):
    for r in ranges:
        assert 1.0 >= r[0] > r[1] >= 0, f"Ranges must be decreasing and between 0 and 1 {r}"

    # Check if we should skip range continuity check
    if os.environ.get("NO_RANGE_CHECK"):
        log.info("Skipping range continuity check")
        return

    # swap ranges so that it matches the definition of union_contains_range
    ranges_swapped = [(b, a) for a, b in ranges]
    if not union_contains_range(ranges_swapped, (1e-2, 1)):
        raise ValueError("Ranges must cover the entire range (0, 1). Disable this check by setting NO_RANGE_CHECK=1")
