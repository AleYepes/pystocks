from typing import Any

from tqdm.auto import tqdm


class _NullProgressBar:
    def update(self, n=1):
        return None

    def set_postfix_str(self, s="", refresh=True):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def track_progress(
    iterable,
    *,
    show_progress=False,
    total=None,
    desc=None,
    unit=None,
    leave=True,
):
    if not show_progress:
        return iterable
    kwargs: dict[str, Any] = {"leave": leave, "dynamic_ncols": True}
    if total is not None:
        kwargs["total"] = total
    if desc is not None:
        kwargs["desc"] = desc
    if unit is not None:
        kwargs["unit"] = unit
    return tqdm(iterable, **kwargs)


def make_progress_bar(
    *,
    show_progress=False,
    total=None,
    desc=None,
    unit=None,
    leave=True,
):
    if not show_progress:
        return _NullProgressBar()
    kwargs: dict[str, Any] = {"leave": leave, "dynamic_ncols": True}
    if total is not None:
        kwargs["total"] = total
    if desc is not None:
        kwargs["desc"] = desc
    if unit is not None:
        kwargs["unit"] = unit
    return tqdm(**kwargs)
