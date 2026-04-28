from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from tqdm.auto import tqdm


class ProgressTracker(Protocol):
    def advance(self, step: int = 1, *, detail: str | None = None) -> None: ...

    def close(self, *, detail: str | None = None) -> None: ...


class ProgressSink(Protocol):
    def stage(
        self,
        label: str,
        *,
        total: int | None = None,
        unit: str = "item",
    ) -> ProgressTracker: ...


class NullProgressTracker:
    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        del step, detail

    def close(self, *, detail: str | None = None) -> None:
        del detail


class NullProgressSink:
    def stage(
        self,
        label: str,
        *,
        total: int | None = None,
        unit: str = "item",
    ) -> ProgressTracker:
        del label, total, unit
        return NullProgressTracker()


class _TqdmProgressTracker:
    def __init__(
        self,
        *,
        bar: Any,
        position: int,
        release_position: Callable[[int], None],
    ) -> None:
        self._bar = bar
        self._position = position
        self._release_position = release_position
        self._closed = False

    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        if detail:
            self._bar.set_postfix_str(detail, refresh=False)
        if step:
            self._bar.update(step)
        elif detail:
            self._bar.refresh()

    def close(self, *, detail: str | None = None) -> None:
        if self._closed:
            return
        if detail:
            self._bar.set_postfix_str(detail, refresh=False)
        self._bar.close()
        self._release_position(self._position)
        self._closed = True


class TqdmProgressSink:
    def __init__(self, *, leave: bool = True) -> None:
        self._leave = leave
        self._next_position = 0
        self._free_positions: list[int] = []

    def stage(
        self,
        label: str,
        *,
        total: int | None = None,
        unit: str = "item",
    ) -> ProgressTracker:
        position = self._allocate_position()
        return _TqdmProgressTracker(
            bar=tqdm(
                total=total,
                desc=label,
                unit=unit,
                leave=self._leave,
                dynamic_ncols=True,
                position=position,
            ),
            position=position,
            release_position=self._release_position,
        )

    def _allocate_position(self) -> int:
        if self._free_positions:
            return self._free_positions.pop(0)
        position = self._next_position
        self._next_position += 1
        return position

    def _release_position(self, position: int) -> None:
        self._free_positions.append(position)
        self._free_positions.sort()


def make_progress_sink(*, show_progress: bool) -> ProgressSink:
    if show_progress:
        return TqdmProgressSink()
    return NullProgressSink()
