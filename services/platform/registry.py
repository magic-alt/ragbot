from __future__ import annotations

from importlib import metadata
from threading import RLock
from typing import Generic, Iterable, Optional, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class RegistrableComponent(Protocol):
    @property
    def component_id(self) -> str: ...


T = TypeVar("T", bound=RegistrableComponent)


class TypedRegistry(Generic[T]):
    """Small deterministic registry used as Ragbot's platform extension boundary.

    Built-ins are registered explicitly by Ragbot. Optional third-party packages
    can expose immutable component specifications through Python entry points.
    The registry only loads code during process bootstrap/discovery; request data
    can never select an arbitrary import target.
    """

    def __init__(self, *, kind: str, entrypoint_group: Optional[str] = None) -> None:
        self.kind = str(kind).strip().lower()
        self.entrypoint_group = entrypoint_group
        self._items: dict[str, T] = {}
        self._entrypoints_loaded = False
        self._lock = RLock()

    def register(self, component: T, *, replace: bool = False) -> T:
        component_id = str(component.component_id).strip().lower()
        if not component_id:
            raise ValueError(f"{self.kind} component_id must be non-empty")
        with self._lock:
            if component_id in self._items and not replace:
                raise ValueError(f"Duplicate {self.kind} component registration: {component_id}")
            self._items[component_id] = component
        return component

    def get(self, component_id: str) -> T:
        key = str(component_id).strip().lower()
        self.load_entry_points()
        try:
            return self._items[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {self.kind} component {component_id!r}; "
                f"available={sorted(self._items)}"
            ) from exc

    def values(self) -> tuple[T, ...]:
        self.load_entry_points()
        return tuple(self._items[key] for key in sorted(self._items))

    def ids(self) -> tuple[str, ...]:
        self.load_entry_points()
        return tuple(sorted(self._items))

    def load_entry_points(self) -> None:
        group = self.entrypoint_group
        if not group:
            return
        with self._lock:
            if self._entrypoints_loaded:
                return
            self._entrypoints_loaded = True
            discovered = metadata.entry_points()
            selected: Iterable[metadata.EntryPoint]
            if hasattr(discovered, "select"):
                selected = discovered.select(group=group)
            else:  # pragma: no cover - Python <3.10 compatibility shape
                selected = discovered.get(group, ())
            for entrypoint in sorted(selected, key=lambda item: (item.name, item.value)):
                loaded = entrypoint.load()
                component = loaded() if callable(loaded) and not hasattr(loaded, "component_id") else loaded
                if not isinstance(component, RegistrableComponent):
                    raise TypeError(
                        f"Entry point {entrypoint.name!r} in {group!r} did not return a registrable component"
                    )
                self.register(component)
