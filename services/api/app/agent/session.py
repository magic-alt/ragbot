"""In-memory session store for multi-turn conversation history.

Production deployments should replace ``InMemorySessionStore`` with a
Redis- or database-backed implementation that conforms to the
``SessionStore`` protocol.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class SessionTurn:
    query: str
    answer: str
    confidence: str
    request_id: str


@dataclass
class Session:
    session_id: str
    tenant_id: str
    user_id: str
    turns: List[SessionTurn] = field(default_factory=list)


class SessionStore(Protocol):
    def load(self, session_id: str) -> Optional[Session]: ...

    def save(self, session: Session) -> None: ...


class InMemorySessionStore:
    """Thread-safe in-memory session store."""

    def __init__(self, max_turns: int = 20) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}
        self._max_turns = max_turns

    def load(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        with self._lock:
            session.turns = session.turns[-self._max_turns:]
            self._sessions[session.session_id] = session

    def add_turn(self, session_id: str, tenant_id: str, user_id: str, turn: SessionTurn) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                self._sessions[session_id] = session
            session.turns.append(turn)
            session.turns = session.turns[-self._max_turns:]
            return session
