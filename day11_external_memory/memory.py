import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryManager:
    """Manages SQLite-based external memory for conversation persistence."""

    def __init__(self, db_path: str):
        """Initialize the memory manager with a database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self) -> None:
        """Establish connection to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

    def _create_tables(self) -> None:
        """Create the messages table if it doesn't exist."""
        if not self.conn:
            return

        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Save all messages to the database."""
        if not self.conn:
            return False

        try:
            cursor = self.conn.cursor()
            timestamp = datetime.utcnow().isoformat()

            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                tool_call_id = message.get("tool_call_id")
                tool_calls = message.get("tool_calls")

                tool_calls_json = None
                if tool_calls:
                    try:
                        tool_calls_json = json.dumps(tool_calls)
                    except (TypeError, ValueError):
                        tool_calls_json = None

                cursor.execute("""
                    INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (session_id, role, content, tool_call_id, tool_calls_json, timestamp))

            self.conn.commit()
            return True
        except Exception as e:
            if self.conn:
                self.conn.rollback()
            return False

    def load_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Load messages from the database for a given session."""
        if not self.conn:
            return []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT role, content, tool_call_id, tool_calls
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
            """, (session_id,))

            messages = []
            for row in cursor.fetchall():
                message: Dict[str, Any] = {
                    "role": row["role"],
                }

                if row["content"] is not None:
                    message["content"] = row["content"]

                if row["tool_call_id"] is not None:
                    message["tool_call_id"] = row["tool_call_id"]

                if row["tool_calls"] is not None:
                    try:
                        message["tool_calls"] = json.loads(row["tool_calls"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                messages.append(message)

            return messages
        except Exception:
            return []

    def get_message_count(self, session_id: str) -> int:
        """Get the count of messages for a session."""
        if not self.conn:
            return 0

        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def clear_messages(self, session_id: Optional[str] = None) -> bool:
        """Clear messages for a specific session, or all messages if session_id is None."""
        if not self.conn:
            return False

        try:
            cursor = self.conn.cursor()
            if session_id is None:
                cursor.execute("DELETE FROM messages")
            else:
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.conn.commit()
            return True
        except Exception:
            if self.conn:
                self.conn.rollback()
            return False

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

