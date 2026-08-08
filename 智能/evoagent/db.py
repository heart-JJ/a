from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL CHECK (kind IN ('atomic', 'workflow', 'router', 'evaluator')),
    scope TEXT NOT NULL CHECK (scope IN ('general', 'domain')),
    risk_tier TEXT NOT NULL DEFAULT 'low' CHECK (risk_tier IN ('low', 'medium', 'high')),
    lifecycle TEXT NOT NULL CHECK (lifecycle IN ('draft', 'experimental', 'active', 'deprecated', 'archived', 'quarantined')),
    latest_version INTEGER NOT NULL,
    active_version INTEGER,
    origin TEXT NOT NULL DEFAULT 'manual',
    protected INTEGER NOT NULL DEFAULT 0 CHECK (protected IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (id, active_version) REFERENCES skill_versions(skill_id, version) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS skill_versions (
    skill_id TEXT NOT NULL REFERENCES skills(id),
    version INTEGER NOT NULL CHECK (version > 0),
    parent_version INTEGER,
    spec_json TEXT NOT NULL CHECK (json_valid(spec_json)),
    content_hash TEXT NOT NULL,
    changelog TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    source_candidate_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (skill_id, version),
    FOREIGN KEY (skill_id, parent_version) REFERENCES skill_versions(skill_id, version)
);

CREATE TABLE IF NOT EXISTS skill_lifecycle_events (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id),
    from_status TEXT,
    to_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_release_events (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    from_version INTEGER,
    to_version INTEGER NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (skill_id, to_version) REFERENCES skill_versions(skill_id, version)
);

CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    input_json TEXT NOT NULL CHECK (json_valid(input_json)),
    output_json TEXT NOT NULL CHECK (json_valid(output_json)),
    pattern_key TEXT NOT NULL,
    tags_json TEXT NOT NULL CHECK (json_valid(tags_json)),
    salience REAL NOT NULL CHECK (salience >= 0 AND salience <= 1),
    technical_success INTEGER NOT NULL CHECK (technical_success IN (0, 1)),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0),
    reflection TEXT NOT NULL DEFAULT '',
    eligible_for_evolution INTEGER NOT NULL DEFAULT 1 CHECK (eligible_for_evolution IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experience_skills (
    experience_id TEXT NOT NULL REFERENCES experiences(id),
    skill_id TEXT NOT NULL,
    skill_version INTEGER NOT NULL,
    position INTEGER NOT NULL,
    match_score REAL NOT NULL,
    decision_json TEXT NOT NULL CHECK (json_valid(decision_json)),
    PRIMARY KEY (experience_id, position),
    FOREIGN KEY (skill_id, skill_version) REFERENCES skill_versions(skill_id, version)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL REFERENCES experiences(id),
    source TEXT NOT NULL CHECK (source IN ('user', 'test', 'tool', 'model')),
    success INTEGER CHECK (success IN (0, 1)),
    score REAL CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    confidence REAL NOT NULL DEFAULT 1 CHECK (confidence >= 0 AND confidence <= 1),
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_usage_events (
    id TEXT PRIMARY KEY,
    experience_id TEXT NOT NULL REFERENCES experiences(id),
    skill_id TEXT NOT NULL,
    skill_version INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    technical_success INTEGER NOT NULL CHECK (technical_success IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE (experience_id, skill_id, skill_version),
    FOREIGN KEY (skill_id, skill_version) REFERENCES skill_versions(skill_id, version)
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('new_skill', 'workflow', 'revision')),
    target_skill_id TEXT REFERENCES skills(id),
    pattern_key TEXT NOT NULL,
    name TEXT NOT NULL,
    proposed_spec_json TEXT NOT NULL CHECK (json_valid(proposed_spec_json)),
    evidence_json TEXT NOT NULL CHECK (json_valid(evidence_json)),
    validation_json TEXT NOT NULL CHECK (json_valid(validation_json)),
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    risk_tier TEXT NOT NULL DEFAULT 'low' CHECK (risk_tier IN ('low', 'medium', 'high')),
    created_at TEXT NOT NULL,
    decided_at TEXT,
    decision_note TEXT
);

CREATE TABLE IF NOT EXISTS candidate_decisions (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected')),
    note TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    input_count INTEGER NOT NULL DEFAULT 0,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    summary_json TEXT NOT NULL CHECK (json_valid(summary_json)),
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS meta_parameters (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id TEXT PRIMARY KEY,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('complete', 'streaming', 'failed', 'cancelled')),
    model TEXT,
    resolved_model TEXT,
    provider TEXT,
    generation_id TEXT,
    finish_reason TEXT,
    error_code TEXT,
    reply_to_message_id TEXT REFERENCES chat_messages(id),
    selected_skills_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(selected_skills_json)),
    tags_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(tags_json)),
    memory_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(memory_refs_json)),
    usage_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(usage_json)),
    error TEXT NOT NULL DEFAULT '',
    experience_id TEXT REFERENCES experiences(id),
    first_token_at TEXT,
    finished_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (conversation_id, sequence)
);

CREATE TABLE IF NOT EXISTS chat_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    assistant_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    client_request_id TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'streaming', 'completed', 'failed', 'cancelled')),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (conversation_id, client_request_id)
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    requested_model TEXT NOT NULL,
    response_model TEXT,
    input_type TEXT NOT NULL CHECK (input_type IN ('query', 'passage')),
    content_hash TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector_blob BLOB,
    normalized INTEGER NOT NULL DEFAULT 0 CHECK (normalized IN (0, 1)),
    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'retry_wait', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    next_retry_at TEXT,
    error_code TEXT,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (message_id, requested_model, content_hash, input_type)
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiences_pattern ON experiences(pattern_key);
CREATE INDEX IF NOT EXISTS idx_evaluations_experience ON evaluations(experience_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_skill ON skill_usage_events(skill_id, skill_version);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_runs_conversation ON chat_runs(conversation_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_runs_active_conversation
ON chat_runs(conversation_id) WHERE status IN ('queued', 'streaming');
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_ready
ON memory_embeddings(requested_model, dimensions, status);
DROP INDEX IF EXISTS uq_candidates_kind_pattern;
CREATE UNIQUE INDEX IF NOT EXISTS uq_candidates_open_pattern
ON candidates(kind, pattern_key) WHERE status IN ('pending', 'approved');

CREATE TRIGGER IF NOT EXISTS immutable_skill_versions_update
BEFORE UPDATE ON skill_versions BEGIN
    SELECT RAISE(ABORT, 'skill versions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_skill_versions_delete
BEFORE DELETE ON skill_versions BEGIN
    SELECT RAISE(ABORT, 'skill versions cannot be deleted');
END;

CREATE TRIGGER IF NOT EXISTS prevent_skill_delete
BEFORE DELETE ON skills BEGIN
    SELECT RAISE(ABORT, 'skills cannot be deleted; archive them instead');
END;

CREATE TRIGGER IF NOT EXISTS immutable_experience_core
BEFORE UPDATE OF id, task, input_json, output_json, pattern_key, tags_json, salience,
                 technical_success, latency_ms, reflection, created_at ON experiences BEGIN
    SELECT RAISE(ABORT, 'experience facts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_evaluations_update
BEFORE UPDATE ON evaluations BEGIN
    SELECT RAISE(ABORT, 'evaluations are append-only');
END;

CREATE TRIGGER IF NOT EXISTS immutable_evaluations_delete
BEFORE DELETE ON evaluations BEGIN
    SELECT RAISE(ABORT, 'evaluations are append-only');
END;

CREATE TRIGGER IF NOT EXISTS immutable_usage_update
BEFORE UPDATE ON skill_usage_events BEGIN
    SELECT RAISE(ABORT, 'usage events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_usage_delete
BEFORE DELETE ON skill_usage_events BEGIN
    SELECT RAISE(ABORT, 'usage events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_audit_update
BEFORE UPDATE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'audit events are immutable');
END;

CREATE TRIGGER IF NOT EXISTS immutable_audit_delete
BEFORE DELETE ON audit_events BEGIN
    SELECT RAISE(ABORT, 'audit events cannot be deleted');
END;
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        with self.read() as connection:
            connection.executescript(SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '2')"
            )
            connection.commit()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Open a connection and always release its Windows file handle."""

        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
