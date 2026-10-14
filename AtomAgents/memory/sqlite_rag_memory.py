import sqlite3
from typing import List, Optional, Dict, Any, Tuple

import json
import math
from pydantic import ConfigDict, BaseModel, Field, PrivateAttr

try:  # removed in langchain-core 1.x; only the pydantic base matters here
    from langchain_core.memory import BaseMemory  # type: ignore
except ImportError:
    BaseMemory = BaseModel  # type: ignore
from langchain_core.messages import BaseMessage, messages_to_dict
from ..utils.utils import _message_from_dict

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    HumanMessage = None
    AIMessage = None
try:
    from langchain_openai import OpenAIEmbeddings
except ModuleNotFoundError:
    OpenAIEmbeddings = None  # type: ignore
import openai
from logging import getLogger

logger = getLogger(__name__)


# Simple OpenAI embedding wrapper when not using langchain
class GPTEmbedding:
    """Wrapper for OpenAI Embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        base_url: str = "https://api.openai.com/v1/",
    ):
        self.model = model
        self.base_url = base_url
        self.client = openai.Client(base_url=self.base_url)

    def embed_documents(self, texts):
        # Use OpenAI API v1 interface for embeddings
        # new interface: openai.embeddings.create
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
        except AttributeError:
            # fallback to legacy if available
            resp = self.client.Embedding.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


try:  # optional legacy backend; the numpy vector cache is the default path
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None  # type: ignore


class SQLiteRAGMemory(BaseMemory):
    """
    Hybrid memory using SQLite for message storage and FAISS vector store for RAG.
    Messages are stored with their OpenAI embeddings.
    Retrieval uses a hybrid strategy:
    - Semantic similarity via cosine over stored embeddings.
    - Lexical relevance via SQLite FTS5 (BM25 ranking) over msg_text.
    Results are merged and re-ranked for better precision/recall.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Static fields for memory attributes (avoid dynamic assignment in __init__)
    sqlite_path: str = "memory.db"
    k: int = 5
    # Any embedder exposing embed_query/embed_documents — OpenAIEmbeddings, the
    # local sentence-transformers backend, or a caller-supplied double. Was
    # annotated OpenAIEmbeddings back when that was the only option.
    embedding_model: Any = None
    # Short human-readable name of the resolved backend, e.g.
    # "local:sentence-transformers/all-MiniLM-L6-v2". Recorded so a run can be
    # audited afterwards for whether semantic recall was actually on.
    embedding_desc: str = ""

    ## after __init__ we can use this to store the embedding function
    embedding_fn: Any = None
    conn: sqlite3.Connection = None
    faiss: bool = False
    # Optional in-memory FAISS index for faster and more scalable similarity
    # search.  Activated when the caller passes *use_faiss=True* **and** the
    # dependency is available.
    vectorstore: Optional[Any] = None

    # Whether the FAISS backend should be used instead of the ad-hoc SQLite
    # cosine_sim user-defined function.  This flag is set by the *use_faiss*
    # keyword argument in ``__init__`` so that existing code keeps working
    # unchanged.
    use_faiss: bool = False
    project_root: str = Field(default_factory=str)

    # Lazy in-memory vector index: (row ids, normalized embedding matrix,
    # agent names). Built on first semantic search, invalidated on writes and
    # rebinds — replaces the O(n) JSON-parsing SQL UDF per query.
    _vec_ids: Any = PrivateAttr(default=None)
    _vec_matrix: Any = PrivateAttr(default=None)
    _vec_agents: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__()
        # Override defaults from kwargs
        # sqlite_path, k, embedding_model, and use_faiss can be provided
        if "sqlite_path" in kwargs:
            self.sqlite_path = str(kwargs["sqlite_path"])
        if "k" in kwargs:
            self.k = kwargs["k"]
        if "embedding_model" in kwargs:
            self.embedding_model = kwargs["embedding_model"]
        # Enable FAISS when requested and available
        if kwargs.get("use_faiss") and FAISS is not None:
            self.use_faiss = True

        # Initialize the embedding model: prefer the one passed in, else resolve
        # from config. The old default was a bare OpenAIEmbeddings(), which
        # inherits OPENAI_BASE_URL — on a gateway deployment that serves no
        # /v1/embeddings, every recall 404'd and this class silently stored rows
        # without vectors. build_embeddings() probes before returning, so an
        # unusable backend is reported once and loudly instead.
        if self.embedding_model is None:
            try:
                from AtomAgents.utils.embeddings import build_embeddings

                self.embedding_model, self.embedding_desc = build_embeddings()
            except Exception:  # noqa: BLE001 - fall back to the legacy path
                self.embedding_model = (
                    OpenAIEmbeddings() if OpenAIEmbeddings is not None else GPTEmbedding()
                )
        self.embedding_fn = getattr(self.embedding_model, "embed_query", lambda x: [])
        self.project_root = kwargs.get("project_root", "")

        # Schema creation and FTS5 setup
        self._ensure_schema()

    def rebind(self, sqlite_path: str, project_root: Optional[str] = None) -> None:
        """Point this memory at a different session database.

        Always use this instead of assigning ``sqlite_path`` directly:
        the new database gets the full schema (messages, indexes, FTS5,
        OKR tables), and a stale FTS index is rebuilt from ``messages``.
        """
        self.sqlite_path = str(sqlite_path)
        if project_root is not None:
            self.project_root = str(project_root)
        self._ensure_schema()

    @property
    def memory_variables(self) -> List[str]:
        return ["history"]

    @staticmethod
    def _strip_bulky_payloads(msg_dict: dict) -> dict:
        """Replace inline base64 payloads (images/PDFs) with short placeholders.

        Measured in real sessions: a single uploaded image was re-serialized in
        8 messages at ~934 KB each — 45% of the whole DB. The base64 data is of
        no retrieval value; artifacts live on disk.
        """

        def _clean_content(content):
            if not isinstance(content, list):
                return content
            cleaned = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if isinstance(url, str) and url.startswith("data:"):
                            part = {
                                "type": "text",
                                "text": f"[inline image omitted: {len(url)} chars]",
                            }
                    elif part.get("type") == "file":
                        data = (part.get("file") or {}).get("file_data", "")
                        if isinstance(data, str) and data.startswith("data:"):
                            part = {
                                "type": "text",
                                "text": f"[inline file omitted: {len(data)} chars]",
                            }
                    elif part.get("type") == "reasoning" and part.get(
                        "encrypted_content"
                    ):
                        # Responses-API reasoning items carry multi-KB opaque
                        # blobs of no retrieval value (same amplification
                        # class as inline images, defect S4). Keep the block
                        # shape, drop the payload.
                        part = {
                            k: v
                            for k, v in part.items()
                            if k != "encrypted_content"
                        }
                cleaned.append(part)
            return cleaned

        data = msg_dict.get("data")
        if isinstance(data, dict) and "content" in data:
            data = dict(data)
            data["content"] = _clean_content(data["content"])
            msg_dict = dict(msg_dict)
            msg_dict["data"] = data
        elif "content" in msg_dict:
            msg_dict = dict(msg_dict)
            msg_dict["content"] = _clean_content(msg_dict["content"])
        return msg_dict

    @staticmethod
    def _tool_call_text(msg_dict: dict) -> str:
        """Synthesize searchable text for tool-call messages whose content is
        empty (36% of rows in real sessions had empty msg_text)."""
        data = msg_dict.get("data", msg_dict)
        parts = []
        for tc in data.get("tool_calls") or []:
            name = tc.get("name", "tool")
            args = json.dumps(tc.get("args", {}), ensure_ascii=False)
            parts.append(f"[tool_call] {name}({args[:2000]})")
        return "\n".join(parts)

    def _add_message(
        self,
        message: BaseMessage,
        job_id=None,
        task_id=None,
        key_result_id=None,
        agent_name: Optional[str] = None,
    ):
        msg_dict = self._strip_bulky_payloads(messages_to_dict([message])[0])
        msg_text = self._get_text_content(msg_dict) or self._tool_call_text(msg_dict)
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            # Skip immediate duplicates (handoffs were historically persisted
            # twice: once inside a history blob, once as plaintext).
            if msg_text and len(msg_text) > 200:
                last = cur.execute(
                    "SELECT msg_text, agent_name FROM messages ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if last and last[0] == msg_text and last[1] == agent_name:
                    logger.debug(
                        "Skipping duplicate message for agent %s (%d chars)",
                        agent_name,
                        len(msg_text),
                    )
                    return
            try:
                emb = self.embedding_fn(msg_text) if msg_text else []
            except Exception as e:
                logger.warning("Embedding failed; storing without vector: %s", e)
                emb = []
            emb_json = json.dumps(emb)
            cur.execute(
                "INSERT INTO messages (msg_json, msg_text, job_id, objective_id, key_result_id, embedding, agent_name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (
                    json.dumps(msg_dict),
                    msg_text,
                    job_id,
                    task_id,
                    key_result_id,
                    emb_json,
                    agent_name,
                ),
            )
            rowid = cur.lastrowid
            # Mirror into FTS index for lexical retrieval
            try:
                cur.execute(
                    "INSERT INTO messages_fts(rowid, msg_text, agent_name) VALUES (?, ?, ?)",
                    (rowid, msg_text, agent_name or ""),
                )
            except sqlite3.Error as e:
                logger.warning("FTS insert failed (lexical search degraded): %s", e)
            conn.commit()
        self._invalidate_vector_cache()

    def save_context(
        self,
        inputs,
        outputs,
        job_id=None,
        task_id=None,
        key_result_id=None,
        agent_name: Optional[str] = None,
    ):
        def _to_iter(val):
            if val is None:
                return []
            if isinstance(val, dict):
                return val.values()
            if isinstance(val, (list, tuple, set)):
                return val
            return [val]

        for user_val in _to_iter(inputs):
            if not user_val:
                continue
            if isinstance(user_val, BaseMessage):
                msg = user_val
            else:
                msg = (
                    HumanMessage(content=user_val)
                    if HumanMessage
                    else BaseMessage(content=user_val)
                )
            self._add_message(msg, job_id, task_id, key_result_id, agent_name)
        for ai_val in _to_iter(outputs):
            if not ai_val:
                continue
            if isinstance(ai_val, BaseMessage):
                msg = ai_val
            else:
                msg = (
                    AIMessage(content=ai_val)
                    if AIMessage
                    else BaseMessage(content=ai_val)
                )
            self._add_message(msg, job_id, task_id, key_result_id, agent_name)

    @staticmethod
    def _flatten_query(value: Any) -> str:
        """Flatten a message-content-shaped query to plain text, never raising.

        With the Responses API (output_version="responses/v1") AIMessage
        content is ALWAYS a block list, often led by a {"type": "reasoning"}
        block — the old coercion took [0] and raised ValueError, killing the
        agent run. A retrieval query must never be able to do that: text-ish
        blocks are joined, everything else (reasoning/function_call/images)
        is ignored.
        """
        from ..utils.text import message_text

        return message_text(value)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not inputs or self.embedding_fn is None:
            return {"history": []}
        query_text = self._flatten_query(
            inputs.get("input") or next(iter(inputs.values()))
        )
        if not query_text:
            return {"history": []}
        exclude_agent = inputs.get("agent_name")
        results = self.search(query_text, k=self.k, exclude_agent=exclude_agent)
        # Per-snippet cap: the RelativeInformation blob was previously
        # unbounded (k snippets × unbounded msg_text) at 5 prompt sites.
        from .. import config as _config

        cap = int(getattr(_config, "RAG_SNIPPET_CHARS", 1200) or 0)
        history = []
        for r in results:
            text = r["msg_text"] or ""
            if cap and len(text) > cap:
                text = text[:cap] + " …[snippet truncated]"
            history.append(text)
        return {"history": history}

    # --- Hybrid search API ---
    def search(
        self,
        query: str,
        k: int = 10,
        exclude_agent: Optional[str] = None,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Hybrid search over memory.

        - alpha controls weight of semantic similarity (0..1).
        - Returns list of {id, msg_text, agent_name, score} sorted by score desc.
        """
        query = (query or "").strip()
        if not query:
            return []

        sem_results = self._semantic_search(query, limit=max(k, 10), exclude_agent=exclude_agent)
        lex_results = self._lexical_search(query, limit=max(k, 10), exclude_agent=exclude_agent)

        # Normalize and fuse scores
        def _norm(vals: List[float]) -> Dict[int, float]:
            if not vals:
                return {}
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-9:
                return {i: 1.0 for i in range(len(vals))}
            return {i: (v - lo) / (hi - lo) for i, v in enumerate(vals)}

        sem_scores = _norm([r["score"] for r in sem_results])
        lex_scores = _norm([r["score"] for r in lex_results])

        fused: Dict[int, Dict[str, Any]] = {}
        # Map by message id for deduplication
        for idx, r in enumerate(sem_results):
            fused[r["id"]] = {**r, "score": alpha * sem_scores.get(idx, 0.0)}
        for jdx, r in enumerate(lex_results):
            prev = fused.get(r["id"], {**r, "score": 0.0})
            prev["score"] = prev["score"] + (1 - alpha) * lex_scores.get(jdx, 0.0)
            fused[r["id"]] = prev

        ranked = sorted(fused.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        return ranked[:k]

    def _invalidate_vector_cache(self) -> None:
        self._vec_ids = None
        self._vec_matrix = None
        self._vec_agents = None

    def _build_vector_cache(self) -> bool:
        """Load all embeddings once into a normalized numpy matrix."""
        try:
            import numpy as np
        except ImportError:
            return False
        ids: List[int] = []
        embs: List[List[float]] = []
        agents: List[Optional[str]] = []
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                "SELECT id, embedding, agent_name FROM messages "
                "WHERE embedding IS NOT NULL AND embedding != '[]'"
            )
            for rid, emb_json, agent in cur:
                try:
                    v = json.loads(emb_json)
                except (TypeError, ValueError):
                    continue
                if v:
                    ids.append(rid)
                    embs.append(v)
                    agents.append(agent)
        if not embs:
            self._vec_ids, self._vec_matrix, self._vec_agents = [], None, []
            return True
        # Guard against mixed embedding models: keep the dominant dimension.
        dims = [len(v) for v in embs]
        dominant = max(set(dims), key=dims.count)
        keep = [i for i, d in enumerate(dims) if d == dominant]
        m = np.asarray([embs[i] for i in keep], dtype="float32")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vec_matrix = m / norms
        self._vec_ids = [ids[i] for i in keep]
        self._vec_agents = [agents[i] for i in keep]
        return True

    def _semantic_search(
        self, query: str, limit: int, exclude_agent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            emb = self.embedding_fn(query)
        except Exception as e:
            # e.g. the LLM proxy serves chat but not /v1/embeddings — degrade
            # to lexical-only retrieval instead of crashing the agent turn.
            logger.warning("Query embedding failed (%s); semantic search skipped", e)
            return []
        if not emb:
            return []
        if self._vec_matrix is None and self._vec_ids is None:
            if not self._build_vector_cache():
                return self._semantic_search_sql(query, emb, limit, exclude_agent)
        if self._vec_matrix is None or not self._vec_ids:
            return []
        import numpy as np

        q = np.asarray(emb, dtype="float32")
        if q.shape[0] != self._vec_matrix.shape[1]:
            logger.warning(
                "Query embedding dim %d != index dim %d; skipping semantic search",
                q.shape[0],
                self._vec_matrix.shape[1],
            )
            return []
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        scores = self._vec_matrix @ (q / qn)
        order = np.argsort(-scores)
        results: List[Dict[str, Any]] = []
        wanted_ids: List[int] = []
        for i in order:
            if exclude_agent and self._vec_agents[i] == exclude_agent:
                continue
            wanted_ids.append(self._vec_ids[i])
            results.append(
                {
                    "id": self._vec_ids[i],
                    "msg_text": None,  # filled from DB below
                    "agent_name": self._vec_agents[i],
                    "score": float(scores[i]),
                }
            )
            if len(results) >= limit:
                break
        if wanted_ids:
            with sqlite3.connect(self.sqlite_path) as conn:
                placeholders = ",".join("?" * len(wanted_ids))
                texts = dict(
                    conn.execute(
                        f"SELECT id, msg_text FROM messages WHERE id IN ({placeholders})",
                        wanted_ids,
                    ).fetchall()
                )
            for r in results:
                r["msg_text"] = texts.get(r["id"], "")
        return results

    def _semantic_search_sql(
        self,
        query: str,
        emb: List[float],
        limit: int,
        exclude_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback path (no numpy): cosine via SQL UDF over every row."""
        emb_json = json.dumps(emb)
        rows: List[Tuple[int, str, Optional[str], float]] = []
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.create_function("cosine_sim", 2, lambda a, b: self._cosine(a, b))
            if exclude_agent:
                cur = conn.execute(
                    "SELECT id, msg_text, agent_name, cosine_sim(embedding, ?) AS s FROM messages WHERE agent_name != ? ORDER BY s DESC LIMIT ?",
                    (emb_json, exclude_agent, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT id, msg_text, agent_name, cosine_sim(embedding, ?) AS s FROM messages ORDER BY s DESC LIMIT ?",
                    (emb_json, limit),
                )
            rows = cur.fetchall()
        return [
            {"id": r[0], "msg_text": r[1], "agent_name": r[2], "score": float(r[3] or 0.0)}
            for r in rows
        ]

    def _lexical_search(
        self, query: str, limit: int, exclude_agent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # FTS5 MATCH parses its input as query SYNTAX — raw prose with dots,
        # parens, or quotes raises "fts5: syntax error". Reduce the query to a
        # bag of word tokens joined with OR (recall-oriented BM25 ranking).
        import re as _re

        tokens = list(dict.fromkeys(_re.findall(r"\w+", query.lower())))[:16]
        if not tokens:
            return []
        q = " OR ".join(f'"{t}"' for t in tokens)
        rows: List[Tuple[int, str, Optional[str], float]] = []
        with sqlite3.connect(self.sqlite_path) as conn:
            try:
                # NOTE: no table alias — `alias MATCH ?` is rejected by some
                # SQLite builds ("no such column"), which silently killed the
                # lexical path even where the FTS table existed.
                if exclude_agent:
                    cur = conn.execute(
                        """
                        SELECT m.id, m.msg_text, m.agent_name, bm25(messages_fts) AS r
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ? AND (m.agent_name IS NULL OR m.agent_name != ?)
                        ORDER BY r ASC
                        LIMIT ?
                        """,
                        (q, exclude_agent, limit),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT m.id, m.msg_text, m.agent_name, bm25(messages_fts) AS r
                        FROM messages_fts
                        JOIN messages m ON m.id = messages_fts.rowid
                        WHERE messages_fts MATCH ?
                        ORDER BY r ASC
                        LIMIT ?
                        """,
                        (q, limit),
                    )
                rows = cur.fetchall()
                # bm25 rank: lower (more negative) = better. Negate so higher
                # score = better; search() min-max normalizes afterwards. (The
                # previous 1/(1+rank) formula inverted ranking for negative
                # ranks due to an operator-precedence bug.)
                return [
                    {
                        "id": r[0],
                        "msg_text": r[1],
                        "agent_name": r[2],
                        "score": -float(r[3]) if r[3] is not None else 0.0,
                    }
                    for r in rows
                ]
            except sqlite3.Error as e:
                logger.warning(
                    "Lexical search unavailable (%s); results are semantic-only",
                    e,
                )
                return []

    def clear(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM messages")
            try:
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
            except sqlite3.Error:
                pass
            conn.commit()
        self._invalidate_vector_cache()

    def get_messages_by_agent(self, agent_name: str) -> List[str]:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                "SELECT msg_text FROM messages WHERE agent_name = ? ORDER BY id",
                (agent_name,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_raw_messages_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                "SELECT msg_json FROM messages WHERE agent_name = ? ORDER BY id",
                (agent_name,),
            )
            rows = cur.fetchall()
        raw_messages: List[Dict[str, Any]] = []
        for (msg_json,) in rows:
            try:
                raw_messages.append(json.loads(msg_json))
            except Exception:
                continue
        return raw_messages

    def get_all_raw_messages(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute("SELECT msg_json FROM messages ORDER BY id")
            rows = cur.fetchall()
        raw_messages: List[Dict[str, Any]] = []
        for (msg_json,) in rows:
            try:
                raw_messages.append(json.loads(msg_json))
            except Exception:
                continue
        # Convert raw messages to BaseMessage objects; one malformed row must
        # not poison the whole history load.
        converted: List[Any] = []
        for msg_dict in raw_messages:
            try:
                if isinstance(msg_dict, dict):
                    converted.append(_message_from_dict(msg_dict))
                else:
                    converted.append(BaseMessage(content=msg_dict))
            except Exception as e:
                logger.warning("Skipping unreadable message row: %s", e)
        return converted

    @property
    def msg_store(self) -> str:
        return self.get_all_raw_messages()

    def _get_text_content(self, msg_dict: dict) -> str:
        """Extract flattenable content from a serialized message dict (both
        the {"data": {"content": ...}} and flat {"content": ...} envelopes),
        then delegate to the canonical flattener."""
        from ..utils.text import message_text

        content = None
        try:
            data = msg_dict
            if "data" in data and "content" in data["data"]:
                content = data["data"]["content"]
            elif "content" in data:
                content = data["content"]
        except Exception:
            return ""
        return message_text(content)

    # --- Schema helpers ----------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create base tables, OKR tables, indexes, and FTS5 if available.

        This is the ONE place schema is defined. Historically the backend kept
        a reduced copy (no FTS, no indexes) that it applied after rebinding
        ``sqlite_path``, which is why every real session DB lacked lexical
        search. Always repoint instances via :meth:`rebind`.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    msg_json TEXT,
                    msg_text TEXT,
                    job_id TEXT,
                    objective_id INTEGER,
                    key_result_id INTEGER,
                    embedding TEXT,
                    agent_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Add columns if absent
            cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
            if "embedding" not in cols:
                conn.execute("ALTER TABLE messages ADD COLUMN embedding TEXT")
            if "agent_name" not in cols:
                conn.execute("ALTER TABLE messages ADD COLUMN agent_name TEXT")
            if "created_at" not in cols:
                # ALTER TABLE cannot add a column with a non-constant default;
                # inserts stamp the timestamp explicitly instead.
                conn.execute("ALTER TABLE messages ADD COLUMN created_at TIMESTAMP")

            # OKR tables (detail store; the job README renders a projection)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    description TEXT,
                    status TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subtasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    description TEXT,
                    status TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS key_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER,
                    subtask_id INTEGER,
                    description TEXT,
                    achieved INTEGER,
                    FOREIGN KEY(task_id) REFERENCES tasks(id),
                    FOREIGN KEY(subtask_id) REFERENCES subtasks(id)
                )
                """
            )
            # Detail stores previously scattered across prompt-driven ad-hoc
            # okr.db files (with divergent schemas). SQL = detail; the job
            # README renders the overview.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artefacts (
                    path TEXT PRIMARY KEY,
                    kind TEXT,
                    bytes INTEGER,
                    hash TEXT,
                    kr_id INTEGER,
                    task_id INTEGER,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT,
                    value REAL,
                    unit TEXT,
                    job_id TEXT,
                    kr_id INTEGER,
                    created_at TIMESTAMP
                )
                """
            )
            # Failure/attempt ledger: retries used to start from zero because
            # no record of prior attempts survived anywhere.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subtask_id INTEGER,
                    description TEXT,
                    command_hash TEXT,
                    result TEXT,
                    error_excerpt TEXT,
                    created_at TIMESTAMP
                )
                """
            )

            # Cosine function for semantic similarity
            def _cosine(a_json: str, b_json: str) -> float:
                try:
                    a = json.loads(a_json)
                    b = json.loads(b_json)
                    dot = sum(x * y for x, y in zip(a, b))
                    na = math.sqrt(sum(x * x for x in a))
                    nb = math.sqrt(sum(y * y for y in b))
                    return dot / (na * nb) if na and nb else 0.0
                except Exception:
                    return 0.0

            conn.create_function("cosine_sim", 2, _cosine)

            # Helpful indexes
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_name)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_job ON messages(job_id)"
                )
            except sqlite3.Error as e:
                logger.debug("Index creation skipped: %s", e)

            # Optional FTS5 for lexical search
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(msg_text, agent_name, content='messages', content_rowid='id')"
                )
                # Backfill: legacy DBs have populated `messages` but an absent/
                # stale FTS index (the backend's reduced schema never created
                # it). COUNT() on an external-content FTS table reads through to
                # the content table, so staleness cannot be detected that way —
                # rebuild unconditionally at bind time (idempotent, runs once
                # per session bind).
                n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                if n_msg:
                    conn.execute(
                        "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
                    )
                    logger.info(
                        "Rebuilt messages_fts for %s (%d messages)",
                        self.sqlite_path,
                        n_msg,
                    )
            except sqlite3.Error as e:
                # FTS5 not available in this SQLite build
                logger.warning("FTS5 unavailable for %s: %s", self.sqlite_path, e)
            conn.commit()
        # Any cached vector index belongs to the previous database.
        self._invalidate_vector_cache()

    @staticmethod
    def _cosine(a_json: str, b_json: str) -> float:
        try:
            a = json.loads(a_json)
            b = json.loads(b_json)
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        except Exception:
            return 0.0

    # --- All OKR TABLE CRUD helpers are rewritten below ---
    # Each method now opens a new connection with 'with sqlite3.connect...'

    # --- Artefact / metric / attempt stores --------------------------------

    def register_artefact(
        self,
        path: str,
        kind: str = "",
        kr_id: Optional[int] = None,
        task_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Register a produced artefact, filesystem-verified.

        Raises FileNotFoundError/ValueError when the file is missing or empty —
        DONE must never be narrated for artefacts that do not exist.
        """
        import hashlib
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Artefact does not exist: {path}")
        size = p.stat().st_size
        if size == 0:
            raise ValueError(f"Artefact is empty (0 bytes): {path}")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO artefacts (path, kind, bytes, hash, kr_id, task_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, bytes=excluded.bytes, "
                "hash=excluded.hash, kr_id=excluded.kr_id, task_id=excluded.task_id, "
                "created_at=excluded.created_at",
                (str(p.resolve()), kind, size, digest, kr_id, task_id),
            )
            conn.commit()
        return {"path": str(p.resolve()), "kind": kind, "bytes": size, "hash": digest}

    def get_all_artefacts(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT path, kind, bytes, hash, kr_id, task_id, created_at FROM artefacts ORDER BY created_at"
            ).fetchall()
        keys = ["path", "kind", "bytes", "hash", "kr_id", "task_id", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def record_metric(
        self,
        metric: str,
        value: float,
        unit: str = "",
        job_id: Optional[str] = None,
        kr_id: Optional[int] = None,
    ) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO validation_metrics (metric, value, unit, job_id, kr_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (metric, value, unit, job_id, kr_id),
            )
            conn.commit()

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT metric, value, unit, job_id, kr_id, created_at FROM validation_metrics ORDER BY id"
            ).fetchall()
        keys = ["metric", "value", "unit", "job_id", "kr_id", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def record_attempt(
        self,
        description: str,
        result: str,
        error_excerpt: str = "",
        subtask_id: Optional[int] = None,
        command_hash: str = "",
    ) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO attempts (subtask_id, description, command_hash, result, error_excerpt, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (subtask_id, description[:2000], command_hash, result[:200], error_excerpt[:2000]),
            )
            conn.commit()

    def get_recent_attempts(self, limit: int = 15) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT subtask_id, description, result, error_excerpt, created_at "
                "FROM attempts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ["subtask_id", "description", "result", "error_excerpt", "created_at"]
        return [dict(zip(keys, r)) for r in reversed(rows)]

    def create_task(
        self, job_id: str, description: str, status: str = "submitted"
    ) -> int:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO tasks (job_id, description, status) VALUES (?, ?, ?)",
                (job_id, description, status),
            )
            conn.commit()
            return cur.lastrowid

    def get_current_task(self, job_id: str = None) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            if job_id:
                cur = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                    (job_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks ORDER BY id DESC LIMIT 1"
                )
            row = cur.fetchone()
            if not row:
                return None
            task = {
                "id": row[0],
                "job_id": row[1],
                "description": row[2],
                "status": row[3],
            }
            sub_rows = conn.execute(
                "SELECT id, task_id, description, status FROM subtasks WHERE task_id = ?",
                (task["id"],),
            ).fetchall()
        task["subtasks"] = [
            {
                "id": r[0],
                "task_id": r[1],
                "description": r[2],
                "status": r[3],
            }
            for r in sub_rows
        ]
        return task

    def get_finished_tasks(self, job_id: str = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            if job_id:
                rows = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks WHERE job_id=? AND status<>?",
                    (job_id, "submitted"),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks WHERE status<>?",
                    ("submitted",),
                ).fetchall()
        return [
            {"id": r[0], "job_id": r[1], "description": r[2], "status": r[3]}
            for r in rows
        ]

    def get_finished_subtasks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT id, task_id, description, status FROM subtasks WHERE status=?",
                ("done",),
            ).fetchall()
        return [
            {"id": r[0], "task_id": r[1], "description": r[2], "status": r[3]}
            for r in rows
        ]

    def get_finished_key_results(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            rows = conn.execute(
                "SELECT id, task_id, subtask_id, description, achieved FROM key_results WHERE achieved=1"
            ).fetchall()
        return [
            {
                "id": r[0],
                "task_id": r[1],
                "subtask_id": r[2],
                "description": r[3],
                "achieved": bool(r[4]),
            }
            for r in rows
        ]

    def get_all_tasks(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            if job_id is not None:
                rows = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks WHERE job_id=?",
                    (job_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, job_id, description, status FROM tasks",
                ).fetchall()
        return [
            {"id": r[0], "job_id": r[1], "description": r[2], "status": r[3]}
            for r in rows
        ]

    def get_all_subtasks(self, task_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            if task_id is not None:
                rows = conn.execute(
                    "SELECT id, task_id, description, status FROM subtasks WHERE task_id=?",
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, task_id, description, status FROM subtasks",
                ).fetchall()
        return [
            {
                "id": r[0],
                "task_id": r[1],
                "description": r[2],
                "status": r[3],
            }
            for r in rows
        ]

    def get_all_key_results(
        self, task_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            if task_id is not None:
                rows = conn.execute(
                    "SELECT id, task_id, subtask_id, description, achieved FROM key_results WHERE task_id=?",
                    (task_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, task_id, subtask_id, description, achieved FROM key_results",
                ).fetchall()
        return [
            {
                "id": r[0],
                "task_id": r[1],
                "subtask_id": r[2],
                "description": r[3],
                "achieved": bool(r[4]),
            }
            for r in rows
        ]

    def update_task_status(
        self, task_id: int, status: str, cascade_subtasks: bool = True
    ) -> None:
        """Set a task's status by its integer primary key.

        NOTE: this previously matched `WHERE job_id=?` (the TEXT UUID column)
        while every caller passed the integer task PK — the update silently
        matched zero rows and tasks stayed 'submitted' forever. Aligned with
        update_subtask_status/delete_task/task_exists, which all key on `id`.
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                "UPDATE tasks SET status=? WHERE id=?",
                (status, task_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Task id {task_id} does not exist")
            if cascade_subtasks:
                conn.execute(
                    "UPDATE subtasks SET status=? WHERE task_id=?",
                    (status, task_id),
                )
            conn.commit()

    def create_subtask(
        self, task_id: int, description: str, status: str = "pending"
    ) -> int:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO subtasks (task_id, description, status) VALUES (?, ?, ?)",
                (task_id, description, status),
            )
            conn.commit()
            return cur.lastrowid

    def update_subtask_status(self, subtask_id: int, status: str):
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "UPDATE subtasks SET status=? WHERE id=?", (status, subtask_id)
            )
            conn.commit()

    def create_key_result(
        self,
        task_id: int,
        description: str,
        subtask_id: int = None,
        achieved: bool = False,
    ) -> int:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO key_results (task_id, subtask_id, description, achieved) VALUES (?, ?, ?, ?)",
                (task_id, subtask_id, description, int(achieved)),
            )
            conn.commit()
            return cur.lastrowid

    def set_key_result_achieved(self, key_result_id: int, achieved: bool = True):
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "UPDATE key_results SET achieved=? WHERE id=?",
                (int(achieved), key_result_id),
            )
            conn.commit()

    def get_schema(self) -> Dict[str, List[Dict[str, Any]]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [row[0] for row in cur.fetchall()]
            schema: Dict[str, List[Dict[str, Any]]] = {}
            for tbl in tables:
                cols_cur = conn.execute(f"PRAGMA table_info({tbl})")
                schema[tbl] = [
                    {"name": col[1], "type": col[2]} for col in cols_cur.fetchall()
                ]
        return schema

    def delete_subtask(self, subtask_id: int) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM key_results WHERE subtask_id=?", (subtask_id,))
            conn.execute("DELETE FROM subtasks WHERE id=?", (subtask_id,))
            conn.commit()

    def delete_task(self, task_id: int) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM key_results WHERE task_id=?", (task_id,))
            sub_rows = conn.execute(
                "SELECT id FROM subtasks WHERE task_id=?", (task_id,)
            ).fetchall()
            for (sub_id,) in sub_rows:
                conn.execute("DELETE FROM key_results WHERE subtask_id=?", (sub_id,))
            conn.execute("DELETE FROM subtasks WHERE task_id=?", (task_id,))
            conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            conn.commit()

    def task_exists(self, task_id: int) -> bool:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,))
            return cur.fetchone() is not None

    def subtask_exists(self, subtask_id: int) -> bool:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.execute("SELECT 1 FROM subtasks WHERE id=?", (subtask_id,))
            return cur.fetchone() is not None

    # Optional: Put your _cosine function as a @staticmethod for use in load_memory_variables if needed
    @staticmethod
    def _cosine(a_json: str, b_json: str) -> float:
        try:
            a = json.loads(a_json)
            b = json.loads(b_json)
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0
        except Exception:
            return 0.0
