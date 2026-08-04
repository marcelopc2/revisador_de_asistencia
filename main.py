import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
import sqlite3
import hashlib
from pathlib import Path
from contextlib import contextmanager
from difflib import SequenceMatcher
from collections import Counter
from itertools import combinations
from decouple import config

# =========================
# Streamlit config
# =========================
st.set_page_config(
    layout="wide",
    page_title="Revisador de asistencia semi automático",
    page_icon="🧑🏻‍💻"
)

st.markdown(
    """
    <style>
    a[href*="github.com"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    a[href*="github.com"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Canvas config
# =========================
BASE_URL = config("URL")   # ej: "https://canvas.uautonoma.cl/api/v1"

# =========================
# Token store (SQLite local, NO versionado)
# =========================
# Los tokens viven en tokens.db, que está en .gitignore. El repo es público:
# nunca guardar credenciales en usage.db (ese sí está versionado).
# Se administran 100% desde el panel (UI) — no se siembran desde .env/secrets,
# porque un token viejo revocado ahí quedaría ensuciando la lista para siempre.
# Filesystem de Streamlit Cloud es efímero: si el contenedor reinicia, hay que
# volver a pegar los tokens vigentes en el panel.
_TOKENS_DB_PATH = Path(__file__).parent / "tokens.db"
TOKENS_CODE = config("TOKENS_CODE", default="ver_tokens")

# Códigos que significan "este token no sirve para esto" -> probar el siguiente.
# 404 va incluido porque Canvas lo devuelve cuando el token es válido pero no
# tiene permiso sobre el curso, que es justo el síntoma de un token de menor nivel.
ROTATE_ON_STATUS = (401, 403, 404)

@contextmanager
def _tokens_db():
    """`with sqlite3.connect(...)` hace commit pero no cierra: aquí sí cerramos,
    porque get_tokens() se llama en cada petición y filtraría conexiones."""
    conn = sqlite3.connect(_TOKENS_DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _init_tokens_db():
    with _tokens_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_tokens (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                label      TEXT NOT NULL,
                token      TEXT NOT NULL UNIQUE,
                added_at   TEXT DEFAULT (datetime('now', 'localtime')),
                last_ok_at TEXT,
                last_error TEXT,
                fail_count INTEGER DEFAULT 0
            )
        """)

def get_tokens() -> list[dict]:
    """Tokens ordenados: primero el que funcionó más recientemente."""
    _init_tokens_db()
    with _tokens_db() as conn:
        rows = conn.execute("""
            SELECT id, label, token, added_at, last_ok_at, last_error, fail_count
            FROM api_tokens
            ORDER BY last_ok_at IS NULL, last_ok_at DESC, id ASC
        """).fetchall()
    cols = ["id", "label", "token", "added_at", "last_ok_at", "last_error", "fail_count"]
    return [dict(zip(cols, r)) for r in rows]

def add_token(label: str, token: str) -> tuple[bool, str]:
    _init_tokens_db()
    token = (token or "").strip()
    label = (label or "").strip() or "sin nombre"
    if not token:
        return False, "El token no puede estar vacío."
    try:
        with _tokens_db() as conn:
            conn.execute("INSERT INTO api_tokens (label, token) VALUES (?, ?)", (label, token))
        return True, f"Token «{label}» agregado."
    except sqlite3.IntegrityError:
        return False, "Ese token ya está en la lista."

def delete_token(token_id: int):
    _init_tokens_db()
    with _tokens_db() as conn:
        conn.execute("DELETE FROM api_tokens WHERE id = ?", (token_id,))

def _mark_token_ok(token_id: int):
    with _tokens_db() as conn:
        conn.execute("""
            UPDATE api_tokens
            SET last_ok_at = datetime('now', 'localtime'), last_error = NULL, fail_count = 0
            WHERE id = ?
        """, (token_id,))

def _mark_token_failed(token_id: int, error: str):
    with _tokens_db() as conn:
        conn.execute("""
            UPDATE api_tokens
            SET last_error = ?, fail_count = fail_count + 1
            WHERE id = ?
        """, (error, token_id))

def mask_token(token: str) -> str:
    token = token or ""
    return f"…{token[-4:]}" if len(token) > 4 else "…"

# =========================
# Helpers: limpieza + similitud
# =========================
def clean_string(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = re.sub(r"[\u0300-\u036f]", "", s)  # sin tildes
    s = re.sub(r"[^a-z0-9\s]", " ", s)     # solo alfanum + espacios
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_tokens(s: str) -> list[str]:
    s = clean_string(s)
    return [t for t in s.split(" ") if t] if s else []

def seq_ratio(a: str, b: str) -> float:
    a = clean_string(a)
    b = clean_string(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def token_ratio_sorted(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Ayuda con 'Perez Juan' vs 'Juan Perez'."""
    a = " ".join(sorted(a_tokens))
    b = " ".join(sorted(b_tokens))
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def token_overlap_score(p_tokens: list[str], s_tokens: list[str]) -> float:
    """
    Overlap con soporte de:
    - exacto
    - prefijo (>=3)  ej: 'alej' -> 'alejandro'
    - inicial (1)    ej: 'j' -> 'juan'
    """
    if not p_tokens or not s_tokens:
        return 0.0

    s_set = set(s_tokens)
    hits = 0.0

    for pt in p_tokens:
        if pt in s_set:
            hits += 1.0
            continue

        if len(pt) == 1:
            if any(stt.startswith(pt) for stt in s_tokens):
                hits += 0.6
            continue

        if len(pt) >= 3:
            if any(stt.startswith(pt) for stt in s_tokens):
                hits += 0.8
                continue

    return min(1.0, hits / max(1, len(p_tokens)))

def best_token_fuzzy(p_tokens: list[str], s_tokens: list[str]) -> float:
    """Mejor similitud token-vs-token para casos raros."""
    if not p_tokens or not s_tokens:
        return 0.0
    best = 0.0
    for pt in p_tokens:
        for stt in s_tokens:
            best = max(best, seq_ratio(pt, stt))
    return best

def is_noise_name(s: str) -> bool:
    """Evita basura tipo 'iphone', 'guest', etc."""
    s = clean_string(s)
    if not s:
        return True
    noise = {"iphone", "android", "zoom", "usuario", "user", "invitado", "guest", "sala", "tablet", "celular", "pc"}
    toks = set(split_tokens(s))
    return len(toks) == 0 or (len(toks) == 1 and list(toks)[0] in noise)

# =========================
# Canvas request helper
# =========================
class CanvasUnavailable(Exception):
    """Ningún token pudo completar la petición. Se lanza (en vez de devolver
    None) para que st.cache_data NO cachee el fallo: si solo devolviéramos un
    DataFrame vacío, Streamlit lo guardaría en caché 5 minutos y un token
    agregado después seguiría sin probarse hasta que la caché expirara."""

def canvas_request(session, method, endpoint, payload=None, paginated=False):
    """
    Ejecuta la petición probando los tokens disponibles en orden. Si un token
    devuelve 401/403/404 (inválido, revocado o sin permiso sobre el recurso),
    lo marca como fallido y reintenta con el siguiente.
    """
    if not BASE_URL:
        raise ValueError("BASE_URL no está configurada (env URL).")

    tokens = get_tokens()
    if not tokens:
        msg = (
            f"No hay tokens configurados. Escribe «{TOKENS_CODE}» en el campo "
            "**ID curso** y presiona Procesar para agregar uno."
        )
        st.error(msg)
        raise CanvasUnavailable(msg)

    start_url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    attempts = []

    for tok in tokens:
        headers = {
            "Authorization": f"Bearer {tok['token']}",
            "Content-Type": "application/json",
        }
        url = start_url
        results = []
        rotate = False

        try:
            while url:
                if payload is not None and method.upper() == "GET":
                    resp = session.request(method.upper(), url, params=payload, headers=headers)
                else:
                    resp = session.request(method.upper(), url, json=payload, headers=headers)

                if resp.status_code in ROTATE_ON_STATUS:
                    _mark_token_failed(tok["id"], f"HTTP {resp.status_code}")
                    attempts.append(f"{tok['label']} ({mask_token(tok['token'])}) → HTTP {resp.status_code}")
                    rotate = True
                    break

                if not resp.ok:
                    # Error que no tiene que ver con el token: no rotamos.
                    msg = f"Error Canvas {resp.status_code}: {resp.text}"
                    st.error(msg)
                    raise CanvasUnavailable(msg)

                data = resp.json()

                if paginated:
                    results.extend(data if isinstance(data, list) else [data])
                    url = resp.links.get("next", {}).get("url")
                else:
                    _mark_token_ok(tok["id"])
                    return data

            if rotate:
                continue

            _mark_token_ok(tok["id"])
            return results

        except requests.exceptions.RequestException as e:
            msg = f"Excepción Canvas: {e}"
            st.error(msg)
            raise CanvasUnavailable(msg)

    msg = (
        "Ningún token pudo completar la petición.\n\n"
        + "\n".join(f"- {a}" for a in attempts)
        + f"\n\nRevisa los tokens escribiendo «{TOKENS_CODE}» en **ID curso**."
    )
    st.error(msg)
    raise CanvasUnavailable(msg)

# =========================
# Canvas: estudiantes matriculados
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_enrolled_students(course_id: str) -> pd.DataFrame:
    with requests.Session() as session:
        payload = {
            "type[]": ["StudentEnrollment"],
            "state[]": ["active"],
            "per_page": 100,
            "include[]": ["user"]
        }
        data = canvas_request(session, "GET", f"/courses/{course_id}/enrollments", payload=payload, paginated=True)

    rows = []
    for enr in data:
        user = enr.get("user") or {}
        rows.append({
            "canvas_user_id": user.get("id"),
            "name": user.get("name") or "",
            "sortable_name": user.get("sortable_name") or "",
            "login_id": user.get("login_id") or "",
            "sis_user_id": user.get("sis_user_id") or "",
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["canvas_user_id"]).reset_index(drop=True)

    df["name_clean"] = df["name"].apply(clean_string)
    df["sortable_clean"] = df["sortable_name"].apply(clean_string)
    df["name_tokens"] = df["name"].apply(split_tokens)
    df["sortable_tokens"] = df["sortable_name"].apply(split_tokens)
    df["login_clean"] = df["login_id"].apply(clean_string)

    # para comparación "pegada"
    df["joined_name_clean"] = df["name_clean"].str.replace(" ", "", regex=False)
    df["joined_sortable_clean"] = df["sortable_clean"].str.replace(" ", "", regex=False)

    return df

# =========================
# CSV: detectar columnas (ES/EN)
# =========================
def guess_csv_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    low = [c.lower().strip() for c in cols]

    name_candidates = [
        "nombre de usuario", "nombre", "participante", "nombre del participante",
        "name", "participant", "participant name", "display name", "user name", "username"
    ]
    email_candidates = [
        "e-mail de usuario", "email de usuario", "correo", "mail",
        "email", "email address", "e-mail"
    ]

    col_name = ""
    col_email = ""

    for cand in name_candidates:
        if cand in low:
            col_name = cols[low.index(cand)]
            break

    for cand in email_candidates:
        if cand in low:
            col_email = cols[low.index(cand)]
            break

    if not col_name:
        for i, c in enumerate(low):
            if "nombre" in c or "name" in c:
                col_name = cols[i]
                break

    if not col_email:
        for i, c in enumerate(low):
            if "mail" in c or "correo" in c or "email" in c or "e-mail" in c:
                col_email = cols[i]
                break

    return col_name, col_email

# =========================
# Índice de apellidos (para regla "apellido único")
# =========================
def build_surname_index(students_df: pd.DataFrame) -> tuple[dict, Counter]:
    surname_to_ids = {}
    for _, r in students_df.iterrows():
        sortable = clean_string(r["sortable_name"])
        if sortable and "," in sortable:
            surname_part = sortable.split(",", 1)[0].strip()
            for s in split_tokens(surname_part):
                surname_to_ids.setdefault(s, []).append(int(r["canvas_user_id"]))
        else:
            nt = r["name_tokens"]
            if nt:
                surname_to_ids.setdefault(nt[-1], []).append(int(r["canvas_user_id"]))

    surname_counts = Counter({k: len(set(v)) for k, v in surname_to_ids.items()})
    return surname_to_ids, surname_counts

# =========================
# Regla username concatenado con saltos
# =========================
def ordered_token_list_for_student(stu_row: pd.Series) -> list[str]:
    base = [t for t in stu_row["name_tokens"] if t]
    extra = [t for t in stu_row["sortable_tokens"] if t and t not in base]
    return base + extra

def rule_username_concatenated(p_clean_join: str, students_df: pd.DataFrame) -> int | None:
    if not p_clean_join or len(p_clean_join) < 6:
        return None

    hits = []

    for _, r in students_df.iterrows():
        if p_clean_join in r["joined_name_clean"] or p_clean_join in r["joined_sortable_clean"]:
            hits.append(int(r["canvas_user_id"]))
            continue

        toks = ordered_token_list_for_student(r)

        for k in (2, 3):
            if len(toks) < k:
                continue

            for idxs in combinations(range(len(toks)), k):
                candidate = "".join(toks[i] for i in idxs)

                if candidate == p_clean_join:
                    hits.append(int(r["canvas_user_id"]))
                    break

                if len(p_clean_join) >= 8 and (p_clean_join in candidate or candidate in p_clean_join):
                    hits.append(int(r["canvas_user_id"]))
                    break
            else:
                continue
            break

    hits = list(set(hits))
    return hits[0] if len(hits) == 1 else None

# =========================
# Regla apellido único
# =========================
def rule_unique_surname(p_tokens: list[str], surname_to_ids: dict, surname_counts: Counter) -> int | None:
    candidates = [t for t in p_tokens if len(t) >= 5 and t.isalpha()]
    for t in candidates:
        if surname_counts.get(t, 0) == 1:
            ids = list(set(surname_to_ids.get(t, [])))
            if len(ids) == 1:
                return ids[0]
    return None

# =========================
# Fuzzy scoring (solo respaldo)
# =========================
def score_student(participant_name: str, participant_email: str, stu_row: pd.Series) -> tuple[float, str]:
    pname_clean = clean_string(participant_name)
    pemail_clean = clean_string(participant_email)
    if pemail_clean in {"nan", "none", "null"}:
        pemail_clean = ""

    if pemail_clean and stu_row["login_clean"] == pemail_clean:
        return 1.0, "email_exact"

    p_tokens = split_tokens(pname_clean)
    if not p_tokens:
        return 0.0, "no_tokens"

    s_tokens = ordered_token_list_for_student(stu_row)
    if not s_tokens:
        return 0.0, "student_no_tokens"

    ov = token_overlap_score(p_tokens, s_tokens)
    tr = token_ratio_sorted(p_tokens, s_tokens)
    sr = max(seq_ratio(participant_name, stu_row["name"]), seq_ratio(participant_name, stu_row["sortable_name"]))

    p_join = pname_clean.replace(" ", "")
    joined_student = (
        stu_row["joined_name_clean"]
        if len(stu_row["joined_name_clean"]) >= len(stu_row["joined_sortable_clean"])
        else stu_row["joined_sortable_clean"]
    )

    peg = 0.0
    if p_join and joined_student:
        peg = 1.0 if p_join in joined_student else seq_ratio(p_join, joined_student)

    tf = best_token_fuzzy(p_tokens, s_tokens)

    score = (0.33 * ov) + (0.18 * tr) + (0.18 * sr) + (0.21 * peg) + (0.10 * tf)
    reason = f"ov={ov:.2f} tr={tr:.2f} sr={sr:.2f} peg={peg:.2f} tf={tf:.2f}"
    return float(score), reason

def match_participant(participant_name: str, participant_email: str, students_df: pd.DataFrame,
                      threshold: float, margin: float, strong_threshold: float,
                      surname_to_ids: dict, surname_counts: Counter) -> dict:
    if is_noise_name(participant_name):
        return {"status": "not_found", "best_id": None, "best_name": None, "best_score": 0.0, "candidates": [], "rule": ""}

    pname_clean = clean_string(participant_name)
    p_tokens = split_tokens(pname_clean)
    p_join = pname_clean.replace(" ", "")

    uid = rule_username_concatenated(p_join, students_df)
    if uid is not None:
        hit = students_df[students_df["canvas_user_id"] == uid].iloc[0]
        return {"status": "matched", "best_id": int(uid), "best_name": hit["name"], "best_score": 0.99, "candidates": [], "rule": "username_concat"}

    uid = rule_unique_surname(p_tokens, surname_to_ids, surname_counts)
    if uid is not None:
        hit = students_df[students_df["canvas_user_id"] == uid].iloc[0]
        return {"status": "matched", "best_id": int(uid), "best_name": hit["name"], "best_score": 0.90, "candidates": [], "rule": "unique_surname"}

    scored = []
    for _, stu in students_df.iterrows():
        sc, reason = score_student(participant_name, participant_email, stu)
        if sc > 0:
            scored.append((sc, int(stu["canvas_user_id"]), stu["name"], reason))

    if not scored:
        return {"status": "not_found", "best_id": None, "best_name": None, "best_score": 0.0, "candidates": [], "rule": ""}

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:5]

    best = scored[0]
    second = scored[1] if len(scored) > 1 else (0.0, None, None, "")

    best_score = best[0]
    gap = best[0] - second[0]

    if best_score >= strong_threshold:
        status = "matched"
    elif best_score >= threshold and gap >= margin:
        status = "matched"
    elif best_score >= threshold:
        status = "ambiguous"
    else:
        status = "not_found"

    return {"status": status, "best_id": best[1], "best_name": best[2], "best_score": float(best_score), "candidates": top5, "rule": "fuzzy"}

# =========================
# Canvas: actividad en plataforma
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_student_activity(course_id: str) -> pd.DataFrame:
    with requests.Session() as session:
        payload = {
            "type[]": ["StudentEnrollment"],
            "state[]": ["active"],
            "per_page": 100,
            "include[]": ["user"],
        }
        data = canvas_request(session, "GET", f"/courses/{course_id}/enrollments", payload=payload, paginated=True)

    rows = []
    for enr in data:
        user = enr.get("user") or {}
        uid = user.get("id")
        last_activity = enr.get("last_activity_at")
        total_secs = enr.get("total_activity_time")

        if total_secs is not None:
            h = total_secs // 3600
            m = (total_secs % 3600) // 60
            s = total_secs % 60
            activity_fmt = f"{h:02}:{m:02}:{s:02}"
        else:
            activity_fmt = "00:00:00"

        rows.append({
            "canvas_user_id": uid,
            "Ha participado": "✔️" if last_activity else "❌",
            "Tiempo en plataforma": activity_fmt,
        })

    return pd.DataFrame(rows)


# =========================
# Canvas: tareas y entregas
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_course_assignments(course_id: str) -> list:
    with requests.Session() as session:
        data = canvas_request(session, "GET", f"/courses/{course_id}/assignments",
                              payload={"per_page": 100}, paginated=True)
    return data or []


@st.cache_data(show_spinner=False, ttl=300)
def fetch_assignment_submitters(course_id: str, assignment_id: int) -> frozenset:
    with requests.Session() as session:
        data = canvas_request(session, "GET",
                              f"/courses/{course_id}/assignments/{assignment_id}/submissions",
                              payload={"per_page": 100}, paginated=True)
    if not data:
        return frozenset()

    submitted = set()
    for s in data:
        wfs = s.get("workflow_state")
        grd = s.get("grade")
        if wfs in ("submitted", "graded"):
            if grd is not None:
                try:
                    if float(grd) > 0:
                        submitted.add(s["user_id"])
                except Exception:
                    submitted.add(s["user_id"])
            else:
                submitted.add(s["user_id"])
    return frozenset(submitted)


def find_assignment_by_keywords(assignments: list, keywords: list, exclude: list = None) -> dict | None:
    """Devuelve la primera tarea cuyo nombre normalizado contiene todas las keywords y ninguna de las exclude."""
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        s = re.sub(r"[\u0300-\u036f]", "", s)
        return s.lower()

    for a in assignments:
        name_norm = norm(a.get("name", ""))
        if all(kw in name_norm for kw in keywords):
            if exclude and any(ex in name_norm for ex in exclude):
                continue
            return a
    return None


# =========================
# Usage tracking (SQLite local)
# =========================
_DB_PATH = Path(__file__).parent / "usage.db"
STATS_CODE = config("STATS_CODE", default="ver_stats")

def _init_db():
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        TEXT    DEFAULT (datetime('now', 'localtime')),
                course_h  TEXT
            )
        """)

def _log_usage(course_id: str):
    _init_db()
    course_h = hashlib.md5(course_id.encode()).hexdigest()[:8]
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute("INSERT INTO usage_log (course_h) VALUES (?)", (course_h,))

def _get_stats() -> dict:
    _init_db()
    with sqlite3.connect(_DB_PATH) as conn:
        total   = conn.execute("SELECT COUNT(*) FROM usage_log").fetchone()[0]
        today   = conn.execute("SELECT COUNT(*) FROM usage_log WHERE date(ts) = date('now', 'localtime')").fetchone()[0]
        week    = conn.execute("SELECT COUNT(*) FROM usage_log WHERE ts >= datetime('now', 'localtime', '-7 days')").fetchone()[0]
        month   = conn.execute("SELECT COUNT(*) FROM usage_log WHERE ts >= datetime('now', 'localtime', '-30 days')").fetchone()[0]
        recent  = conn.execute("SELECT ts FROM usage_log ORDER BY id DESC LIMIT 15").fetchall()
        by_day  = conn.execute("""
            SELECT date(ts) as dia, COUNT(*) as usos
            FROM usage_log
            WHERE ts >= datetime('now', 'localtime', '-30 days')
            GROUP BY dia ORDER BY dia DESC
        """).fetchall()
    return {"total": total, "today": today, "week": week, "month": month,
            "recent": [r[0] for r in recent], "by_day": by_day}


# =========================
# Panel de tokens
# =========================
def render_tokens_panel():
    st.markdown("### 🔑 Tokens de Canvas")
    st.caption(
        "Se prueban de arriba hacia abajo; el que funcionó más recientemente queda primero. "
        f"Si uno responde {', '.join(str(c) for c in ROTATE_ON_STATUS)}, se marca como fallido "
        "y se reintenta con el siguiente."
    )

    with st.form("add_token_form", clear_on_submit=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            new_label = st.text_input("Nombre", placeholder="Ej: cuenta admin 2026")
        with c2:
            new_token = st.text_input("Token", type="password", placeholder="Pega el token de Canvas")
        if st.form_submit_button("Agregar token", type="primary"):
            ok, msg = add_token(new_label, new_token)
            (st.success if ok else st.error)(msg)

    tokens = get_tokens()
    if not tokens:
        st.warning("No hay tokens cargados.")
    else:
        st.markdown(f"**{len(tokens)} token(s) disponibles**")
        for tok in tokens:
            c1, c2, c3, c4 = st.columns([2.2, 1.2, 2.4, 0.8])
            c1.markdown(f"**{tok['label']}**  \n`{mask_token(tok['token'])}`")
            # last_error se limpia en cada éxito, así que revisarlo primero
            # evita mostrar "OK" en un token que falló después de su último éxito.
            c2.markdown(
                "⚠️ Falló" if tok["last_error"] else ("✅ OK" if tok["last_ok_at"] else "⏳ Sin usar")
            )
            detalle = []
            if tok["last_ok_at"]:
                detalle.append(f"Último OK: {tok['last_ok_at']}")
            if tok["last_error"]:
                detalle.append(f"Último error: {tok['last_error']} ({tok['fail_count']} fallos)")
            detalle.append(f"Agregado: {tok['added_at']}")
            c3.caption("  \n".join(detalle))
            if c4.button("🗑", key=f"del_token_{tok['id']}", help="Eliminar este token"):
                delete_token(tok["id"])
                st.rerun()

    st.divider()
    if st.button("Cerrar panel"):
        st.session_state["show_tokens"] = False
        st.rerun()


# =========================
# UI
# =========================
# st.title("🧑🏻‍💻 Revisador de asistencia semi automático")
st.markdown("#### 🧑🏻‍💻 Revisador de asistencia semi automático")
st.info(
    "**¿Cómo usar?**\n\n"
    "1. Ingresa el **ID del curso** en Canvas.\n"
    "2. Sube el **CSV de asistencia** (exportado desde Canvas/Zoom).\n"
    "3. Presiona **Procesar** — la app cruza los nombres automáticamente y genera la tabla con **P** (presente) y **A** (ausente).\n"
    "4. Para copiar la tabla a Excel manteniendo el formato, usa **Ctrl+Shift+V** al pegar.\n\n"
    "💡 Activa **Solo asistencia** si quieres el resultado más rápido."
)

# c1, c2 = st.columns([2, 3])
# with c1:
course_id = st.text_input("ID curso", placeholder="Ej: 12345")
# with c2:
uploaded = st.file_uploader("CSV de asistencia", type=["csv"])
# c1, c2, c3, c4 = st.columns([1.2, 1.1, 1.1, 1.6])
# with c1:
#     
# with c2:
#     threshold = st.slider("Umbral match (fuzzy)", 0.55, 0.90, 0.68, 0.01)
# with c3:
#     margin = st.slider("Margen de ambigüedad", 0.00, 0.20, 0.05, 0.01)
# with c4:
#     strong_threshold = st.slider("Match fuerte (auto)", 0.80, 0.98, 0.86, 0.01)

solo_asistencia = st.checkbox("Solo asistencia (más rápido)", value=False)
process = st.button("Procesar", type="primary", width='stretch')

# --- Panel de tokens (código secreto en el campo ID curso) ---
if process and course_id.strip() == TOKENS_CODE:
    st.session_state["show_tokens"] = True

if st.session_state.get("show_tokens"):
    render_tokens_panel()
    st.stop()

if process:
    # --- Modo estadísticas ---
    if course_id.strip() == STATS_CODE:
        stats = _get_stats()
        st.markdown("### 📊 Estadísticas de uso")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total usos", stats["total"])
        c2.metric("Hoy", stats["today"])
        c3.metric("Últimos 7 días", stats["week"])
        c4.metric("Últimos 30 días", stats["month"])
        if stats["by_day"]:
            st.markdown("**Usos por día (últimos 30 días)**")
            st.dataframe(
                pd.DataFrame(stats["by_day"], columns=["Día", "Usos"]),
                hide_index=True, width="stretch"
            )
        if stats["recent"]:
            st.markdown("**Últimas 15 ejecuciones**")
            st.dataframe(
                pd.DataFrame(stats["recent"], columns=["Timestamp"]),
                hide_index=True, width="stretch"
            )
        st.stop()

    if not course_id.strip():
        st.error("Debes ingresar el ID de curso.")
        st.stop()
    if uploaded is None:
        st.error("Debes subir un CSV.")
        st.stop()

    _log_usage(course_id.strip())

    # Leer CSV
    try:
        try:
            csv_df = pd.read_csv(uploaded)
        except Exception:
            uploaded.seek(0)
            csv_df = pd.read_csv(uploaded, sep=";")
    except Exception as e:
        st.error(f"No pude leer el CSV: {e}")
        st.stop()

    if csv_df.empty:
        st.error("El CSV está vacío.")
        st.stop()

    col_name, col_email = guess_csv_columns(csv_df)
    if not col_name:
        st.error("No pude detectar la columna de nombre. En tu CSV debería ser 'Nombre de usuario'.")
        st.stop()

    #st.caption(f"Columnas detectadas → Nombre: '{col_name}' | Email: '{col_email if col_email else '(no detectado)'}'")

    # Canvas students
    with st.spinner("Consultando estudiantes matriculados en el curso..."):
        try:
            students_df = fetch_enrolled_students(course_id.strip())
        except CanvasUnavailable:
            st.stop()

    if students_df.empty:
        st.error("No pude obtener estudiantes (o no hay estudiantes activos).")
        st.stop()

    surname_to_ids, surname_counts = build_surname_index(students_df)

    # Preparar participantes (FIX NaN + dedup correcto)
    participants = csv_df.copy()
    participants["participant_name"] = participants[col_name].fillna("").astype(str).str.strip()

    if col_email:
        participants["participant_email"] = participants[col_email].fillna("").astype(str).str.strip()
    else:
        participants["participant_email"] = ""

    participants["name_clean"] = participants["participant_name"].apply(clean_string)
    participants["email_clean"] = participants["participant_email"].apply(clean_string)
    participants.loc[participants["email_clean"].isin(["nan", "none", "null"]), "email_clean"] = ""

    participants = participants[participants["name_clean"] != ""]
    participants = participants[~participants["participant_name"].apply(is_noise_name)]

    with_email = participants[participants["email_clean"] != ""].copy()
    no_email = participants[participants["email_clean"] == ""].copy()

    if not with_email.empty:
        with_email = with_email.sort_values(["email_clean", "name_clean"]).drop_duplicates(subset=["email_clean"], keep="first")
    if not no_email.empty:
        no_email = no_email.sort_values(["name_clean"]).drop_duplicates(subset=["name_clean"], keep="first")

    participants = pd.concat([with_email, no_email], ignore_index=True)

    # Matching
    matched_student_ids = set()
    debug_rows = []
    ambiguous_rows = []
    not_found_rows = []

    with st.spinner("Comparando csv vs matriculados..."):
        for _, p in participants.iterrows():
            pname = p["participant_name"]
            pemail = p["participant_email"]

            m = match_participant(
                participant_name=pname,
                participant_email=pemail,
                students_df=students_df,
                # threshold=threshold,
                # margin=margin,
                # strong_threshold=strong_threshold,
                threshold=0.55,
                margin=0.05,
                strong_threshold=0.86,
                surname_to_ids=surname_to_ids,
                surname_counts=surname_counts
            )

            top5 = m["candidates"]
            top5_fmt = " | ".join([f"{nm} ({sc:.2f})" for sc, _id, nm, _rsn in top5]) if top5 else ""

            if m["status"] == "matched":
                matched_student_ids.add(m["best_id"])
                debug_rows.append({
                    "CSV Nombre": pname,
                    # "CSV_email": pemail,
                    "status": "MATCHED",
                    "matched_student": m["best_name"],
                    "Puntaje de match": round(m["best_score"], 3),
                    "rule": m.get("rule", ""),
                    "top5": top5_fmt
                })
            elif m["status"] == "ambiguous":
                ambiguous_rows.append({
                    "CSV Nombre": pname,
                    # "CSV_email": pemail,
                    "Mejor candidato": m["best_name"],
                    "Puntaje de match": round(m["best_score"], 3),
                    "Top 5 posibles match": top5_fmt
                })
                debug_rows.append({
                    "CSV Nombre": pname,
                    # "CSV_email": pemail,
                    "status": "AMBIGUOUS",
                    "matched_student": m["best_name"],
                    "Puntaje de match": round(m["best_score"], 3),
                    # "rule": m.get("rule", ""),
                    "Top 5 posibles match": top5_fmt
                })
            else:
                not_found_rows.append({
                    "CSV Nombre": pname,
                    # "CSV_email": pemail,
                    "Top 5 posibles match": top5_fmt
                })
                debug_rows.append({
                    "CSV Nombre": pname,
                    # "CSV_email": pemail,
                    "status": "NOT_FOUND",
                    "matched_student": None,
                    "Puntaje match": round(m["best_score"], 3),
                    "rule": m.get("rule", ""),
                    "Top 5 posibles match": top5_fmt
                })

    # =========================
    # Tabla principal (solo columnas solicitadas y renombradas)
    # =========================
    result = students_df[["sortable_name", "login_id"]].copy()
    result["Asistencia"] = students_df["canvas_user_id"].apply(lambda uid: "P" if int(uid) in matched_student_ids else "A")

    result = result.sort_values("sortable_name").reset_index(drop=True)
    result = result.rename(columns={
        "sortable_name": "Nombre alumno",
        "login_id": "Email",
        "Asistencia": "Asistencia"
    })

    # =========================
    # Columnas extendidas: participación en plataforma y tareas
    # =========================
    if not solo_asistencia:
        with st.spinner("Obteniendo datos de actividad y tareas del curso..."):
            try:
                activity_df      = fetch_student_activity(course_id.strip())
                assignments_list = fetch_course_assignments(course_id.strip())

                a_foro   = find_assignment_by_keywords(assignments_list, ["foro"])
                a_equipo = find_assignment_by_keywords(assignments_list, ["equipo"])
                a_final  = find_assignment_by_keywords(assignments_list, ["final"], exclude=["equipo"])

                foro_ids   = fetch_assignment_submitters(course_id.strip(), a_foro["id"])   if a_foro   else frozenset()
                equipo_ids = fetch_assignment_submitters(course_id.strip(), a_equipo["id"]) if a_equipo else frozenset()
                final_ids  = fetch_assignment_submitters(course_id.strip(), a_final["id"])  if a_final  else frozenset()
            except CanvasUnavailable:
                st.stop()

        uid_act = {}
        if not activity_df.empty:
            uid_act = activity_df.set_index("canvas_user_id")[["Ha participado", "Tiempo en plataforma"]].to_dict("index")

        ext_rows = []
        for _, row in students_df.iterrows():
            uid = int(row["canvas_user_id"])
            act = uid_act.get(uid, {})
            ext_rows.append({
                "Nombre alumno":        row["sortable_name"],
                "Ha participado":       act.get("Ha participado", "❌"),
                "Tiempo en plataforma": act.get("Tiempo en plataforma", "00:00:00"),
                "Foro académico":       "✔️" if uid in foro_ids   else "❌",
                "Trabajo en equipo":    "✔️" if uid in equipo_ids else "❌",
                "Trabajo final":        "✔️" if uid in final_ids  else "❌",
            })
        ext_df = pd.DataFrame(ext_rows)
        result = result.merge(ext_df, on="Nombre alumno", how="left")

    def style_attendance(val):
        if val == "P":
            return "background-color: #c6efce; color: #006100; font-weight: 800; text-align: center;"
        return "background-color: #ffc7ce; color: #9c0006; font-weight: 800; text-align: center;"

    def style_bool(val):
        if val == "✔️":
            return "background-color: #c6efce; color: #006100; text-align: center;"
        if val == "❌":
            return "background-color: #ffc7ce; color: #9c0006; text-align: center;"
        return ""

    styled = result.style.map(style_attendance, subset=["Asistencia"])
    if not solo_asistencia:
        bool_cols = ["Ha participado", "Foro académico", "Trabajo en equipo", "Trabajo final"]
        styled = styled.map(style_bool, subset=bool_cols)

    st.success(
        f"Matriculados: {len(result)} | Presentes (P): {sum(result['Asistencia']=='P')} | "
        f"Ausentes (A): {sum(result['Asistencia']=='A')} | Nombres únicos en CSV: {len(participants)} | "
        f"Ambiguos: {len(ambiguous_rows)} | No encontrados: {len(not_found_rows)}"
    )

    st.subheader("📋 Tabla resultados asistencia")
    st.dataframe(styled, width='stretch', hide_index=True)

    # st.download_button(
    #     "Descargar tabla (CSV)",
    #     result.to_csv(index=False).encode("utf-8"),
    #     file_name=f"asistencia_curso_{course_id}.csv",
    #     mime="text/csv",
    #     width='stretch'
    # )

    # =========================
    # Debug SIEMPRE visible (sin botón)
    # =========================
    # st.divider()
    # st.subheader("🧪 Resumen de razonamiento")

    st.markdown("##### ⚠️ Ambiguos (No se pusieron presentes)")
    st.dataframe(pd.DataFrame(ambiguous_rows), width='stretch', hide_index=True)

    st.markdown("##### ❌ No encontrados")
    st.dataframe(pd.DataFrame(not_found_rows), width='stretch', hide_index=True)

    st.markdown("##### ✅ Con quien fueron emparejados")
    df_matched = pd.DataFrame([r for r in debug_rows if r.get("status") == "MATCHED"])
    if not df_matched.empty:
        df_matched = df_matched.drop(columns=["status", "rule"], errors="ignore")
        df_matched = df_matched.rename(columns={
            "matched_student": "Match",
            "top5": "Top 5 posibles match"
        })
    st.dataframe(df_matched, width='stretch', hide_index=True)

    # st.markdown("### 🔎 Todo procesado")
    # st.dataframe(pd.DataFrame(debug_rows), width='stretch', hide_index=True)
