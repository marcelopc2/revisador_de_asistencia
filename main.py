import streamlit as st
import pandas as pd
import requests
import re
import unicodedata
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
API_TOKEN = config("TOKEN")

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

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
def canvas_request(session, method, endpoint, payload=None, paginated=False):
    if not BASE_URL:
        raise ValueError("BASE_URL no está configurada (env URL).")

    url = endpoint if endpoint.startswith("http") else f"{BASE_URL}{endpoint}"
    results = []

    try:
        while url:
            if payload is not None and method.upper() == "GET":
                resp = session.request(method.upper(), url, params=payload, headers=HEADERS)
            else:
                resp = session.request(method.upper(), url, json=payload, headers=HEADERS)

            if not resp.ok:
                st.error(f"Error Canvas {resp.status_code}: {resp.text}")
                return None

            data = resp.json()

            if paginated:
                results.extend(data if isinstance(data, list) else [data])
                url = resp.links.get("next", {}).get("url")
            else:
                return data

        return results

    except requests.exceptions.RequestException as e:
        st.error(f"Excepción Canvas: {e}")
        return None

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
        if data is None:
            return pd.DataFrame()

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
        if data is None:
            return pd.DataFrame()

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
# UI
# =========================
# st.title("🧑🏻‍💻 Revisador de asistencia semi automático")
st.markdown("#### 🧑🏻‍💻 Revisador de asistencia semi automático")
st.info(
    "**¿Cómo usar?**\n\n"
    "1. Ingresa el **ID del curso** en Canvas.\n"
    "2. Sube el **CSV de asistencia** (exportado desde Canvas/Zoom).\n"
    "3. Presiona **Procesar** — la app cruza los nombres automáticamente y genera la tabla con **P** (presente) y **A** (ausente).\n\n"
    "💡 Activa **Solo asistencia** si quieres el resultado más rápido. "
    "La tabla se puede copiar directo a Excel con **Ctrl+Shift+V** para mantener el formato."
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

if process:
    if not course_id.strip():
        st.error("Debes ingresar el ID de curso.")
        st.stop()
    if uploaded is None:
        st.error("Debes subir un CSV.")
        st.stop()

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
        students_df = fetch_enrolled_students(course_id.strip())

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
            activity_df      = fetch_student_activity(course_id.strip())
            assignments_list = fetch_course_assignments(course_id.strip())

            a_foro   = find_assignment_by_keywords(assignments_list, ["foro"])
            a_equipo = find_assignment_by_keywords(assignments_list, ["equipo"])
            a_final  = find_assignment_by_keywords(assignments_list, ["final"], exclude=["equipo"])

            foro_ids   = fetch_assignment_submitters(course_id.strip(), a_foro["id"])   if a_foro   else frozenset()
            equipo_ids = fetch_assignment_submitters(course_id.strip(), a_equipo["id"]) if a_equipo else frozenset()
            final_ids  = fetch_assignment_submitters(course_id.strip(), a_final["id"])  if a_final  else frozenset()

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
