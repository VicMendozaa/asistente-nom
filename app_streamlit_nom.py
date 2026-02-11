import pandas as pd
import json, re
from pathlib import Path

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Config
# ----------------------------
KB_PATH = Path(__file__).with_name("nom_kb_chunks.json")

NOMS = {
    "NOM-017-STPS-2008": {
        "alias": ["017", "nom 017", "nom-017", "epp", "equipo de protección personal", "equipo de proteccion personal"],
        "tema": "Equipo de Protección Personal (selección, uso, mantenimiento, evidencia)."
    },
    "NOM-018-STPS-2015": {
        "alias": ["018", "nom 018", "nom-018", "sga", "sistema armonizado", "hds", "hoja de datos de seguridad", "pictograma", "etiquetado"],
        "tema": "Comunicación de peligros por sustancias químicas (SGA/GHS: etiquetas, HDS, pictogramas, capacitación)."
    },
    "NOM-027-STPS-2008": {
        "alias": ["027", "nom 027", "nom-027", "soldadura", "corte", "oxiacet", "arco", "electrodo", "cilindro", "gases"],
        "tema": "Condiciones de seguridad e higiene en actividades de soldadura y corte."
    }
}

INTENT_RULES = {
    "comparativa": ["compar", "diferencia", "vs", "versus", "entre", "cuál aplica", "cual aplica", "cuál nom", "cual nom", "relación", "relacion"],
    "implementacion": ["implementar", "implementación", "pasos", "ruta", "cómo cumplo", "como cumplo", "plan", "secuencial", "simultánea", "simultanea"],
    "conflictos": ["conflicto", "duplicidad", "duplicado", "duplicar", "se repite", "repetir", "no repetir", "sin repetir", "sin duplicar", "contradic", "incompatible"]
}

TOPIC_HINTS = {
    "epp": ["epp", "casco", "lentes", "careta", "guantes", "botas", "respirador", "arnés", "arnes", "protección personal", "proteccion personal"],
    "quimicos": ["sustancia", "químic", "quimic", "etiquet", "pictograma", "hds", "hoja de datos", "clasificación", "clasificacion", "riesgo químico", "riesgo quimico"],
    "soldadura": ["soldadura", "corte", "flama", "chisp", "arco", "electrodo", "oxiacet", "cilindro", "gas", "esmeril"]
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def detect_intents(q: str):
    qn = normalize(q)
    intents = []
    for intent, kws in INTENT_RULES.items():
        if any(kw in qn for kw in kws):
            intents.append(intent)
    return intents or ["consulta"]

def detect_noms(q: str):
    qn = normalize(q)
    hits = []
    # explicit NOM mention
    for nom, info in NOMS.items():
        if any(a in qn for a in info["alias"]):
            hits.append(nom)

    # topic-based fallback
    if not hits:
        if any(k in qn for k in TOPIC_HINTS["soldadura"]):
            hits.append("NOM-027-STPS-2008")
        if any(k in qn for k in TOPIC_HINTS["quimicos"]):
            hits.append("NOM-018-STPS-2015")
        if any(k in qn for k in TOPIC_HINTS["epp"]):
            hits.append("NOM-017-STPS-2008")

    # if still none, assume it could touch all
    return hits or list(NOMS.keys())

@st.cache_resource
def load_kb_and_index():
    chunks = json.loads(KB_PATH.read_text(encoding="utf-8"))
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=60000)
    X = vectorizer.fit_transform(corpus)
    return chunks, vectorizer, X

def retrieve(chunks, vectorizer, X, query, nom_filter=None, topk=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idxs = np.argsort(-sims)

    out = []
    for idx in idxs:
        c = chunks[idx]
        if nom_filter and c["nom"] not in nom_filter:
            continue
        if sims[idx] < 0.06:
            continue
        out.append({**c, "score": float(sims[idx])})
        if len(out) >= topk:
            break
    return out

def build_integrated_route(noms_detected):
    # A simple integrated implementation route for these three NOMs
    steps = []
    if "NOM-018-STPS-2015" in noms_detected:
        steps.append("1) **NOM-018**: Levanta inventario de sustancias químicas, clasifica peligros y asegura **etiquetas + Hojas de Datos de Seguridad (HDS)** accesibles. Define cómo se comunicará el riesgo.")
    if "NOM-027-STPS-2008" in noms_detected:
        steps.append("2) **NOM-027**: Haz el análisis de riesgos en soldadura/corte (fuentes de ignición, humos/gases, espacios confinados, etc.) y define controles del proceso (área, ventilación, equipos, extinción, permisos/procedimientos).")
    if "NOM-017-STPS-2008" in noms_detected:
        steps.append("3) **NOM-017**: Con base en los riesgos (incluyendo químicos y del proceso), selecciona **EPP** adecuado, capacita en uso/limitaciones y establece mantenimiento/rehabilitación, entrega y registros.")
    steps.append("4) Integra evidencia: unifica capacitación, listas de verificación y registros para evitar duplicidades (un solo expediente que cubra 017+018+027).")
    return "\n".join(steps)

def answer(query, intents, noms_detected, evidence):
    # Response template that always ties norms together when relevant.
    lines=[]
    lines.append("### Respuesta integrada")
    lines.append(f"**Intención detectada:** {', '.join(intents)}")
    lines.append(f"**Norma(s) detectada(s):** {', '.join(noms_detected)}")
    lines.append("")

    # If user asks for implementation route
    if "implementacion" in intents:
        lines.append("#### Ruta sugerida de implementación integrada")
        lines.append(build_integrated_route(noms_detected))
        lines.append("")

    # If comparative
    if "comparativa" in intents:
        lines.append("#### Comparación rápida (qué aporta cada NOM)")
        for nom in noms_detected:
            lines.append(f"- **{nom}**: {NOMS[nom]['tema']}")
        lines.append("")
        lines.append("#### Recomendación")
        lines.append("Si tu proceso es **soldadura/corte** y hay **sustancias químicas** (gases, solventes, humos), normalmente se aplican **NOM-027 (proceso)** + **NOM-018 (comunicación de peligros)** + **NOM-017 (EPP)** de forma complementaria.")
        lines.append("")

    if "conflictos" in intents:
        lines.append("#### Posibles duplicidades/conflictos típicos")
        lines.append("- **Capacitación**: aparece en varias NOM → conviene un **plan único** con módulos por tema (SGA, soldadura/corte, EPP) y un solo registro de asistencia.")
        lines.append("- **Señalización/etiquetado**: NOM-018 exige comunicación de peligros químicos; NOM-027 puede pedir controles del área → evita duplicar formatos: usa etiquetas/HDS como base y complementa con avisos del área de trabajo.")
        lines.append("- **Evidencias**: centraliza en un expediente por área/proceso (soldadura) con anexos de sustancias (018) y EPP (017).")
        lines.append("")

    # Use evidence to ground the answer
    lines.append("#### Evidencia localizada en las NOM")
    if not evidence:
        lines.append("No se encontró un fragmento suficientemente cercano con la búsqueda actual. Prueba con palabras más específicas (p. ej., 'obligaciones del patrón', 'HDS', 'careta para soldar').")
        return "\n".join(lines)

    for ev in evidence[:4]:
        preview = re.sub(r"\s+", " ", ev["text"]).strip()
        if len(preview) > 450:
            preview = preview[:450] + "..."
        lines.append(f"- **{ev['nom']} (p. {ev['page']})** — {preview}")

    lines.append("")
lines.append("#### Interpretación práctica")

parts = []
if "NOM-027-STPS-2008" in noms_detected:
    parts.append("- **Proceso (NOM-027):** controles del área y procedimiento de soldadura/corte (ventilación, orden, control de chispas/flama, cilindros, prevención de incendios).")
if "NOM-018-STPS-2015" in noms_detected:
    parts.append("- **Comunicación/SGA (NOM-018):** inventario, etiquetas, pictogramas y **HDS** de solventes/gases; capacitación para interpretarlas.")
if "NOM-017-STPS-2008" in noms_detected:
    parts.append("- **EPP (NOM-017):** selección/entrega/capacitación/mantenimiento y registros del EPP según riesgos del proceso y químicos.")

lines.append("\n".join(parts))
lines.append("")
lines.append("Si me das tu caso (tipo de soldadura, sustancias y área), lo aterrizo a un checklist integrado.")
return "\n".join(lines)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Asistente NOM 017-018-027", layout="wide")
st.title("Asistente Virtual NOM (NOM-017 / NOM-018 / NOM-027)")
st.caption("Prototipo funcional: identifica la(s) NOM aplicable(s), recupera evidencia y sugiere implementación integrada.")

chunks, vectorizer, X = load_kb_and_index()

with st.sidebar:
    st.subheader("NOM integradas")
    for nom, info in NOMS.items():
        st.markdown(f"**{nom}**\n\n- {info['tema']}\n")
    st.divider()
    topk = st.slider("Fragmentos a recuperar (top-k)", 3, 10, 5)
    st.write("Tip: si preguntas de forma comparativa, usa 'diferencias' o 'cuál aplica'.")
# ----------------------------
# Mapa comparativo (tabla)
# ----------------------------
MATRIX_PATH = Path(__file__).with_name("matriz_comparativa_nom.csv")

st.divider()
st.subheader("Mapa comparativo de NOM (017 / 018 / 027)")

if st.button("📌 Generar mapa comparativo"):
    if MATRIX_PATH.exists():
        df_map = pd.read_csv(MATRIX_PATH, engine="python", sep=",", on_bad_lines="skip")
        st.caption("Tabla comparativa (extraída y resumida de las NOM).")
        st.dataframe(df_map, use_container_width=True)

        csv_bytes = df_map.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar mapa comparativo (CSV)",
            data=csv_bytes,
            file_name="mapa_comparativo_nom.csv",
            mime="text/csv",
        )
    else:
        st.error("No se encontró 'matriz_comparativa_nom.csv' en el repo. Súbelo junto con la app.")




query = st.text_area("Escribe tu consulta (puede ser comparativa):", height=120, placeholder="Ej. Para soldadura con gas y uso de solventes, ¿qué aplica de la NOM-027, NOM-018 y NOM-017 y en qué orden lo implemento?")

if st.button("Consultar") and query.strip():
    intents = detect_intents(query)
    noms_detected = detect_noms(query)
    evidence = retrieve(chunks, vectorizer, X, query, nom_filter=noms_detected, topk=topk)

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Detección")
        st.write("**Intención:**", ", ".join(intents))
        st.write("**NOM(s):**", ", ".join(noms_detected))

    with col2:
        st.subheader("Evidencia (fragmentos)")
        for ev in evidence:
            with st.expander(f"{ev['nom']} — p. {ev['page']} — score {ev['score']:.2f}"):
                st.write(ev["text"])

    st.markdown(answer(query, intents, noms_detected, evidence))
