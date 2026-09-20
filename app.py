# -*- coding: utf-8 -*-
"""
app.py — เว็บแอป Streamlit "AI แนะนำอาชีพเสริม"
โหลดโมเดลที่ฝึกจาก Orange Data Mining (*.pkcls) แล้วประเมินว่าผู้ใช้
"จำเป็นต้องหารายได้เสริม" หรือไม่ จากข้อมูลทางการเงินที่กรอก
(ไม่เจาะจงชื่ออาชีพ เพราะโมเดลไม่มีข้อมูลอาชีพ แนะนำแค่ระดับความจำเป็น)

รันด้วยคำสั่ง:  streamlit run app.py
"""

import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Domain, Table   # โครงสร้างข้อมูลของ Orange (โมเดล .pkcls ต้องการ)

# ===================== 1) ตั้งค่าหน้าเว็บ =====================
st.set_page_config(page_title="AI แนะนำอาชีพเสริม", page_icon="💼", layout="centered")

# ---- ชื่อโมเดลสวย ๆ ที่จะแสดง (ไม่ใส่ก็ได้ แอปจะใช้ชื่อไฟล์แทน) ----
PRETTY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "neural_network": "Neural Network",
}

# ---- ข้อความแทนค่าคลาส 0/1 ของตัวแปรตาม (ธีม "อาชีพเสริม" ไม่เจาะจงชื่ออาชีพ) ----
CLASS_LABELS = {
    "0": "ควรหารายได้เสริมเพิ่มเติม",
    "1": "ยังไม่จำเป็นต้องหารายได้เสริมเร่งด่วน",
}

# ---- ชื่อภาษาไทยของแต่ละคอลัมน์ ----
THAI_NAMES = {
    "Monthly_Income": "รายได้ต่อเดือน (บาท)",
    "Monthly_Expenditure": "รายจ่ายต่อเดือน (บาท)",
    "Market_Volatility_Index": "ดัชนีความผันผวนของตลาด",
    "Inflation_Rate": "อัตราเงินเฟ้อ (%)",
    "Investment_Amount": "เงินลงทุนสะสม (บาท)",
    "Savings_Ratio": "สัดส่วนการออม",
    "Credit_Score": "คะแนนเครดิต",
    "Debt_to_Income_Ratio": "สัดส่วนหนี้สินต่อรายได้",
    "Risk_Tolerance_Level": "ระดับการยอมรับความเสี่ยง",
    "Economic_Sentiment_Score": "ดัชนีความเชื่อมั่นทางเศรษฐกิจ",
    "Investor_Confidence": "ความเชื่อมั่นของนักลงทุน",
    "Financial_Stability_Index": "ดัชนีเสถียรภาพทางการเงิน",
}

# ---- จัดกลุ่มคอลัมน์เป็นหมวดที่คนทั่วไปอ่านแล้วเข้าใจง่าย ----
# (ตัวแปรใดไม่อยู่ในกลุ่มไหนเลย จะถูกใส่ไว้ในกลุ่ม "ข้อมูลอื่น ๆ" ให้อัตโนมัติ)
FEATURE_GROUPS = [
    ("💰 รายรับ-รายจ่าย", ["Monthly_Income", "Monthly_Expenditure"]),
    ("📈 การออมและการลงทุน", ["Investment_Amount", "Savings_Ratio", "Debt_to_Income_Ratio"]),
    ("🏦 เครดิตและความเสี่ยง", ["Credit_Score", "Risk_Tolerance_Level"]),
    ("🌍 ภาพรวมเศรษฐกิจ", ["Market_Volatility_Index", "Inflation_Rate",
                          "Economic_Sentiment_Score", "Investor_Confidence",
                          "Financial_Stability_Index"]),
]


# ===================== 2) ฟังก์ชันค้นหา/โหลดโมเดล =====================
BASE_DIR = Path(__file__).parent.resolve()


def discover_models():
    """สแกนหาไฟล์ .pkcls ทุกไฟล์ในโปรเจกต์ (รวมโฟลเดอร์ย่อย) คืนค่า {ชื่อที่แสดง: Path}"""
    found = {}
    for path in sorted(BASE_DIR.rglob("*.pkcls")):
        if ".git" in path.parts:
            continue
        key = path.stem.lower()
        label = PRETTY_NAMES.get(key, path.stem.replace("_", " "))
        found[label] = path
    return found


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model_from_path(path: str):
    """โหลดไฟล์ .pkcls จากดิสก์ด้วย joblib และแคชไว้ไม่ให้โหลดซ้ำทุกครั้งที่หน้าเว็บรีเฟรช"""
    return joblib.load(path)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model_from_bytes(raw: bytes):
    """โหลดโมเดลจากไฟล์ที่ผู้ใช้อัปโหลดผ่านหน้าเว็บ (ใช้เมื่อหาไฟล์ในโปรเจกต์ไม่เจอ)"""
    return joblib.load(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def get_feature_stats(_model, cache_key: str):
    """ดึงค่า min/max/mean ของแต่ละฟีเจอร์จากข้อมูลฝึก เพื่อตั้งขอบเขต slider ให้เหมาะสม"""
    model = _model
    stats = {}
    data = getattr(model, "instances", None)
    if data is not None and getattr(data, "X", None) is not None and len(data.X):
        X = np.asarray(data.X, dtype=float)
        for i, var in enumerate(model.original_domain.attributes):
            col = X[:, i]
            stats[var.name] = {
                "min": float(np.nanmin(col)),
                "max": float(np.nanmax(col)),
                "mean": float(np.nanmean(col)),
            }
    return stats


# ===================== 3) ส่วนหัวของแอป =====================
st.title("💼 โปรแกรม AI แนะนำอาชีพเสริมที่เหมาะสม")
st.caption(
    "กรอกข้อมูลทางการเงินของคุณ แล้วให้ AI ประเมินว่า **ควรหารายได้เสริมเพิ่มเติม** "
    "หรือฐานะการเงินปัจจุบันยังไปได้ดีอยู่แล้ว"
)

# ---- ค้นหาไฟล์โมเดลทั้งหมดที่มีอยู่จริงในโปรเจกต์ ----
available_models = discover_models()

with st.sidebar:
    st.header("⚙️ ตั้งค่า")

if available_models:
    with st.sidebar:
        model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้ทำนาย", list(available_models.keys()))
    model_file = available_models[model_name]
    model = load_model_from_path(str(model_file))
    cache_key = str(model_file)
else:
    st.error(
        "ไม่พบไฟล์โมเดลนามสกุล .pkcls ในโปรเจกต์นี้\n\n"
        "ถ้า deploy บน Streamlit Cloud ต้อง commit ไฟล์ .pkcls ขึ้น GitHub repo ด้วย"
    )
    with st.expander("🔍 ไฟล์ที่พบในโปรเจกต์ (ใช้ตรวจสอบชื่อไฟล์/ตำแหน่ง)"):
        st.write(f"โฟลเดอร์ของ app.py: `{BASE_DIR}`")
        all_files = sorted(p.relative_to(BASE_DIR).as_posix() for p in BASE_DIR.rglob("*")
                           if p.is_file() and ".git" not in p.parts)
        st.code("\n".join(all_files[:200]) or "(ไม่พบไฟล์ใด ๆ)")

    uploaded = st.file_uploader(
        "หรืออัปโหลดไฟล์โมเดล .pkcls ที่นี่เพื่อทดลองใช้งานชั่วคราว",
        type=["pkcls", "pkl"],
    )
    if uploaded is None:
        st.stop()
    model_name = uploaded.name
    model = load_model_from_bytes(uploaded.getvalue())
    cache_key = uploaded.name

# ---- อ่านโครงสร้างคอลัมน์จากโมเดลที่โหลดได้ ----
domain = model.original_domain
features = list(domain.attributes)
class_var = domain.class_var
stats = get_feature_stats(model, cache_key)

# รายละเอียดทางเทคนิคซ่อนไว้ใน expander ไม่ให้หน้าเว็บดูรก
with st.sidebar:
    st.success(f"โมเดลพร้อมใช้งาน: **{model_name}**")
    with st.expander("รายละเอียดทางเทคนิค"):
        st.write(f"จำนวนตัวแปรต้น: {len(features)} ตัว")
        st.write(f"ตัวแปรตาม: {class_var.name}")


# ===================== 4) สร้างช่องกรอกค่าตัวแปรต้น (แบบฟอร์มเดียว) =====================
st.subheader("📝 กรอกข้อมูลของคุณ")

# จัดกลุ่มฟีเจอร์ตาม FEATURE_GROUPS แล้วเอาที่เหลือ (ถ้ามี) ไปไว้กลุ่มท้ายสุด
grouped_names = {n for _, names in FEATURE_GROUPS for n in names}
remaining = [v.name for v in features if v.name not in grouped_names]
groups = list(FEATURE_GROUPS)
if remaining:
    groups.append(("📌 ข้อมูลอื่น ๆ", remaining))

feature_by_name = {v.name: v for v in features}
user_values = {}

with st.form("predict_form"):
    for group_title, names in groups:
        names_in_model = [n for n in names if n in feature_by_name]
        if not names_in_model:
            continue
        st.markdown(f"**{group_title}**")
        cols = st.columns(2)
        for idx, name in enumerate(names_in_model):
            var = feature_by_name[name]
            box = cols[idx % 2]
            label = THAI_NAMES.get(var.name, var.name)

            if var.is_discrete:
                # คอลัมน์ชนิดข้อความ/หมวดหมู่ → selectbox
                user_values[var.name] = box.selectbox(label, list(var.values), key=f"in_{var.name}")
            else:
                # คอลัมน์ชนิดตัวเลข → slider ปรับง่ายด้วยนิ้ว/เมาส์ ไม่ต้องพิมพ์เอง
                s = stats.get(var.name, {})
                lo = s.get("min", 0.0)
                hi = s.get("max", 100.0)
                default = s.get("mean", (lo + hi) / 2)
                span = max(hi - lo, 1e-6)
                # ตัดสินใจจำนวนทศนิยม/สเต็ปจากขนาดของช่วงข้อมูล
                if span <= 1.5:
                    step, fmt = 0.01, "%.2f"
                elif span <= 200:
                    step, fmt = max(span / 100, 0.1), "%.1f"
                else:
                    step, fmt = max(round(span / 100), 1), "%.0f"
                user_values[var.name] = box.slider(
                    label,
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    format=fmt,
                    key=f"in_{var.name}",
                )
        st.write("")  # เว้นระยะระหว่างกลุ่ม

    submitted = st.form_submit_button("🔮 ประเมินผล", type="primary", use_container_width=True)


# ===================== 5) ประมวลผลและแสดงผลลัพธ์ =====================
if submitted:
    # ---- 5.1 จัดรูปแบบข้อมูลให้ตรงกับตอนฝึกโมเดล ----
    row = []
    for var in features:
        v = user_values[var.name]
        row.append(float(var.values.index(v)) if var.is_discrete else float(v))
    X = np.array([row], dtype=float)

    # ส่งเป็น Orange Table เพื่อให้ Orange แปลงข้อมูลตาม preprocess ของแต่ละโมเดลให้อัตโนมัติ
    table = Table.from_numpy(Domain(domain.attributes), X)

    # ---- 5.2 ทำนายผล ----
    pred_idx, probs = model(table, model.ValueProbs)
    pred_idx = int(pred_idx[0])
    probs = np.asarray(probs)[0]
    pred_label = str(class_var.values[pred_idx])
    pred_text = CLASS_LABELS.get(pred_label, pred_label)
    confidence = float(probs[pred_idx])

    # ---- 5.3 แสดงผลลัพธ์แบบเข้าใจง่าย (ไม่เจาะจงชื่ออาชีพ) ----
    st.subheader("📊 ผลการประเมิน")
    if pred_label == "1":
        st.success(f"✅ {pred_text}")
        st.write(
            "จากข้อมูลที่กรอก ฐานะการเงินของคุณอยู่ในเกณฑ์ที่มั่นคง "
            "ยังไม่มีความจำเป็นเร่งด่วนที่ต้องหารายได้เสริม "
            "แต่ยังสามารถพิจารณาอาชีพเสริมเพื่อเพิ่มความมั่นคงในระยะยาวได้"
        )
    else:
        st.warning(f"⚠️ {pred_text}")
        st.write(
            "จากข้อมูลที่กรอก ฐานะการเงินของคุณอาจยังไม่มั่นคงเพียงพอ "
            "แนะนำให้พิจารณาหารายได้เสริมเพิ่มเติม เพื่อช่วยเสริมสร้างความมั่นคงทางการเงิน"
        )

    st.metric("ความมั่นใจของการประเมิน (Probability)", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    with st.expander("ดูรายละเอียดเพิ่มเติม"):
        st.write("**ความน่าจะเป็นแยกตามผลลัพธ์**")
        st.dataframe(
            pd.DataFrame({
                "ผลลัพธ์": [CLASS_LABELS.get(str(v), str(v)) for v in class_var.values],
                "ความน่าจะเป็น (%)": [f"{p * 100:.2f}" for p in probs],
            }),
            hide_index=True,
        )
        st.write("**ข้อมูลที่ใช้ประเมิน**")
        st.dataframe(
            pd.DataFrame({
                "ตัวแปร": [THAI_NAMES.get(v.name, v.name) for v in features],
                "ค่าที่กรอก": [user_values[v.name] for v in features],
            }),
            hide_index=True,
        )
else:
    st.info("กรอกข้อมูลด้านบนให้ครบ แล้วกดปุ่ม **ประเมินผล** เพื่อดูคำแนะนำ")
