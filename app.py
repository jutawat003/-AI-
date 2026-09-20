# -*- coding: utf-8 -*-
"""
app.py — เว็บแอป Streamlit สำหรับใช้งานโมเดลที่ฝึกจาก Orange Data Mining (*.pkcls)
รันด้วยคำสั่ง:  streamlit run app.py
"""

import io                      # ใช้รองรับกรณีผู้ใช้อัปโหลดไฟล์โมเดลเข้ามาเอง
from pathlib import Path        # ใช้หาตำแหน่งไฟล์โมเดลแบบอ้างอิงตำแหน่งของ app.py
import joblib                  # ใช้โหลดไฟล์โมเดล .pkcls
import numpy as np             # ใช้จัดรูปแบบข้อมูลเป็นเมทริกซ์ก่อนส่งเข้าโมเดล
import pandas as pd            # ใช้แสดงตารางสรุปค่าที่ผู้ใช้กรอก
import streamlit as st         # ไลบรารีหลักสำหรับสร้างหน้าเว็บ
from Orange.data import Domain, Table   # โครงสร้างข้อมูลของ Orange (โมเดล .pkcls ต้องการ)

# ===================== 1) ตั้งค่าหน้าเว็บ =====================
st.set_page_config(page_title="AI แนะนำอาชีพเสริม", page_icon="💰", layout="centered")

# ---- ชื่อที่อยากให้แสดงสวย ๆ แทนชื่อไฟล์ (ไม่ใส่ก็ได้ แอปจะใช้ชื่อไฟล์แทน) ----
PRETTY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "neural_network": "Neural Network",
}

# ---- ข้อความที่จะแสดงแทนค่าคลาส 0/1 ของตัวแปรตาม ----
CLASS_LABELS = {
    "0": "ไม่บรรลุเป้าหมายการออมเงิน",
    "1": "บรรลุเป้าหมายการออมเงิน",
}

# ---- ชื่อภาษาไทยของแต่ละคอลัมน์ (ถ้าไม่มีในนี้ จะใช้ชื่อคอลัมน์เดิม) ----
THAI_NAMES = {
    "Monthly_Income": "รายได้ต่อเดือน (บาท)",
    "Monthly_Expenditure": "รายจ่ายต่อเดือน (บาท)",
    "Market_Volatility_Index": "ดัชนีความผันผวนของตลาด",
    "Inflation_Rate": "อัตราเงินเฟ้อ (%)",
    "Investment_Amount": "เงินลงทุนสะสม (บาท)",
    "Savings_Ratio": "สัดส่วนการออม (0-1)",
    "Credit_Score": "คะแนนเครดิต",
    "Debt_to_Income_Ratio": "สัดส่วนหนี้สินต่อรายได้ (0-1)",
    "Risk_Tolerance_Level": "ระดับการยอมรับความเสี่ยง (0-1)",
    "Economic_Sentiment_Score": "ดัชนีความเชื่อมั่นทางเศรษฐกิจ (-1 ถึง 1)",
    "Investor_Confidence": "ความเชื่อมั่นของนักลงทุน (0-100)",
    "Financial_Stability_Index": "ดัชนีเสถียรภาพทางการเงิน (0-1)",
}


# ===================== 2) ฟังก์ชันค้นหา/โหลดโมเดล =====================
# โฟลเดอร์ที่ app.py อยู่ (สำคัญมากตอน deploy เพราะ working directory อาจไม่ใช่โฟลเดอร์นี้)
BASE_DIR = Path(__file__).parent.resolve()


def discover_models():
    """
    สแกนหาไฟล์โมเดลนามสกุล .pkcls ทุกไฟล์ในโปรเจกต์ (รวมโฟลเดอร์ย่อย เช่น models/)
    แล้วคืนค่าเป็น dict {ชื่อที่จะแสดงบนหน้าเว็บ: Path ของไฟล์}
    วิธีนี้ทำให้ไม่ต้องกังวลว่าชื่อไฟล์จะเป็นตัวพิมพ์เล็ก/ใหญ่หรือวางไว้โฟลเดอร์ไหน
    """
    found = {}
    for path in sorted(BASE_DIR.rglob("*.pkcls")):
        if ".git" in path.parts:          # ข้ามไฟล์ภายในของ git
            continue
        key = path.stem.lower()           # ชื่อไฟล์ (ไม่รวมนามสกุล) แบบพิมพ์เล็ก
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
    """
    ดึงค่าสถิติ (ต่ำสุด/สูงสุด/ค่าเฉลี่ย) ของแต่ละฟีเจอร์จากข้อมูลที่ใช้ฝึกโมเดล
    เพื่อนำมาตั้งเป็นค่าเริ่มต้นและขอบเขตของช่องกรอกให้สมเหตุสมผล
    (พารามิเตอร์ _model ขึ้นต้นด้วย _ เพื่อบอก Streamlit ว่าไม่ต้องนำไปคำนวณ hash)
    """
    model = _model
    stats = {}
    data = getattr(model, "instances", None)   # Orange เก็บข้อมูลฝึกไว้ใน .instances
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
st.title("โปรแกรม AI แนะนำอาชีพเสริมที่เหมาะสม")
st.caption("ทำนายว่าผู้ใช้จะ *บรรลุเป้าหมายการออมเงิน* หรือไม่ จากข้อมูลทางการเงินที่กรอก")

# ---- ค้นหาไฟล์โมเดลทั้งหมดที่มีอยู่จริงในโปรเจกต์ ----
available_models = discover_models()

if available_models:
    # เจอไฟล์โมเดล → ให้ผู้ใช้เลือกจากรายการที่มีอยู่จริง
    with st.sidebar:
        st.header("⚙️ ตั้งค่า")
        model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้ทำนาย", list(available_models.keys()))
    model_file = available_models[model_name]
    model = load_model_from_path(str(model_file))
    cache_key = str(model_file)
else:
    # ไม่เจอไฟล์เลย → แจ้งเตือน แสดงรายการไฟล์ที่มี และเปิดช่องให้อัปโหลดชั่วคราว
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
        st.stop()      # ยังไม่มีโมเดล → หยุดการทำงานของหน้าเว็บไว้ก่อน
    model_name = uploaded.name
    model = load_model_from_bytes(uploaded.getvalue())
    cache_key = uploaded.name

# ---- อ่านโครงสร้างคอลัมน์จากโมเดลที่โหลดได้ ----
domain = model.original_domain          # โครงสร้างคอลัมน์ "ก่อนผ่าน preprocess" ของ Orange
features = list(domain.attributes)      # รายชื่อตัวแปรต้นทั้งหมดที่ใช้ตอนฝึก
class_var = domain.class_var            # ตัวแปรตาม (ผลลัพธ์ที่ทำนาย)
stats = get_feature_stats(model, cache_key)

with st.sidebar:
    st.success(f"โหลดโมเดลสำเร็จ: **{model_name}**")
    st.write(f"จำนวนตัวแปรต้น: **{len(features)}** ตัว")
    st.write(f"ตัวแปรตาม: **{class_var.name}**")


# ===================== 4) สร้างช่องกรอกค่าตัวแปรต้น =====================
st.subheader("📝 กรอกข้อมูลของผู้ใช้")

user_values = {}   # เก็บค่าที่ผู้ใช้กรอก {ชื่อคอลัมน์: ค่า}
cols = st.columns(2)   # จัดช่องกรอกเป็น 2 คอลัมน์ให้ดูง่าย

for idx, var in enumerate(features):
    box = cols[idx % 2]                       # สลับซ้าย-ขวา
    label = THAI_NAMES.get(var.name, var.name)  # ชื่อภาษาไทย (ถ้ามี)

    if var.is_discrete:
        # --- คอลัมน์ชนิดข้อความ/หมวดหมู่ → ใช้ selectbox ---
        choice = box.selectbox(label, list(var.values), key=f"in_{var.name}")
        user_values[var.name] = choice
    else:
        # --- คอลัมน์ชนิดตัวเลข → ใช้ number_input ---
        s = stats.get(var.name, {})
        lo = s.get("min", 0.0)
        hi = s.get("max", 100.0)
        default = s.get("mean", (lo + hi) / 2)
        span = max(hi - lo, 1e-6)
        step = float(f"{span / 100:.1g}")      # ขนาดก้าวการปรับค่าให้เหมาะกับช่วงข้อมูล
        user_values[var.name] = box.number_input(
            label,
            min_value=float(lo - span),        # เผื่อช่วงให้กรอกนอกขอบเขตข้อมูลฝึกได้เล็กน้อย
            max_value=float(hi + span),
            value=float(default),
            step=step,
            format="%.3f",
            key=f"in_{var.name}",
            help=f"ช่วงข้อมูลที่ใช้ฝึก: {lo:,.3f} ถึง {hi:,.3f}",
        )


# ===================== 5) ปุ่มทำนายผล =====================
st.divider()

if st.button("🔮 ทำนายผล", type="primary"):

    # ---- 5.1 จัดรูปแบบข้อมูลให้ตรงกับตอนฝึกโมเดล ----
    # โมเดลจาก Orange ต้องการข้อมูลเรียงคอลัมน์ตาม domain.attributes เป๊ะ ๆ
    # โดยคอลัมน์ชนิดหมวดหมู่ต้องแปลงเป็น "ดัชนีของค่า" (Orange จะทำ one-hot /
    # discretize ที่เหลือให้เองผ่าน compute_value ตอน transform)
    row = []
    for var in features:
        v = user_values[var.name]
        if var.is_discrete:
            row.append(float(var.values.index(v)))   # ข้อความ → ตัวเลขดัชนี
        else:
            row.append(float(v))
    X = np.array([row], dtype=float)                 # รูปร่าง (1, จำนวนฟีเจอร์)

    # สร้าง Orange Table จาก domain เดิม (ไม่รวมตัวแปรตาม)
    # การส่งเป็น Table ทำให้ Orange แปลงข้อมูลตาม preprocess ของแต่ละโมเดลให้อัตโนมัติ
    table = Table.from_numpy(Domain(domain.attributes), X)

    # ---- 5.2 ส่งเข้าโมเดลเพื่อทำนาย ----
    pred_idx, probs = model(table, model.ValueProbs)   # คืนค่าทั้งคลาสที่ทำนายและความน่าจะเป็น
    pred_idx = int(pred_idx[0])
    probs = np.asarray(probs)[0]
    pred_label = class_var.values[pred_idx]                     # ค่าคลาสดิบ เช่น "0" / "1"
    pred_text = CLASS_LABELS.get(str(pred_label), str(pred_label))  # แปลงเป็นข้อความไทย
    confidence = float(probs[pred_idx])                         # ความน่าจะเป็นของคลาสที่ทำนาย

    # ---- 5.3 แสดงผลลัพธ์ ----
    st.subheader("📊 ผลการทำนาย")
    if str(pred_label) == "1":
        st.success(f"✅ ผลการทำนาย: **{pred_text}**")
        st.write("คำแนะนำ: แผนการเงินปัจจุบันมีแนวโน้มดี สามารถต่อยอดด้วยอาชีพเสริม "
                 "ที่ใช้ทุนต่ำ เช่น ฟรีแลนซ์ออนไลน์ หรือขายของออนไลน์ เพื่อเร่งการออมให้เร็วขึ้น")
    else:
        st.error(f"⚠️ ผลการทำนาย: **{pred_text}**")
        st.write("คำแนะนำ: ควรลดรายจ่ายหรือเพิ่มรายได้ เช่น รับงานเสริมนอกเวลา "
                 "งานสอนพิเศษ หรือขายสินค้าออนไลน์ เพื่อเพิ่มสัดส่วนการออม")

    # แสดงค่าความเชื่อมั่นเป็นตัวเลขและแถบสถานะ
    st.metric("ความน่าจะเป็นของผลลัพธ์ (Probability)", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    # ตารางความน่าจะเป็นของทุกคลาส
    st.write("**ความน่าจะเป็นแยกตามคลาส**")
    st.dataframe(
        pd.DataFrame({
            "ผลลัพธ์": [CLASS_LABELS.get(str(v), str(v)) for v in class_var.values],
            "ความน่าจะเป็น (%)": [f"{p * 100:.2f}" for p in probs],
        }),
        hide_index=True,
    )

    # ---- 5.4 สรุปค่าที่ผู้ใช้กรอก (ไว้ตรวจทาน) ----
    with st.expander("ดูข้อมูลที่ใช้ทำนาย"):
        st.dataframe(
            pd.DataFrame({
                "ตัวแปร": [THAI_NAMES.get(v.name, v.name) for v in features],
                "ค่าที่กรอก": [user_values[v.name] for v in features],
            }),
            hide_index=True,
        )
else:
    st.info("กรอกข้อมูลให้ครบแล้วกดปุ่ม **ทำนายผล** เพื่อดูผลลัพธ์")
