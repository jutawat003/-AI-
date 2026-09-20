# -*- coding: utf-8 -*-
"""
app.py — เว็บแอป Streamlit สำหรับใช้งานโมเดลที่ฝึกจาก Orange Data Mining (*.pkcls)
รันด้วยคำสั่ง:  streamlit run app.py
"""

import os                      # ใช้ตรวจสอบว่ามีไฟล์โมเดลอยู่จริงหรือไม่
import joblib                  # ใช้โหลดไฟล์โมเดล .pkcls
import numpy as np             # ใช้จัดรูปแบบข้อมูลเป็นเมทริกซ์ก่อนส่งเข้าโมเดล
import pandas as pd            # ใช้แสดงตารางสรุปค่าที่ผู้ใช้กรอก
import streamlit as st         # ไลบรารีหลักสำหรับสร้างหน้าเว็บ
from Orange.data import Domain, Table   # โครงสร้างข้อมูลของ Orange (โมเดล .pkcls ต้องการ)

# ===================== 1) ตั้งค่าหน้าเว็บ =====================
st.set_page_config(page_title="AI แนะนำอาชีพเสริม", page_icon="💰", layout="centered")

# ---- รายชื่อโมเดลที่ให้ผู้ใช้เลือก (ชื่อที่แสดง : ชื่อไฟล์) ----
MODEL_FILES = {
    "Logistic Regression": "Logistic_Regression.pkcls",
    "Naive Bayes": "Naive_Bayes.pkcls",
    "Neural Network": "Neural_Network.pkcls",
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


# ===================== 2) ฟังก์ชันโหลดโมเดล =====================
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    """โหลดไฟล์ .pkcls ด้วย joblib และแคชไว้เพื่อไม่ต้องโหลดซ้ำทุกครั้งที่หน้าเว็บรีเฟรช"""
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def get_feature_stats(path: str):
    """
    ดึงค่าสถิติ (ต่ำสุด/สูงสุด/ค่าเฉลี่ย) ของแต่ละฟีเจอร์จากข้อมูลที่ใช้ฝึกโมเดล
    เพื่อนำมาตั้งเป็นค่าเริ่มต้นและขอบเขตของช่องกรอกให้สมเหตุสมผล
    """
    model = load_model(path)
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

# ---- Sidebar: ให้ผู้ใช้เลือกโมเดลเอง ----
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้ทำนาย", list(MODEL_FILES.keys()))
    model_path = MODEL_FILES[model_name]

# ---- ตรวจสอบว่ามีไฟล์โมเดลอยู่จริง ----
if not os.path.exists(model_path):
    st.error(f"ไม่พบไฟล์โมเดล «{model_path}» กรุณาวางไฟล์ .pkcls ไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# ---- โหลดโมเดลที่เลือก ----
model = load_model(model_path)
domain = model.original_domain          # โครงสร้างคอลัมน์ "ก่อนผ่าน preprocess" ของ Orange
features = list(domain.attributes)      # รายชื่อตัวแปรต้นทั้งหมดที่ใช้ตอนฝึก
class_var = domain.class_var            # ตัวแปรตาม (ผลลัพธ์ที่ทำนาย)
stats = get_feature_stats(model_path)

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
