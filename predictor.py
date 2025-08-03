import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# 设置页面标题
st.title("Prediction of Cardiovascular Risk in New–onset T2D")
st.caption("Based on TyG Index and Carotid Ultrasound Features")

# ===== 加载模型和测试数据 =====
model = joblib.load('LGB.pkl')
X_test = pd.read_csv('x_test.csv')

# ===== 特征名称（统一顺序）=====
feature_names = [
    "Age (years)",
    "Hypertension",
    "IMT (mm)",
    "TyG index",
    "Carotid plaque burden",
    "Maximum plaque thickness (mm)"
]

# ===== 输入表单 =====
with st.form("input_form"):
    st.subheader("Please enter the following clinical and ultrasound features:")
    inputs = []

    for col in feature_names:
        if col == "Hypertension":
            inputs.append(st.selectbox(col, options=[0, 1], index=0))

        elif col == "Age (years)":
            min_val = int(X_test[col].min())
            max_val = 100
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "IMT (mm)":
            min_val = 0.0
            max_val = 1.5
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

        elif col == "TyG index":
            min_val = 0.0
            max_val = 15.0
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.01, format="%.2f")
            )

        elif col == "Carotid plaque burden":
            min_val = int(X_test[col].min())
            max_val = 15
            default_val = int(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=1)
            )

        elif col == "Maximum plaque thickness (mm)":
            min_val = 0.0
            max_val = 7.0
            default_val = float(X_test[col].median())
            inputs.append(
                st.number_input(col, value=default_val, min_value=min_val, max_value=max_val, step=0.1, format="%.2f")
            )

    submitted = st.form_submit_button("Submit Prediction")

# ===== 预测与解释 =====
if submitted:
    input_data = pd.DataFrame([inputs], columns=feature_names)
    input_data = input_data.round(2)
    st.subheader("Model Input Features")
    st.dataframe(input_data)

    # 保证传入模型的顺序一致
    model_input = pd.DataFrame([{
        "Age (years)": input_data["Age (years)"].iloc[0],
        "Hypertension": input_data["Hypertension"].iloc[0],
        "IMT (mm)": input_data["IMT (mm)"].iloc[0],
        "TyG index": input_data["TyG index"].iloc[0],
        "Carotid plaque burden": input_data["Carotid plaque burden"].iloc[0],
        "Maximum plaque thickness (mm)": input_data["Maximum plaque thickness (mm)"].iloc[0]
    }])

    # 模型预测概率
    predicted_proba = model.predict_proba(model_input)[0]
    probability = predicted_proba[1] * 100

    # 分层风险判断
    y_probs = model.predict_proba(X_test)[:, 1]
    low_threshold = np.percentile(y_probs, 50.0)
    mid_threshold = np.percentile(y_probs, 88.07)

    if predicted_proba[1] <= low_threshold:
        risk_level = "🟢 **You are currently at a low risk of cardiovascular disease.**"
        suggestion = "✅ Please continue to maintain a healthy lifestyle and attend regular follow-up visits."
    elif predicted_proba[1] <= mid_threshold:
        risk_level = "🟡 **You are at a moderate risk of cardiovascular disease.**"
        suggestion = "⚠️ It is advised to monitor your condition closely and consider preventive interventions."
    else:
        risk_level = "🔴 **You are at a high risk of cardiovascular disease.**"
        suggestion = "🚨 It is recommended to consult a physician promptly and take proactive medical measures."

    # 显示结果
    st.subheader("Prediction Result & Explanation")
    st.markdown(f"**Estimated probability:** {probability:.1f}%")
    st.info(risk_level)
    st.markdown(suggestion)

    # ===== SHAP 可解释性分析并保存图片 =====
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_input)

    if isinstance(shap_values, list):
        shap_value_sample = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_value_sample = shap_values
        expected_value = explainer.expected_value

    shap.force_plot(
        base_value=expected_value,
        shap_values=shap_value_sample,
        features=model_input,
        matplotlib=True,
        show=False
    )

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.close()
    st.image("shap_force_plot.png", caption="SHAP Force Plot", use_column_width=True)


