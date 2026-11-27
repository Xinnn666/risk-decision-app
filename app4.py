import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="风险型决策智能体", layout="wide")
st.title("风险型决策（期望值准则）智能体")
st.caption("Risk Decision Agent · 多方案损益矩阵 · 期望值计算 · 决策树分析")

# ---------- 侧边栏：参数输入与配置 ----------
with st.sidebar:
    st.header("决策情境配置")
    num_states = st.number_input("自然状态数量", min_value=2, max_value=5, value=2, step=1)
    state_names = []
    state_probs = []
    
    st.markdown("---")
    st.subheader("自然状态及发生概率")
    for i in range(num_states):
        col1, col2 = st.columns([1, 1])
        with col1:
            name = st.text_input(f"状态 {i+1} 名称", value=f"状态{i+1}", key=f"sn_{i}")
            state_names.append(name)
        with col2:
            prob = st.slider(f"{name} 概率", 0.0, 1.0, float(1/num_states), step=0.05, key=f"sp_{i}")
            state_probs.append(prob)
            
    if sum(state_probs) != 1.0:
        st.warning(f"当前概率总和为 {sum(state_probs):.2f}，建议调整为 1.0")

    st.markdown("---")
    num_plans = st.number_input("备选方案数量", min_value=2, max_value=5, value=3, step=1)
    plan_names = [f"方案 {chr(65+i)}" for i in range(num_plans)]

# ---------- 主界面：损益矩阵输入 ----------
st.subheader("1. 方案损益矩阵输入 (单位: 万元)")
st.markdown("请填写每个方案在不同自然状态下的收益（亏损请填负数）：")

# 初始化一个全0的矩阵
default_data = np.zeros((num_plans, num_states))
df = pd.DataFrame(default_data, index=plan_names, columns=state_names)
# 为了让演示有数据，给个默认值
if num_plans == 3 and num_states == 2:
    df.iloc[0] = [100, -20]  # 方案A：高收益高风险
    df.iloc[1] = [50, 20]    # 方案B：中等
    df.iloc[2] = [10, 10]    # 方案C：保本

edited_df = st.data_editor(df, use_container_width=True)

# ---------- 核心计算层 ----------
# 计算期望货币值 (EMV) = 损益 * 概率 的求和
emvs = []
for plan in plan_names:
    emv = sum(edited_df.loc[plan, state_names[i]] * state_probs[i] for i in range(num_states))
    emvs.append(emv)

result_df = pd.DataFrame({
    "方案": plan_names,
    "期望收益 (EMV)": emvs
})

# ---------- 决策输出层 ----------
st.subheader("2. 智能体决策结果")
best_plan_idx = np.argmax(emvs)
best_plan = plan_names[best_plan_idx]
best_emv = emvs[best_plan_idx]

c1, c2 = st.columns(2)
c1.metric("最优决策方案", best_plan)
c2.metric("最大期望货币值", f"{best_emv:.2f} 万元")

if sum(state_probs) == 1.0:
    st.success(f"**决策建议**：根据期望值准则，推荐选择 **{best_plan}**。该方案在综合考虑各状态发生的概率后，长期平均收益最高。")
else:
    st.error("**警告**：自然状态概率之和不等于 1.0，当前期望值计算无效！请调整左侧概率。")

# 展示对比表格
st.markdown("#### 各方案期望收益对比")
st.dataframe(result_df.set_index("方案").style.format("{:.2f}"), use_container_width=True)

# ---------- 敏感性分析（智能预警） ----------
st.subheader("3. 决策敏感性分析（概率变动预警）")
st.markdown("如果自然状态发生的概率发生微小变化，最优方案会翻转吗？智能体自动进行模拟测试：")

# 简单扫描：假设第一个状态的概率从 0 到 1 变化，看最优方案怎么变
p1_range = np.linspace(0, 1, 50)
best_plans_over_p1 = []
for p1 in p1_range:
    # 保持其他状态概率同比例缩放
    if num_states == 2:
        current_probs = [p1, 1-p1]
    else:
        # 简化处理，多状态时只变动前两个，后续假设不变（为了可视化简化）
        current_probs = [p1, 1-p1] + [0]*(num_states-2) 
        
    current_emvs = [sum(edited_df.loc[plan, state_names[i]] * current_probs[i] for i in range(num_states)) for plan in plan_names]
    best_plans_over_p1.append(plan_names[np.argmax(current_emvs)])

# 检查最优方案是否发生过翻转
unique_best = list(dict.fromkeys(best_plans_over_p1))
if len(unique_best) > 1:
    st.warning(f"**敏感性警告**：当 **{state_names[0]}** 的发生概率发生变动时，最优方案在 {', '.join(unique_best)} 之间摇摆！这意味着该决策对概率的预测极其敏感，建议收集更多数据以精确估计概率。")
else:
    st.info(f"**稳健性提示**：在 **{state_names[0]}** 概率从 0 到 1 的测试范围内，**{best_plan}** 始终保持为最优方案，决策非常稳健。")

# ---------- 图表可视化 ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 期望收益对比图")
    fig_bar = go.Figure(data=[go.Bar(x=plan_names, y=emvs, marker_color=['#4C78A8' if p != best_plan else '#E45756' for p in plan_names])])
    fig_bar.update_layout(yaxis_title="期望收益 (万元)", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("#### 决策树逻辑图 (简图)")
    # 用简单的折线图模拟决策树结构
    fig_tree = go.Figure()
    # 根节点
    fig_tree.add_trace(go.Scatter(x=[0], y=[0.5], mode="markers+text", marker=dict(size=20, color="blue"), text=["决策"], textposition="bottom center", showlegend=False))
    # 方案节点
    y_plans = np.linspace(0, 1, num_plans)
    for i, y in enumerate(y_plans):
        fig_tree.add_trace(go.Scatter(x=[1], y=[y], mode="markers+text", marker=dict(size=15, color="orange"), text=[plan_names[i]], textposition="bottom center", showlegend=False))
        fig_tree.add_trace(go.Scatter(x=[0, 1], y=[0.5, y], mode="lines", line=dict(color="gray", width=1), showlegend=False))
        # 状态节点及连线的复杂度此处省略，用点表示
        fig_tree.add_trace(go.Scatter(x=[2], y=[y], mode="markers+text", marker=dict(size=10, color="green"), text=[f"EMV:{emvs[i]:.1f}"], textposition="middle right", showlegend=False))
        fig_tree.add_trace(go.Scatter(x=[1, 2], y=[y, y], mode="lines", line=dict(color="green", width=1), showlegend=False))
        
    fig_tree.update_layout(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False), 
                           yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), 
                           height=400, plot_bgcolor="white")
    st.plotly_chart(fig_tree, use_container_width=True)

st.caption("风险型决策智能体")
