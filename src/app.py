import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
import io
import os

from simulator import closed_loop
from constants import MP

# Настройка страницы
st.set_page_config(layout="wide")
st.title("Diafiltration Control Demo")

# Проверка состояния
if "run" not in st.session_state:
    st.session_state.run = False

# Отображение картинки и описания проекта
if not st.session_state.run:
    # Путь до картинки
    image_path = os.path.join("assets", "tank_image.png")
    image = Image.open(image_path)

    # Конвертировать изображение в base64 для вставки в HTML
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Центрируем изображение и ограничиваем по ширине
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" style="max-width: 600px; width: 100%; height: auto;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        ## Project Overview
        This project simulates a protein purification process via diafiltration, 
        comparing open-loop and closed-loop (MPC) control.
    """)

    # Кнопка запуска симуляции
    if st.button("Start Simulation"):
        st.session_state.run = True
        st.rerun()

# Основной режим: MPC-симуляция
else:
    N = st.slider("Prediction Horizon N", 5, 50, 20)
    st.markdown("### Running Closed-loop MPC...")

    with st.spinner("Simulating..."):
        t, V, ML, u = closed_loop(N=N)
        cP = MP / V
        cL = ML / V

        # Построение графиков
        fig, ax = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
        ax[0].plot(t/3600, cP)
        ax[0].axhline(100, ls='--', color='k')
        ax[0].set_ylabel("$c_P$")

        ax[1].plot(t/3600, cL)
        ax[1].axhline(15, ls='--', color='k')
        ax[1].axhline(570, ls=':', color='r')
        ax[1].set_ylabel("$c_L$")

        ax[2].step(t[:len(u)]/3600, u, where='post')
        ax[2].set_ylabel("$u$")
        ax[2].set_xlabel("Time [h]")

        st.pyplot(fig)

        # Результаты симуляции
        st.success("✅ Simulation complete!")
        st.markdown("""
        The plots show how MPC achieves the target specifications while respecting lactose constraints:

        - Protein concentration rises to $c_P^* = 100$
        - Lactose falls below $c_L^* = 15$, without exceeding $c_L^{max} = 570$
        - Control actions $u$ vary over time to ensure target attainment
        """)
