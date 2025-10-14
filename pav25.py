
# 1. Importar librerías 
# =============================================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
from scipy import optimize
from scipy.stats import norm
from PIL import Image
import os
import base64
# =============================================================================================================
# 2. Configuración de Página y Estilos
# ============================================================================================================
st.set_page_config("Diseño de Pavimentos - UNAM", "🛣️", "wide", "expanded")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h1, h2, h3 { color: #1E3A8A; font-weight: 600; }
    .sidebar .sidebar-content { background-color: #E0E7FF; border-radius: 10px; padding: 20px; }
    .stSelectbox, .stNumberInput { border-radius: 8px; }
    .dataframe { font-size: 14px; border-radius: 10px; border: 1px solid #E2E8F0; }
    div[data-testid="metric-container"] {
        background-color: #EFF6FF; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    div[data-testid="metric-container"] > div { background-color: transparent; }
    div[data-testid="metric-container"] label { color: #1E3A8A; font-weight: 600; }
    .stContainer { border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); padding: 20px; margin-bottom: 20px; }
    .stButton>button { background-color: #3B82F6; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 500; }
    .stButton>button:hover { background-color: #2563EB; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #EFF6FF; border-radius: 8px 8px 0px 0px; padding: 10px 20px; border: none; }
    .stTabs [aria-selected="true"] { background-color: #3B82F6; color: white; }
</style>
""", unsafe_allow_html=True)
# =============================================================================================================
# Funciones Auxiliares
# =============================================================================================================

# 1. Calcular el factor carril de proyecto (fcp)
# =============================================================================================================
def calcular_fcp(nc): return 0.5 if nc == 1 else 0.45 if nc == 2 else 0.4

# 2. transformar vehículos a número de ejes, definiendo tipo de eje y sus cargas segun tipo de camino

def transformar_vehiculos_a_ejes(tc_nombre, params, cargados, vacios):

    A2, B2, B36, B38 = params["A2"], params["B2"], params["B36"], params["B38"]
    B4, C2, C36, C38 = params["B4"], params["C2"], params["C36"], params["C38"]
    C2R2, C3R2, C3R3 = params["C2R2"], params["C3R2"], params["C3R3"]
    C2R3, T2S1, T2S2 = params["C2R3"], params["T2S1"], params["T2S2"]
    T3S2, T3S3, T2S3 = params["T3S2"], params["T3S3"], params["T2S2"]
    T3S1, T2S1R2, T2S1R3 = params["T3S1"], params["T2S1R2"], params["T2S1R3"]
    T2S2R2, T3S1R2, T3S1R3 = params["T2S2R2"], params["T3S1R2"], params["T3S1R3"]
    T3S2R2, T3S2R4, T3S2R3 = params["T3S2R2"], params["T3S2R4"], params["T3S2R3"]
    T3S3S2, T2S2S2, T3S2S2 = params["T3S3S2"], params["T2S2S2"], params["T3S2S2"]   
    
    # Cargas por tipo de camino
    cargas = {
        "ET y A": [1.0, 6.5, 12.5, 10.0, 11.0, 11.0, 4.0, 7.0, 17.5, 21.0, 17.0, 19.0, 18.0, 4.5, 23.5, 26.5, 5.0],
        "Tipo B": [1.0, 6.0, 10.5, 9.5, 9.5, 10.5, 4.0, 7.0, 13.0, 17.0, 15.0, 15.0, 17.0, 4.5, 22.5, 22.5, 5.0],
        "Tipo C": [1.0, 5.5, 9.0, 8.0, 8.0, 9.0, 4.0, 7.0, 11.5, 14.5, 13.5, 13.5,14.5, 4.5, 20.0, 20.0, 5.0],
        "Tipo D": [1.0, 5.0, 8.0, 7.0, 7.0, 8.0, 4.0, 7.0, 11.0, 13.5, 12.0, 12.0, 13.5, 4.5, 18.0, 18.0, 5.0]
    }

    if tc_nombre not in cargas:
        raise ValueError(f"Tipo de camino '{tc_nombre}' no reconocido.")

    # Datos base
    data = {
        "Condición": ["Cargado", "Cargado", "Cargado",  "Cargado", "Cargado", "Cargado", "Vacío", "Vacío", 
                "Cargado", "Cargado", "Cargado", "Cargado", "Cargado","Vacío", 
                "Cargado", "Cargado", "Vacío"],
        "Descripción": ["Sencillo", "Sencillo", "Sencillo", "Sencillo", "Sencillo", "Sencillo", "Sencillo","Sencillo",
                "Tándem", "Tándem", "Tándem", "Tándem", "Tándem","Tándem",
                "Trídem", "Trídem", "Trídem" ],
        "Cargas (Ton)": cargas[tc_nombre]  # <- Esto debe coincidir en longitud con las listas anteriores
    }

    df = pd.DataFrame(data)

    # Fórmulas de transformación a ejes (1er Año)
    formulas = [
        lambda: 2 * A2 * (fvp + fvv),
        lambda: (100 - A2 + B4) * fvp,
        lambda: (B2 + C2 + T2S1 + T2S2 + T2S3 + T2S2S2) * fvp,
        lambda: (2*C2R2 + 2*C3R2 + C3R3 + C2R3 + 3*T2S1R2 + 2*T2S1R3 +
                2*T2S2R2 + 3*T3S1R2 + 2*T3S1R3 + 2*T3S2R2 + T3S2R3) * fvp,
        lambda: (T2S1 + T3S1) * fvp,
        lambda: (C2R2 + C2R3 + T2S1R2 + T2S1R3 + T2S2R2) * fvp,
        lambda: (B2 + B36 + B38 + 2*B4 + 2*C2 + C36 + C38 + 4*C2R2 + 3*C3R2 +
                2*C3R3 + 3*C2R3 + 3*T2S1 + 2*T2S2 + T3S2 + T3S3 + 2*T3S1 +
                5*T2S1R2 + 4*T2S1R3 + 4*T2S2R2 + 4*T3S1R2 + 3*T3S1R3 +
                3*T3S2R2 + T3S2R4 + 2*T3S2R3 + T3S3S2 + 2*T2S2S2 + T3S2S2) * fvv,
        lambda: (B2 + B36 + B38 + B4) * fvv,
        lambda: (B36 + B4 + C36 + T3S1R3) * fvp,
        lambda: (B38 + C38 + T3S2 + T3S3 + T3S1 + T3S2S2) * fvp,
        lambda: (C3R3 + C2R3 + T2S1R3 + T2S2R2 + T3S1R3 + T3S2R2 +
                3*T3S2R4 + 2*T3S2R3 + 2*T2S2S2 + 2*T3S2S2) * fvp,
        lambda: (T2S2 + T3S2 + T3S3S2) * fvp,
        lambda: (C3R2 + C3R3 + T3S1R2 + T3S2R2 + T3S2R4 + T3S2R3 + T3S3S2) * fvp,
        lambda: (C36 + C38 + C3R2 + 2*C3R3 + C2R3 + T2S2 + 2*T3S2 + T3S3 + T3S1 +
                T2S1R3 + T2S2R2 + T3S1R2 + 2*T3S1R3 + 2*T3S2R2 + 4*T3S2R4 +
                3*T3S2R3 + 2*T3S3S2 + 2*T2S2S2 + 3*T3S2S2) * fvv,
        lambda: T3S3S2 * fvp,
        lambda: (T3S3 + T2S3) * fvp,
        lambda: T3S3S2 * fvv
    ]


    # Calcular ejes
    df["Ejes 1er Año"] = [formula() for formula in formulas]

    # Convertir toneladas a kips
    df["Cargas (Kip)"] = df["Cargas (Ton)"] * 2.2046226218517

    # Orden final
    df = df[["Descripción", "Condición", "Cargas (Ton)", "Cargas (Kip)", "Ejes 1er Año"]]

    # Asegurar tipos
    df = df.astype({
        "Descripción": str,
        "Condición": str,
        "Cargas (Ton)": float,
        "Cargas (Kip)": float,
        "Ejes 1er Año": float
    })

    return df

# 3. Para calcular los ESAL'S en función de la Z

def esals(Z):
    """
    Calcula los ESAL's acumulados en la vida de proyecto a partir de la profundidad Z (cm).
    No muestra DataFrame ni resultados intermedios.
    """
    # Cálculo del esfuerzo vertical de un eje estándar
    sigma_z_st = 5.8 * (1 - (Z**3) / ((15**2 + Z**2)**(1.5)))

    # Transformar vehículos a ejes
    df = transformar_vehiculos_a_ejes(tc_nombre, params, fvp, fvv)
    df["Radio placa"] = np.nan

    # Cálculo del radio de placa
    for i in range(8):
        P = df.loc[i, "Cargas (Ton)"]
        q = 2 if i == 0 else 6
        df.loc[i, "Radio placa"] = np.sqrt((1000 * P) / (2 * np.pi * q))

    for i in range(8, 14):
        P = df.loc[i, "Cargas (Ton)"]
        q = 6
        if Z < 30:
            radio_placa = np.sqrt((1000 * P) / (4 * np.pi * q))
        else:
            radio_placa = np.sqrt((1111 * P) / (4 * np.pi * q))
        df.loc[i, "Radio placa"] = radio_placa

    for i in range(14, 17):
        P = df.loc[i, "Cargas (Ton)"]
        q = 6
        if Z < 30:
            radio_placa = np.sqrt((1000 * P) / (6 * np.pi * q))
        else:
            radio_placa = np.sqrt((1333 * P) / (6 * np.pi * q))
        df.loc[i, "Radio placa"] = radio_placa

    # Cálculo del esfuerzo vertical para cada fila
    esfuerzo_vert = []
    for i, row in df.iterrows():
        a = row['Radio placa']
        q = 2 if i == 0 else 6
        numerador = Z**3
        denominador = (a**2 + Z**2)**(1.5)
        sigma_z = q * (1 - (numerador / denominador))
        esfuerzo_vert.append(sigma_z)

    df["Esfuerzo vert."] = esfuerzo_vert

    # Cálculo del daño unitario
    daño_unitario = []
    for i, row in df.iterrows():
        sigma_z_i = row['Esfuerzo vert.']
        if i <= 7:
            N = 1
        elif 8 <= i <= 13:
            N = 2 if Z < 30 else 1
        else:
            N = 3 if Z < 30 else 1

        d = (10 ** ((np.log10(sigma_z_i) - np.log10(sigma_z_st)) / np.log10(1.5))) * N
        daño_unitario.append(d)

    df["Daño unitario"] = daño_unitario

    # Ejes equivalentes del primer año
    df["Ejes Equivalentes"] = df["Ejes 1er Año"] * df["Daño unitario"]
    total_ejes_equivalentes = df["Ejes Equivalentes"].sum()

    # Factor de crecimiento de tráfico CT
    if tca != 0:
        CT = ((1 + (tca/100)) ** vida - 1) / (tca/100)
    else:
        CT = vida

    # Cálculo final de los ESAL's acumulados
    ESALs = CT * total_ejes_equivalentes

    return ESALs

def calcular_CT(tca, vida):
    if tca != 0:  # Evita la división por cero
        CT = ((1 + (tca / 100)) ** vida - 1) / (tca / 100)
    else:
        CT = vida  # Si tca es 0, el crecimiento es lineal, CT = vida

    return CT
# Función para codificar la imagen en base64
def cargar_imagen_base64(ruta):
    with open(ruta, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ============================================================================================================ f2
# Título principal con ícono
st.markdown("<h3 style='text-align: center;'>🛣️ Análisis y diseño de pavimentos Método UNAM </h3>", unsafe_allow_html=True)         

# Crear contenedor principal
main_container = st.container()
# =============================================================================================================
# 3. Sidebar -Diccionarios de Opciones y Entradas de Usuario
# =============================================================================================================
# CSS para modificar el ancho del sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 300px;   /* Ancho mínimo */
            max-width: 300px;   /* Ancho máximo */
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    #st.markdown("# 🚗 Datos Generales")
    st.markdown("<h1 style='text-align: center;'>🚗 Datos generales</h1>", unsafe_allow_html=True)
    # Diccionario para mapear opciones a valores numéricos
    opciones_camino = {"ET y A": 1, "Tipo B": 2, "Tipo C": 3, "Tipo D": 4}
    opciones_ncarriles = {"Un carril por sentido": 1, "Dos carriles por sentido": 2, "Tres o más carriles por sentido": 3}

    tc_nombre = st.selectbox("Camino Tipo", list(opciones_camino.keys()))
    tc = opciones_camino[tc_nombre]
    nc_nombre = st.selectbox("No.Carriles x S.C.", list(opciones_ncarriles.keys()))
    nc = opciones_ncarriles[nc_nombre]     
    vc = float(st.text_input("Veh.cargados(%)", value="80", key="vc_text"))
    vida = float(st.text_input("Vida útil años", value="15", key="vida_text"))  
    tca = float(st.text_input("Tas.crec.anual(%)", value="3.5", key="tca_text"))
    tdpa = float(st.text_input("TDPA ambos Sc", value="7500", key="tdpa_text"))
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #718096; font-size: 12px;'>
        <p>Desarrollado Por | M. en I. Martín Olvera Corona</p>
        <p>tel 961-6622-614<p>
        <p>🛣️ OlverPav UNAM  Versión 1.0 - 2025</p>
    </div>
    """, unsafe_allow_html=True)




# =============================================================================================================
# 5. Cálculos Base
# =============================================================================================================

fcp = calcular_fcp(nc)
vcp = tdpa * fcp  # TDPA en el carril de proyecto
fvp = (vcp * 3.65 * vc) / 100  # Vehículos cargados
fvv = (vcp * 3.65 * (100 - vc)) / 100  # Vehículos vacíos

# =============================================================================================================
# 7. Contenido Principal - Tabs
# =============================================================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Composición vehicular",
    "Transforma veh's a ejes", 
    "Definición de espesores",
    "Solo ejes equivalentes", 
    "Ayuda y guía de apoyo",
    "Memoria de cálculo " 
   
])
# ============================================================================================================ 

with tab1:
    st.markdown("<h2 style='text-align: center;'>🚛 Composición vehicular (%)</h2>", unsafe_allow_html=True)
    # Crear dos columnas para los campos
    col1, col2, col3, col4, col5, = st.columns(5)
    
    with col1:
        A2 = float(st.text_input("**:red[A2]**", value="85", key="a2_text"))      
        B2 = float(st.text_input("**:red[B2]**", value="2", key="b2_text"))
        B36 = float(st.text_input("B3 6 llantas ", value="0", key="b36_text"))
        B38 = float(st.text_input("B3 8 llantas ", value="0", key="b38_text"))
        B4 = float(st.text_input("B4 ", value="0", key="b4_text"))
        C2 = float(st.text_input("**:red[C2]**", value="2", key="c2_text"))
          
        
    with col2:
        
        C36 = float(st.text_input("C3 6 llantas", value="0", key="c36_text"))
        C38 = float(st.text_input("**:red[C3 8 llantas]**", value="2", key="c38_text"))
        C2R2 = float(st.text_input("C2R2 ", value="0", key="c2r2_text"))
        C3R2 = float(st.text_input("C3R2 ", value="0", key="c3r2_text"))
        C3R3 = float(st.text_input("C3R3 ", value="0", key="c3r3_text"))
        C2R3 = float(st.text_input("C2R3 ", value="0", key="c2r3_text"))
             
        
        
    with col3:

        T2S1 = float(st.text_input("T2S1 ", value="0", key="t2s1_text"))
        T2S2 = float(st.text_input("T2S2 ", value="0", key="t2s2_text"))
        T3S2 = float(st.text_input("**:red[T3S2]**", value="2", key="t3s2_text")) 
        T3S3 = float(st.text_input("**:red[T3S3]**", value="5", key="t3s3_text"))
        T2S3 = float(st.text_input("T2S3 ", value="0", key="t2s3_text"))
        T3S1 = float(st.text_input("T3S1 ", value="0", key="t3s1_text"))
        
    with col4:

        T2S1R2 = float(st.text_input("T2S1R2 ", value="0", key="t2s1r2_text"))
        T2S1R3 = float(st.text_input("T2S1R3 ", value="0", key="t2s1r3_text"))
        T2S2R2 = float(st.text_input("T2S2R2 ", value="0", key="t2s2r2_text"))
        T3S1R2 = float(st.text_input("T3S1R2 ", value="0", key="t3s1r2_text"))
        T3S1R3 = float(st.text_input("T3S1R3 ", value="0", key="t3s1r3_text"))
        T3S2R2 = float(st.text_input("T3S2R2 ", value="0", key="t3s2r2_text"))        

    with col5:

        T3S2R4 = float(st.text_input("**:red[T3S2R4]**", value="2", key="t3s2r4_text"))
        T3S2R3 = float(st.text_input("T3S2R3 ", value="0", key="t3s2r3_text"))
        T3S3S2 = float(st.text_input("T3S3S2 ", value="0", key="t3s3s2_text"))
        T2S2S2 = float(st.text_input("T2S2S2 ", value="0", key="t2s2s2_text"))
        T3S2S2 = float(st.text_input("T3S2S2 ", value="0", key="t3s2s2_text"))

    params = {
        "A2": A2, "B2": B2, "B36": B36, "B38": B38, "B4": B4, "C2": C2,
        "C36": C36, "C38": C38, "C2R2": C2R2, "C3R2": C3R2, "C3R3": C3R3, "C2R3": C2R3,
        "T2S1": T2S1, "T2S2": T2S2, "T3S2": T3S2, "T3S3": T3S3, "T2S3": T2S3, "T3S1": T3S1,
        "T2S1R2": T2S1R2, "T2S1R3": T2S1R3, "T2S2R2": T2S2R2, "T3S1R2": T3S1R2, "T3S1R3": T3S1R3,
        "T3S2R2": T3S2R2,  "T3S2R4": T3S2R4, "T3S2R3": T3S2R3, "T3S3S2": T3S3S2, "T2S2S2": T2S2S2, "T3S2S2": T3S2S2 
    }

    suma_acumulada = sum(params.values())

    # Mostrar la suma acumulada en el sidebar con indicador visual
    #st.progress(min(suma_acumulada/100, 1.0))
    if suma_acumulada == 100:
        st.success(f"**Suma:** {suma_acumulada:.1f}% ✓")
    else:
        st.warning(f"**Suma:** {suma_acumulada:.1f}% (debe ser 100%)")

with tab2:
    df_ejes = transformar_vehiculos_a_ejes(tc_nombre, params, fvp, fvv)

    st.dataframe(
        df_ejes.style.format({
            "Cargas (Ton)": "{:.2f}",
            "Cargas (Kip)": "{:.2f}",
            "Ejes 1er Año": "{:,.0f}"
        }).set_properties(**{
            "text-align": "center",
            "border": "1px solid #E2E8F0",
            "padding": "8px"
        }).set_table_styles([
            {"selector": "th", "props": [("background-color", "#3B82F6"), ("color", "white"), ("font-weight", "bold")]}
        ]),
        height=650
    )

with tab3:
    # Método UNAM 
    col1, col2, col3 = st.columns(3)
    with col1:
        qu = float(st.text_input("Nivel de confianza %", value="90", key="qu_text"))
        Qu = qu/100        

    with col2:
        # Cálculo de T
        T = np.sqrt(np.log(1 / ((1 - Qu) ** 2)))
        st.latex(fr"T = \sqrt{{ \ln \left( \frac{{1}}{{(1 - {Qu})^2}} \right) }} = {T:.4f}")
    with col3:   
        # Título centrado
        st.markdown(
            "<div style='text-align: center; font-size:16px; font-weight:600;'>Constantes distribución normal:</div>",
            unsafe_allow_html=True
        )

        # Constantes
        c1, c2, c3 = 2.515517, 0.802853, 0.010328
        c4, c5, c6 = 1.432788, 0.189269, 0.001308

        # Mostrarlas en 3 renglones de 2 columnas
        st.markdown(
            f"<div style='text-align: center;'>C1 = {c1:.6f} &nbsp;&nbsp;&nbsp;&nbsp; C2 = {c2:.6f}</div>",
         unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center;'>C3 = {c3:.6f} &nbsp;&nbsp;&nbsp;&nbsp; C4 = {c4:.6f}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div style='text-align: center;'>C5 = {c5:.6f} &nbsp;&nbsp;&nbsp;&nbsp; C6 = {c6:.6f}</div>",
            unsafe_allow_html=True
        )
        # 👇 Línea en blanco como separación
        st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Título centrado
        st.markdown(
            "<div style='text-align: center; font-size:16px; font-weight:600;'>Abscisa nivel de confianza:</div>",
            unsafe_allow_html=True
        )

        # Cálculo de U
        numerador_U = c1 + c2 * T + c3 * T**2
        denominador_U = 1 + c4 * T + c5 * T**2 + c6 * T**3
        U = T - (numerador_U / denominador_U)

        # Renglón 1: Fórmula general
        st.latex(r"U = T - \frac{C_1 + C_2 T + C_3 T^2}{1 + C_4 T + C_5 T^2 + C_6 T^3}")

        # Renglón 2: Resultado numérico centrado
        st.markdown(
            fr"<div style='text-align: center; font-size:18px;'>U = {U:.4f}</div>",
            unsafe_allow_html=True
        )
    with col2:
        # Título centrado
        st.markdown(
            "<div style='text-align: center; font-size:16px; font-weight:600;'>Constante experimental:</div>",
            unsafe_allow_html=True
        )

        # Cálculo de B1 y B2
        B1 = 0.8477 + 0.12 * U      # Para Bases
        B2 = 0.4547 + 0.1593 * U    # Para subbase e inferiores

        # Fórmulas simbólicas
        st.latex(r"B_1 = 0.8477 + 0.12 \cdot U")
        st.markdown(
            fr"<div style='text-align: center; font-size:18px;'>Para bases ➞ B₁ = {B1:.4f}</div>",
            unsafe_allow_html=True
        )

        VRS01 = 10 ** B1
        st.latex(fr"VRS_0 = 10^{{B_1}} = {VRS01:.4f}")

    with col3:
        st.markdown(
            "<div style='text-align: center; font-size:16px; font-weight:600;'>Para Subbases y terracerías:</div>",
            unsafe_allow_html=True
        )

        st.latex(r"B_2 = 0.4547 + 0.1593 \cdot U")
        st.markdown(
            fr"<div style='text-align: center; font-size:18px;'>SBB y SBR's ➞ B₂ = {B2:.4f}</div>",
            unsafe_allow_html=True
        )

        VRS02 = 10 ** B2
        st.latex(fr"VRS_0 = 10^{{B_2}} = {VRS02:.4f}")
    # Configurar las columnas (más angostas)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
# ============================================================================================================ t2 Col1
    with col1:
        st.markdown("### Ingrese el CBR(%) ➡️")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.caption("Carpeta asfáltica (cm)")
        D1 = st.number_input(
            " ", min_value=0.0, max_value=50.0, value=5.0, step=1.0,
            key="D1", label_visibility="collapsed"
        )
        
        st.markdown("### Profundidad de daño Z(cm)")
        st.markdown("### Ejes equivalentes ∑L(Zi)👉")
        st.markdown(
        "<div style='text-align: center; font-size:16px; font-weight:600;'>Factor de influencia Boussinesq:</div>",
        unsafe_allow_html=True
        )
        st.latex(r"f_z = \frac{VRS_z}{VRS_0 \cdot 1.5^{\log(\sum L)}}")
        st.markdown(
        "<div style='text-align: center; font-size:16px; font-weight:600;'>Espesor en grava equivalente requerido:</div>",
        unsafe_allow_html=True
        )
        st.latex(r"Z_G = \frac{15}{\sqrt{ \dfrac{1}{(1 - f_z)^{2/3}} - 1 }}")
        st.markdown(
        "<div style='text-align: center; font-size:16px; font-weight:600;'>Espesor en grava equivalente real:</div>",
        unsafe_allow_html=True
        )
        st.latex(r"ZG_{\text{REAL}} = a_1 D_1 + a_2 D_2 + \dots + a_n D_n")
    with col2:
        vrs1 = float(st.text_input("CBR Base hidráulica", value="80", key="vrs1_text"))        
        st.caption("Base asfáltica (cm)")
        D2 = st.number_input(
            " ", min_value=0.0, max_value=50.0, value=5.0, step=1.0,
            key="D2", label_visibility="collapsed"
        )
        Prof1 = D1 + D2
        st.latex(fr"Z_1 = {Prof1:.0f}")
        Esal1 = esals(Prof1)
        st.latex(fr"\sum L(Z_1) = {Esal1:,.0f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo de fz
        fz1 = vrs1 / ((VRS01 * (1.5) ** (np.log10(Esal1))))
        st.latex(fr"fz_1 = {fz1:.4f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo final de Z
        Zg1 = 15 / np.sqrt((1/(1-fz1)**(2/3))-1)
        st.latex(fr"ZG_1 = {Zg1:.0f}")        
        zge1 = (D1*2)+(D2*1.5)
        st.markdown("&nbsp;", unsafe_allow_html=True)
                 
        st.latex(fr"ZG1_{{\text{{REAL}}}} = {zge1:.0f}")

        if zge1 >= Zg1:
            st.markdown("<div style='text-align: center; font-size:18px; color: green;'>✅ Cumple</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; font-size:18px; color: red;'>❌ No cumple</div>", unsafe_allow_html=True)

    with col3:
        
        vrs2 = float(st.text_input("CBR Subbase hidráulica", value="30", key="vrs2_text"))
        
        st.caption("Base hidráulica (cm)")
        D3 = st.number_input(
            " ", min_value=0.0, max_value=50.0, value=15.0, step=1.0,
            key="D3", label_visibility="collapsed"
        )
        zge2 = (D1*2) + (D2*1.5) + D3
        Prof2 = D1 + D2 + D3
        st.latex(fr"Z_2 = {Prof2:.0f}")
        Esal2 = esals(Prof2)
        st.latex(fr"\sum L(Z_2) = {Esal2:,.0f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo de fz
        fz2 = vrs2 / ((VRS01 * (1.5) ** (np.log10(Esal2))))
        st.latex(fr"fz_2 = {fz2:.4f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo final de Z
        Zg2 = 15 / np.sqrt((1/(1-fz2)**(2/3))-1)
        st.latex(fr"ZG_2 = {Zg2:.0f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
                 
        st.latex(fr"ZG2_{{\text{{REAL}}}} = {zge2:.0f}")

        if zge2 >= Zg2:
            st.markdown("<div style='text-align: center; font-size:18px; color: green;'>✅ Cumple</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; font-size:18px; color: red;'>❌ No cumple</div>", unsafe_allow_html=True)

    with col4:
        
        vrs3 = float(st.text_input("CBR Subrasante", value="5", key="vrs3_text"))
       
        st.caption("Subbase hidráulica (cm)")
        D4 = st.number_input(
            " ", min_value=0.0, max_value=50.0, value=15.0, step=1.0,
            key="D4", label_visibility="collapsed"
        )
        Prof3 = D1 + D2 + D3 + D4
        st.latex(fr"Z_3 = {Prof3:.0f}")
        Esal3 = esals(Prof3)
        st.latex(fr"\sum L(Z_3) = {Esal3:,.0f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo de fz
        fz3 = vrs3 / ((VRS02 * (1.5) ** (np.log10(Esal3))))
        st.latex(fr"fz_3 = {fz3:.4f}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        # Cálculo final de Z
        Zg3 = 15 / np.sqrt((1/(1-fz3)**(2/3))-1)
        st.latex(fr"ZG_3 = {Zg3:.0f}")
        zge3 = (D1*2) + (D2*1.5) + D3 + D4
        st.markdown("&nbsp;", unsafe_allow_html=True)        
                         
        st.latex(fr"ZG3_{{\text{{REAL}}}} = {zge3:.0f}")

        if zge3 >= Zg3:
            st.markdown("<div style='text-align: center; font-size:18px; color: green;'>✅ Cumple</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; font-size:18px; color: red;'>❌ No cumple</div>", unsafe_allow_html=True)
with tab4:
    # Solo ejes equivalentes
    
    # Ingreso de la profundidad de daño Z (profz) con input numérico
    st.markdown("### 📏 Profundidad de daño")
    Z = float(st.text_input("Z (cm)", value="5", key="Z_text"))    
    # Cálculo del esfuerzo vertical de un eje estandar
    sigma_z_st = 5.8 * (1 - (Z**3) / ((15**2 + Z**2)**(3/2)))
    # Clonar el DataFrame para trabajar en esta pestaña
    #df_tab2 = df.copy()
    df_tab2 = transformar_vehiculos_a_ejes(tc_nombre, params, fvp, fvv)
    # Crear la nueva columna "Radio placa" inicializada con NaN
    df_tab2["Radio placa"] = np.nan
    # Llenar las filas 0 a 6
    # Llenar las filas 0 a 6
    for i in range(8):  # De la fila 0 a la 6 inclusive
        P = df_tab2.loc[i, "Cargas (Ton)"]
        q = 2 if i == 0 else 6  # q=2 para la fila 0, q=6 para las demás
        radio_placa = np.sqrt((1000 * P) / (2 * np.pi * q))
        df_tab2.loc[i, "Radio placa"] = radio_placa
    for i in range(8, 14):  # De la fila 7 a la 16 inclusive
        P = df_tab2.loc[i, "Cargas (Ton)"]
        q = 6  # q = 6 en todas estas filas
        if Z < 30:
            radio_placa = np.sqrt((1000 * P) / (4 * np.pi * q))
        else:
            radio_placa = np.sqrt((1111 * P) / (4 * np.pi * q))
        df_tab2.loc[i, "Radio placa"] = radio_placa
    for i in range(14, 17):  
        P = df_tab2.loc[i, "Cargas (Ton)"]
        q = 6
        if Z < 30:
            radio_placa = np.sqrt((1000 * P) / (6 * np.pi * q))
        else:
            radio_placa = np.sqrt((1333 * P) / (6 * np.pi * q))
        df_tab2.loc[i, "Radio placa"] = radio_placa
    # Crear una nueva columna "Esfuerzo vert."
    esfuerzo_vert = []
     
    for i, row in df_tab2.iterrows():
        a = row['Radio placa']  # Tomamos 'a' de la columna
        if i == 0:
            q = 2
        else:
            q = 6

        numerador = Z**3
        denominador = (a**2 + Z**2)**(1.5)
                
        sigma_z = q * (1 - (numerador / denominador))
        esfuerzo_vert.append(sigma_z)
    
    # Asignar la columna al DataFrame
    df_tab2["Esfuerzo vert."] = esfuerzo_vert
    # Crear lista para el daño unitario
    daño_unitario = []
    for i, row in df_tab2.iterrows():
        sigma_z_i = row['Esfuerzo vert.']  # Tomamos el esfuerzo vertical de la fila
                
        # Definir el valor de N dependiendo de la fila y de Z_value
        if i <= 7:
            N = 1
        elif 8 <= i <= 13:
            N = 2 if Z < 30 else 1
        else:
            N = 3 if Z < 30 else 1

        # Calcular daño unitario usando la fórmula dada
        d = (10 ** ((np.log10(sigma_z_i) - np.log10(sigma_z_st)) / np.log10(1.5)))*N

        daño_unitario.append(d)
    # Asignar la columna al DataFrame
    df_tab2["Daño unitario"] = daño_unitario
    # Crear nueva columna "Ejes Equivalentes"
    df_tab2["Ejes Equivalentes"] = df_tab2["Ejes 1er Año"] * df_tab2["Daño unitario"]
    # Primero, configuramos el botón de Mostrar/Ocultar

    # Inicializar variable de sesión
    if 'mostrar_tabla' not in st.session_state:
        st.session_state.mostrar_tabla = False

    # Botón tipo texto
    if st.button('📄 Mostrar/Ocultar ejes primer año'):
        st.session_state.mostrar_tabla = not st.session_state.mostrar_tabla

    # Si el usuario decidió mostrar la tabla
    if st.session_state.mostrar_tabla:
        # Filtrar filas donde "Ejes 1er Año" sea diferente de cero
        df_filtrado = df_tab2[df_tab2["Ejes 1er Año"] != 0]

        # Mostrar el DataFrame actualizado
        st.dataframe(
            df_filtrado.style.format({
                "Cargas (Ton)": "{:.2f}",
                "Cargas (Kip)": "{:.2f}",
                "Ejes 1er Año": "{:,.0f}",
                "Radio placa": "{:.2f}",
                "Esfuerzo vert.": "{:.2f}",
                "Daño unitario": "{:.5f}",
                "Ejes Equivalentes": "{:,.0f}"
            }).set_properties(**{
                "text-align": "center",
                "border": "1px solid #E2E8F0",
                "padding": "8px"
        }).set_table_styles([
            {"selector": "th", "props": [("background-color", "#3B82F6"), ("color", "white"), ("font-weight", "bold")]}
        ]),
        height=450
    )    
                
        #use_container_width=True
    #)
    # Calcular y mostrar la suma total de "Ejes Equivalentes"
    total_ejes_equivalentes = df_tab2["Ejes Equivalentes"].sum()

    st.markdown(f"""
    <div style='text-align: left; font-size: 24px; font-weight: bold; color: #3B82F6;'>
        Total de Ejes Equivalentes 1er año: {total_ejes_equivalentes:,.0f}
    </div>
    """, unsafe_allow_html=True)
    # Cálculo de CT
    CT = calcular_CT(tca, vida)    
    ESALs = CT * total_ejes_equivalentes

    # Mostrar el resultado
    st.markdown("### 🚛 ESAL'S acumulados en la vida de proyecto")
    st.metric("ESAL's en la vida útil", f"{ESALs:,.2f}")

    with tab5:
        # Ayuda tutorial con el método   

        # Parámetros
        IMAGE_FOLDER = "imagen"
        TOTAL_IMGS = 12
        image_files = [f"unam_{i}.png" for i in range(TOTAL_IMGS)]

        # Estado de navegación
        if "img_index" not in st.session_state:
            st.session_state.img_index = 0
        mostrar_ayuda = st.checkbox("🔍 Ayuda", value=False)
        # Botón de descarga del PDF en el sidebar
        with open("guia_unam.pdf", "rb") as pdf_file:
            st.download_button(
                label="📥 Descargar guía (PDF)",
                data=pdf_file,
             file_name="guia_unam.pdf",
            mime="application/pdf"
        )
        # Funciones de navegación
        def ir_al_inicio():
            st.session_state.img_index = 0

        def ir_al_final():
            st.session_state.img_index = TOTAL_IMGS - 1

        def ir_atras():
            if st.session_state.img_index > 0:
                st.session_state.img_index -= 1

        def ir_adelante():
            if st.session_state.img_index < TOTAL_IMGS - 1:
                st.session_state.img_index += 1

        # Visor de ayuda
        if mostrar_ayuda:
    

         # Botones arriba
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.button("⏮ Inicio", on_click=ir_al_inicio)
            with col2:
                st.button("◀ Atrás", on_click=ir_atras)
            with col3:
                st.button("▶ Adelante", on_click=ir_adelante)
            with col4:
                st.button("⏭ Fin", on_click=ir_al_final)

            # Mostrar imagen centrada con estilo limitado
            current_file = os.path.join(IMAGE_FOLDER, image_files[st.session_state.img_index])
            with open(current_file, "rb") as file:
                img_bytes = file.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <img src="data:image/png;base64,{img_base64}"
                            style="max-width:100%; max-height:80vh; object-fit:contain;"/>
                        <p style="margin-top:10px;">Imagen {st.session_state.img_index + 1} de {TOTAL_IMGS}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    with tab6:
         # Título centrado
        st.markdown(
            "<div style='text-align: center; font-size:20px; font-weight:600;'>Memoria de cálculo para el diseño del pavimento por el método UNAM:</div>",
            unsafe_allow_html=True
        )
        # 👇 Línea en blanco como separación
       

        col1, col2 = st.columns(2)
        with col1:
            nombreVia = st.text_input("Carretera", value="Tuxtla Gutiérrez - San Cristóbal", key="nombreVia_text")
            kminicio = st.text_input("De km", value="52+000", key="kminicio_text")
        with col2:
            tramo = st.text_input("Tramo", value="Escopetazo - San Cristóbal", key="tramo_text")
            kmfin = st.text_input("De km", value="79+650", key="kmfin_text")
        st.markdown(
            "<div style='text-align: left; font-size:20px; font-weight:600;'>A) Datos generales:</div>",
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>1. Clasificación oficial RPyD :&nbsp;&nbsp;&nbsp;{tc_nombre}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>2. No. de carriles por sentido:&nbsp;&nbsp;&nbsp;{nc:,.0f}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>3. % de vehículos cargados    :&nbsp;&nbsp;&nbsp;&nbsp;{vc:,.0f}</div>",
            unsafe_allow_html=True
        )      
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>4. Vida útil o periodo años :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{vida:,.0f}</div>",
            unsafe_allow_html=True
        )      

            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>5. Tasa crecimiento anual % :&nbsp;&nbsp;{tca}</div>",
            unsafe_allow_html=True
        )      

            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>6. TDPA en ambos sentidos :&nbsp;&nbsp;&nbsp;{tdpa:,.0f}</div>",
            unsafe_allow_html=True
        )     
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>7. Nivel de confianza % :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{qu}</div>",
            unsafe_allow_html=True
        )       
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
            "<div style='text-align: left; font-size:20px; font-weight:600;'>B) Composición vehicular:</div>",
            unsafe_allow_html=True
        )

            # Filtrar vehículos con valor > 0
            vehiculos_filtrados = [(v, val) for v, val in params.items() if val > 0]

            # Crear tres columnas internas
            subcol1, subcol2, subcol3 = st.columns(3)

            # Dividir la lista en 3 partes balanceadas
            divisiones = np.array_split(vehiculos_filtrados, 3)

            with subcol1:
                for vehiculo, valor in divisiones[0]:
                    st.markdown(
                        fr"<div style='text-align: left; font-size:16px;'>{vehiculo} :&nbsp;&nbsp;&nbsp;{valor}</div>",
                        unsafe_allow_html=True
                    )

            with subcol2:
                for vehiculo, valor in divisiones[1]:
                    st.markdown(
                        fr"<div style='text-align: left; font-size:16px;'>{vehiculo} :&nbsp;&nbsp;&nbsp;{valor}</div>",
                        unsafe_allow_html=True
                    )

            with subcol3:
                for vehiculo, valor in divisiones[2]:
                    st.markdown(
                        fr"<div style='text-align: left; font-size:16px;'>{vehiculo} :&nbsp;&nbsp;&nbsp;{valor}</div>",
                        unsafe_allow_html=True
                    )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
            "<div style='text-align: left; font-size:20px; font-weight:600;'>D) Estructuración capas </div>", unsafe_allow_html=True)
            # 👇 Línea en blanco como separación
            
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>1. CBR subrasante % :&nbsp;&nbsp;&nbsp;{vrs3}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>2. CBR Sub-base % :&nbsp;&nbsp;&nbsp;{vrs2}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: left; font-size:18px;'>3. CBR Base %    :&nbsp;&nbsp;&nbsp;&nbsp;{vrs1}</div>",
            unsafe_allow_html=True
        )      
            

        with col2:
            st.markdown(
            "<div style='text-align: center; font-size:20px; font-weight:600;'>C) Transformar vehículos a ejes 1er año</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            df_ejes = transformar_vehiculos_a_ejes(tc_nombre, params, fvp, fvv)

            # Filtrar filas donde "Ejes 1er Año" sea mayor a 0
            df_ejes_filtrado = df_ejes[df_ejes["Ejes 1er Año"] > 0]

            st.dataframe(
                df_ejes_filtrado.style.format({
                    "Cargas (Ton)": "{:.2f}",
                    "Cargas (Kip)": "{:.2f}",
                    "Ejes 1er Año": "{:,.0f}"
                }).set_properties(**{
                    "text-align": "center",
                    "border": "1px solid #E2E8F0",
                    "padding": "8px"
                }).set_table_styles([
                    {"selector": "th", "props": [("background-color", "#3B82F6"), ("color", "white"), ("font-weight", "bold")]}
                ]),
                height=450
            )      
        
        with col3:
           
            st.markdown(
            "<div style='text-align: right; font-size:20px; font-weight:600;'>E) Cálculos y Resultados</div>", unsafe_allow_html=True)
            st.markdown(
            fr"<div style='text-align:right; font-size:18px;'>Abscisa nivel de confianza U :&nbsp;&nbsp;&nbsp;{U:.4f}</div>",
            unsafe_allow_html=True
        )      

            

            st.markdown(
            fr"<div style='text-align:right; font-size:18px;'>Cte exper. para subbase y Subras. VRS0:&nbsp;&nbsp;&nbsp;{VRS02:.3f}</div>",
            unsafe_allow_html=True    
        )

            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Z definida a daño prof. en cm:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{Prof3:.2f}</div>",
            unsafe_allow_html=True
        ) 
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Ejes equivalentes a resistir :&nbsp;&nbsp;&nbsp;{Esal3:,.0f}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Espesor en Grava Equiv. requerido en cm :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{Zg3:.2f}</div>",
            unsafe_allow_html=True
        )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
            fr"<div style='text-align:right; font-size:18px;'>Cte exper. para bases VRS0:&nbsp;&nbsp;{VRS01:.3f}</div>",
            unsafe_allow_html=True
        )      
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Z definida a daño superf. en cm:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{Prof1:.2f}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Ejes equivalentes a resistir :&nbsp;&nbsp;&nbsp;{Esal1:,.0f}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Espesor en Grava Equiv. requerido en cm :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{Zg1:.2f}</div>",
            unsafe_allow_html=True
        )                    
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
            "<div style='text-align: right; font-size:20px; font-weight:600;'>F) Estructura del pavimento en cm</div>", unsafe_allow_html=True)
            
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Carpeta asfáltica:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{D1}</div>",
            unsafe_allow_html=True
        ) 
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Base asfáltica:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{D2}</div>",
            unsafe_allow_html=True
        )
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Base hidráulica:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{D3}</div>",
            unsafe_allow_html=True
        )
            
            st.markdown(
            fr"<div style='text-align: right; font-size:18px;'>Sub-base hidráulica:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{D4}</div>",
            unsafe_allow_html=True
        )