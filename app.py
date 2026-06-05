import math
import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_js_eval import get_geolocation
except ImportError:
    get_geolocation = None


MATERIAL_SPEEDS = {
    "PVC": 900,
    "PEAD / HDPE": 700,
    "Acero": 1200,
    "Hierro fundido": 1000,
    "Concreto": 850,
    "Personalizado": 950,
}

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
ACTIVE_SENSOR_CSV_PATH = DATA_DIR / "datos_sensores_activos.csv"
LAST_UPLOADED_CSV_PATH = DATA_DIR / "ultimo_csv_cargado.csv"
LAST_CAPTURE_CSV_PATH = DATA_DIR / "ultima_captura_sensores.csv"

REQUIRED_COLUMNS = {
    "sensor_id",
    "latitud",
    "longitud",
    "amplitud_rms",
    "ruido_base",
    "frecuencia_dominante_hz",
    "tiempo_llegada_s",
}

SENSOR_DATA_COLUMNS = [
    "sensor_id",
    "latitud",
    "longitud",
    "amplitud_rms",
    "ruido_base",
    "frecuencia_dominante_hz",
    "tiempo_llegada_s",
]

CSV_SEARCH_DIRS = [
    DATA_DIR,
    APP_DIR,
]

COPYRIGHT_DATA = {
    "Entidad": "Consejo Profesional Nacional de Ingenieria - COPNIA",
    "Pais": "Republica de Colombia",
    "Matricula profesional": "25269-255714 CND",
    "Profesion": "Ingeniero de Sistemas con enfasis en Software",
    "Nombre": "Adolfo Barrera Vargas",
}


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def calculate_leak_position(distance_m, acoustic_speed_m_s, time_delta_s):
    raw_position = (distance_m - acoustic_speed_m_s * time_delta_s) / 2
    return clamp(raw_position, 0, distance_m), raw_position


def calculate_segment_length(horizontal_m, elevation_change_m):
    return math.sqrt(horizontal_m**2 + elevation_change_m**2)


def calculate_total_route_length(route_data):
    route_data = route_data.copy()
    required_route_columns = ["tramo", "longitud_horizontal_m", "desnivel_m", "direccion", "tipo"]
    for column in required_route_columns:
        if column not in route_data.columns:
            route_data[column] = None
    for column in ["latitud_fin", "longitud_fin"]:
        if column not in route_data.columns:
            route_data[column] = None

    route_data["longitud_horizontal_m"] = pd.to_numeric(
        route_data["longitud_horizontal_m"], errors="coerce"
    )
    route_data["desnivel_m"] = pd.to_numeric(route_data["desnivel_m"], errors="coerce")
    route_data["latitud_fin"] = pd.to_numeric(route_data["latitud_fin"], errors="coerce")
    route_data["longitud_fin"] = pd.to_numeric(route_data["longitud_fin"], errors="coerce")
    route_data = route_data.dropna(subset=["longitud_horizontal_m", "desnivel_m"])
    route_data = route_data[route_data["longitud_horizontal_m"] >= 0].reset_index(drop=True)
    route_data["tramo"] = route_data["tramo"].fillna("").replace("", "Tramo")
    route_data["direccion"] = route_data["direccion"].fillna("Derecha")
    route_data["tipo"] = route_data["tipo"].fillna("Recto")

    route_data["longitud_real_m"] = route_data.apply(
        lambda row: calculate_segment_length(
            float(row["longitud_horizontal_m"]),
            float(row["desnivel_m"]),
        ),
        axis=1,
    )
    return route_data, float(route_data["longitud_real_m"].sum())


def calculate_intensity(amplitude_a, amplitude_b, noise_a, noise_b):
    signal_a = max(amplitude_a - noise_a, 0)
    signal_b = max(amplitude_b - noise_b, 0)
    normalized = (signal_a + signal_b) / 2
    return clamp(normalized, 0, 1)


def classify_intensity(intensity):
    if intensity <= 0.25:
        return "Baja", "#2f9e44"
    if intensity <= 0.50:
        return "Media", "#f59f00"
    if intensity <= 0.75:
        return "Alta", "#f76707"
    return "Critica", "#d6336c"


def load_local_sensor_csv(path):
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return None, f"No se pudo leer el archivo local: {exc}"
    return validate_sensor_data(data)


def find_repository_csv_files():
    csv_files = []
    seen_paths = set()
    for directory in CSV_SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            resolved_path = path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            csv_files.append(path)
    return csv_files


def format_repository_csv_path(path):
    try:
        return str(path.relative_to(APP_DIR))
    except ValueError:
        return str(path)


def save_sensor_data_copy(sensor_data, path):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        sensor_data[SENSOR_DATA_COLUMNS].to_csv(path, index=False)
    except Exception as exc:
        st.warning(f"No se pudo guardar copia local en {path}: {exc}")


def validate_sensor_data(data):
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return None, f"Faltan columnas obligatorias: {missing}"

    if len(data) < 2:
        return None, "Ingrese al menos dos registros de sensores."

    numeric_columns = [
        "latitud",
        "longitud",
        "amplitud_rms",
        "ruido_base",
        "frecuencia_dominante_hz",
        "tiempo_llegada_s",
    ]
    data = data.copy()
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[numeric_columns].isna().any().any():
        return None, "Hay valores vacios o no numericos en los datos de sensores."

    invalid_gps = (
        ~data["latitud"].between(-90, 90)
        | ~data["longitud"].between(-180, 180)
    )
    if invalid_gps.any():
        return None, "Hay coordenadas GPS fuera de rango."

    data = data.reset_index(drop=True)
    if len(data) >= 2:
        sensor_a = data.iloc[0]
        sensor_b = data.iloc[1]
        same_lat = math.isclose(float(sensor_a["latitud"]), float(sensor_b["latitud"]), abs_tol=1e-7)
        same_lon = math.isclose(float(sensor_a["longitud"]), float(sensor_b["longitud"]), abs_tol=1e-7)
        if same_lat and same_lon:
            return None, (
                "Sensor A y Sensor B tienen las mismas coordenadas GPS. "
                "Capture o ingrese la ubicacion fisica de cada sensor por separado."
            )

    return data, None


def build_default_sensor_capture_data():
    return pd.DataFrame(
        [
            {
                "sensor_id": "Sensor A",
                "latitud": 4.711234,
                "longitud": -74.072345,
                "amplitud_rms": 0.82,
                "ruido_base": 0.18,
                "frecuencia_dominante_hz": 780,
                "tiempo_llegada_s": 0.000,
            },
            {
                "sensor_id": "Sensor B",
                "latitud": 4.711234,
                "longitud": -74.070542,
                "amplitud_rms": 0.55,
                "ruido_base": 0.20,
                "frecuencia_dominante_hz": 780,
                "tiempo_llegada_s": 0.050,
            },
        ]
    )


def load_initial_sensor_capture_data():
    for path in [LAST_CAPTURE_CSV_PATH, ACTIVE_SENSOR_CSV_PATH]:
        if path.exists():
            sensor_data, error = load_local_sensor_csv(path)
            if error is None:
                return sensor_data
    return build_default_sensor_capture_data()


def get_sensor_locations(sensor_data):
    sensor_a = sensor_data.iloc[0]
    sensor_b = sensor_data.iloc[1]
    return {
        "sensor_a_id": str(sensor_a["sensor_id"]),
        "sensor_b_id": str(sensor_b["sensor_id"]),
        "sensor_a_lat": float(sensor_a["latitud"]),
        "sensor_a_lon": float(sensor_a["longitud"]),
        "sensor_b_lat": float(sensor_b["latitud"]),
        "sensor_b_lon": float(sensor_b["longitud"]),
    }


def estimate_leak_gps(sensor_locations, leak_from_a_m, distance_m):
    if distance_m <= 0:
        ratio = 0.0
    else:
        ratio = clamp(leak_from_a_m / distance_m, 0, 1)

    leak_lat = sensor_locations["sensor_a_lat"] + ratio * (
        sensor_locations["sensor_b_lat"] - sensor_locations["sensor_a_lat"]
    )
    leak_lon = sensor_locations["sensor_a_lon"] + ratio * (
        sensor_locations["sensor_b_lon"] - sensor_locations["sensor_a_lon"]
    )
    return leak_lat, leak_lon


def estimate_leak_gps_from_route(sensor_locations, route_data, leak_from_a_m, distance_m, route_mode):
    route_points = [
        (sensor_locations["sensor_a_lat"], sensor_locations["sensor_a_lon"]),
    ]

    if route_data[["latitud_fin", "longitud_fin"]].isna().any().any():
        leak_lat, leak_lon = estimate_leak_gps(sensor_locations, leak_from_a_m, distance_m)
        return leak_lat, leak_lon, "Linea recta entre sensores"

    gps_method = (
        "Linea recta entre sensores"
        if route_mode == "Tramo linea recta"
        else "Trazado GPS con codos"
    )

    for _, segment in route_data.iterrows():
        route_points.append((float(segment["latitud_fin"]), float(segment["longitud_fin"])))

    accumulated_m = 0.0
    for index, segment in route_data.iterrows():
        segment_length_m = float(segment["longitud_real_m"])
        next_accumulated_m = accumulated_m + segment_length_m
        if accumulated_m <= leak_from_a_m <= next_accumulated_m:
            ratio = 0 if segment_length_m == 0 else (leak_from_a_m - accumulated_m) / segment_length_m
            start_lat, start_lon = route_points[index]
            end_lat, end_lon = route_points[index + 1]
            leak_lat = start_lat + ratio * (end_lat - start_lat)
            leak_lon = start_lon + ratio * (end_lon - start_lon)
            return leak_lat, leak_lon, gps_method
        accumulated_m = next_accumulated_m

    return route_points[-1][0], route_points[-1][1], gps_method


def build_gps_report_points(sensor_locations, route_data, leak_lat, leak_lon):
    points = [
        {
            "Punto": sensor_locations["sensor_a_id"],
            "Latitud": sensor_locations["sensor_a_lat"],
            "Longitud": sensor_locations["sensor_a_lon"],
        }
    ]

    has_sensor_b = False
    for index, segment in route_data.iterrows():
        if pd.isna(segment["latitud_fin"]) or pd.isna(segment["longitud_fin"]):
            continue

        is_last_segment = index == len(route_data) - 1
        point_name = sensor_locations["sensor_b_id"] if is_last_segment else f"Fin {segment['tramo']}"
        if is_last_segment:
            has_sensor_b = True

        points.append(
            {
                "Punto": point_name,
                "Latitud": float(segment["latitud_fin"]),
                "Longitud": float(segment["longitud_fin"]),
            }
        )

    if not has_sensor_b:
        points.append(
            {
                "Punto": sensor_locations["sensor_b_id"],
                "Latitud": sensor_locations["sensor_b_lat"],
                "Longitud": sensor_locations["sensor_b_lon"],
            }
        )

    points.append(
        {
            "Punto": "Fuga estimada",
            "Latitud": leak_lat,
            "Longitud": leak_lon,
        }
    )
    return pd.DataFrame(points)


def build_google_maps_urls(leak_lat, leak_lon):
    coordinate_query = f"{leak_lat:.7f},{leak_lon:.7f}"
    encoded_query = urllib.parse.quote_plus(coordinate_query)
    return {
        "embed": f"https://www.google.com/maps?q={encoded_query}&z=17&output=embed",
        "link": f"https://www.google.com/maps/search/?api=1&query={encoded_query}",
    }


@st.cache_data(ttl=86400)
def locate_municipality_from_gps(lat, lon):
    query = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "zoom": "10",
            "addressdetails": "1",
        }
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "IANC_FUGAS/1.0 reverse-geocoding",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return "No identificado"

    address = payload.get("address", {})
    for key in ["municipality", "city", "town", "village", "county", "state_district"]:
        value = address.get(key)
        if value:
            return value
    return "No identificado"


def get_sensor_defaults(sensor_data):
    if sensor_data is None:
        return {
            "frequency_hz": 780,
            "amplitude_a": 0.82,
            "amplitude_b": 0.55,
            "noise_a": 0.18,
            "noise_b": 0.20,
            "time_delta_s": 0.050,
        }

    sensor_a = sensor_data.iloc[0]
    sensor_b = sensor_data.iloc[1]
    time_delta_s = float(sensor_b["tiempo_llegada_s"]) - float(sensor_a["tiempo_llegada_s"])

    return {
        "frequency_hz": int(clamp(round(float(sensor_a["frecuencia_dominante_hz"])), 1, 2000)),
        "amplitude_a": clamp(float(sensor_a["amplitud_rms"]), 0, 1),
        "amplitude_b": clamp(float(sensor_b["amplitud_rms"]), 0, 1),
        "noise_a": clamp(float(sensor_a["ruido_base"]), 0, 1),
        "noise_b": clamp(float(sensor_b["ruido_base"]), 0, 1),
        "time_delta_s": time_delta_s,
    }


def plot_pipe(route_data, distance_m, leak_position_m, severity_color, intensity):
    fig, ax = plt.subplots(figsize=(10, 3.4))

    x_points = [0.0]
    y_points = [0.0]
    accumulated_x = 0.0
    accumulated_y = 0.0
    leak_x = 0.0
    leak_y = 0.0
    accumulated_length = 0.0

    for _, segment in route_data.iterrows():
        horizontal_m = float(segment["longitud_horizontal_m"])
        elevation_change_m = float(segment["desnivel_m"])
        segment_length = float(segment["longitud_real_m"])
        direction = segment.get("direccion", "Derecha")
        if direction == "Izquierda":
            delta_x, delta_y = -horizontal_m, 0.0
        elif direction == "Arriba":
            delta_x, delta_y = 0.0, horizontal_m
        elif direction == "Abajo":
            delta_x, delta_y = 0.0, -horizontal_m
        else:
            delta_x, delta_y = horizontal_m, 0.0

        next_x = accumulated_x + delta_x
        next_y = accumulated_y + delta_y

        if accumulated_length <= leak_position_m <= accumulated_length + segment_length:
            ratio = 0 if segment_length == 0 else (leak_position_m - accumulated_length) / segment_length
            leak_x = accumulated_x + delta_x * ratio
            leak_y = accumulated_y + delta_y * ratio

        x_points.append(next_x)
        y_points.append(next_y)
        accumulated_x = next_x
        accumulated_y = next_y
        accumulated_length += segment_length

    ax.plot(x_points, y_points, color="#495057", linewidth=10, solid_capstyle="round")
    ax.scatter([x_points[0], x_points[-1]], [y_points[0], y_points[-1]], s=240, color="#1c7ed6", zorder=3)
    ax.scatter([leak_x], [leak_y], s=420, color=severity_color, zorder=4)

    pulse_radius = 0.10 + intensity * 0.25
    theta = np.linspace(0, 2 * math.pi, 200)
    ax.plot(
        leak_x + np.cos(theta) * max(distance_m, 1) * 0.015,
        leak_y + np.sin(theta) * pulse_radius,
        color=severity_color,
        linewidth=2,
        alpha=0.45,
    )

    ax.text(x_points[0], y_points[0] + 0.35, "Sensor A", ha="center", va="bottom", fontsize=10, color="#1c7ed6")
    ax.text(x_points[-1], y_points[-1] + 0.35, "Sensor B", ha="center", va="bottom", fontsize=10, color="#1c7ed6")
    ax.text(
        leak_x,
        leak_y - 0.42,
        f"Fuga probable\n{leak_position_m:.2f} m",
        ha="center",
        va="top",
        fontsize=10,
        color=severity_color,
        fontweight="bold",
    )

    x_padding = max(max(x_points) - min(x_points), 1) * 0.08
    y_padding = max(max(y_points) - min(y_points), 1) * 0.35
    ax.set_xlim(min(x_points) - x_padding, max(x_points) + x_padding)
    ax.set_ylim(min(y_points) - y_padding - 0.7, max(y_points) + y_padding + 0.7)
    ax.set_xlabel("Trazado en planta X (m)")
    ax.set_ylabel("Trazado en planta Y (m)")
    ax.grid(linestyle="--", alpha=0.25)
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_vertical_profile(route_data, leak_position_m, severity_color):
    fig, ax = plt.subplots(figsize=(10, 2.8))

    distance_points = [0.0]
    elevation_points = [0.0]
    accumulated_distance = 0.0
    accumulated_elevation = 0.0
    leak_elevation = 0.0

    for _, segment in route_data.iterrows():
        segment_length = float(segment["longitud_real_m"])
        elevation_change_m = float(segment["desnivel_m"])
        next_distance = accumulated_distance + segment_length
        next_elevation = accumulated_elevation + elevation_change_m

        if accumulated_distance <= leak_position_m <= next_distance:
            ratio = 0 if segment_length == 0 else (leak_position_m - accumulated_distance) / segment_length
            leak_elevation = accumulated_elevation + elevation_change_m * ratio

        distance_points.append(next_distance)
        elevation_points.append(next_elevation)
        accumulated_distance = next_distance
        accumulated_elevation = next_elevation

    ax.plot(distance_points, elevation_points, color="#495057", linewidth=4, marker="o")
    ax.scatter([leak_position_m], [leak_elevation], s=260, color=severity_color, zorder=4)
    ax.text(
        leak_position_m,
        leak_elevation,
        f"  Fuga probable\n  {leak_position_m:.2f} m",
        ha="left",
        va="center",
        fontsize=10,
        color=severity_color,
        fontweight="bold",
    )

    ax.set_xlabel("Longitud real acumulada (m)")
    ax.set_ylabel("Desnivel acumulado (m)")
    ax.grid(linestyle="--", alpha=0.25)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_acoustic_signals(
    frequency_hz,
    amplitude_a,
    amplitude_b,
    noise_a,
    noise_b,
    time_delta_s,
):
    fig, ax = plt.subplots(figsize=(10, 3.0))

    window_s = min(0.20, max(0.02, 6 / max(frequency_hz, 1), abs(time_delta_s) * 1.2))
    time_s = np.linspace(0, window_s, 900)
    signal_a = amplitude_a * np.sin(2 * math.pi * frequency_hz * time_s)
    signal_b = amplitude_b * np.sin(2 * math.pi * frequency_hz * (time_s - time_delta_s))

    noise_wave_a = noise_a * 0.25 * np.sin(2 * math.pi * frequency_hz * 2.7 * time_s)
    noise_wave_b = noise_b * 0.25 * np.sin(2 * math.pi * frequency_hz * 2.3 * time_s)

    ax.plot(time_s * 1000, signal_a + noise_wave_a, color="#1c7ed6", linewidth=1.8, label="Sensor A")
    ax.plot(time_s * 1000, signal_b + noise_wave_b, color="#d6336c", linewidth=1.8, label="Sensor B")
    ax.axvline(abs(time_delta_s) * 1000, color="#495057", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        abs(time_delta_s) * 1000,
        ax.get_ylim()[1] * 0.85,
        f" dt = {time_delta_s:.6f} s",
        ha="left",
        va="center",
        fontsize=9,
        color="#495057",
    )

    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Amplitud relativa")
    ax.grid(linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="IANC FUGAS",
    page_icon="",
    layout="wide",
)

st.title("IANC FUGAS - Localizacion acustica de fugas")
st.caption("Analisis basado en correlacion entre dos senales acusticas.")
st.caption(
    "Los sensores aportan las mediciones acusticas. Las coordenadas GPS corresponden "
    "al punto fisico donde el operario instala cada sensor."
)
st.caption(
    "Derechos de autor: "
    f"{COPYRIGHT_DATA['Nombre']} - {COPYRIGHT_DATA['Profesion']} - "
    f"Matricula profesional {COPYRIGHT_DATA['Matricula profesional']}."
)

input_mode = st.radio(
    "Fuente de datos",
    ["CSV del repositorio", "Captura de datos en tiempo real"],
    horizontal=True,
)

loaded_sensor_data = None
csv_error = None

if input_mode == "CSV del repositorio":
    repository_csv_files = find_repository_csv_files()
    if not repository_csv_files:
        st.info("No se encontraron archivos CSV en el repositorio. Agregue los CSV en la carpeta data/.")
    else:
        selected_csv_path = st.selectbox(
            "Seleccione un CSV incluido en el repositorio",
            repository_csv_files,
            format_func=format_repository_csv_path,
        )
        loaded_sensor_data, csv_error = load_local_sensor_csv(selected_csv_path)
        if csv_error:
            st.info(csv_error)
        else:
            st.success("CSV del repositorio cargado correctamente.")
            save_sensor_data_copy(loaded_sensor_data, ACTIVE_SENSOR_CSV_PATH)
            save_sensor_data_copy(loaded_sensor_data, LAST_UPLOADED_CSV_PATH)
            st.caption(f"Archivo seleccionado: {format_repository_csv_path(selected_csv_path)}")
            st.caption(f"Datos activos guardados en {format_repository_csv_path(ACTIVE_SENSOR_CSV_PATH)}")
            st.caption(f"Copia local guardada en {format_repository_csv_path(LAST_UPLOADED_CSV_PATH)}")
            st.dataframe(loaded_sensor_data, width="stretch", hide_index=True)
else:
    st.subheader("Captura de datos en tiempo real")
    st.caption(
        "Use esta tabla cuando el operario registre los datos desde celular, tablet o PC. "
        "El GPS debe corresponder al punto exacto donde se instala cada sensor."
    )

    if "sensor_capture_data" not in st.session_state:
        st.session_state.sensor_capture_data = load_initial_sensor_capture_data()
    if "gps_capture_nonce" not in st.session_state:
        st.session_state.gps_capture_nonce = 0
    if "gps_capture_target" not in st.session_state:
        st.session_state.gps_capture_target = None
    if "gps_applied_nonce" not in st.session_state:
        st.session_state.gps_applied_nonce = 0
    if "sensor_capture_editor_version" not in st.session_state:
        st.session_state.sensor_capture_editor_version = 0

    if get_geolocation is None:
        st.info(
            "GPS automatico no disponible. Instale streamlit-js-eval o ingrese "
            "latitud y longitud manualmente desde el celular/GPS."
        )
    else:
        gps_cols = st.columns(2)
        if gps_cols[0].button("Capturar GPS Sensor A"):
            st.session_state.gps_capture_target = "Sensor A"
            st.session_state.gps_capture_nonce += 1
        if gps_cols[1].button("Capturar GPS Sensor B"):
            st.session_state.gps_capture_target = "Sensor B"
            st.session_state.gps_capture_nonce += 1

        gps_target = st.session_state.gps_capture_target
        current_location = None
        if gps_target:
            component_key = f"gps_{gps_target}_{st.session_state.gps_capture_nonce}"
            current_location = get_geolocation(component_key=component_key)

        if current_location and current_location.get("coords"):
            coords = current_location["coords"]
            st.write(
                f"GPS {gps_target}: {coords['latitude']:.7f}, "
                f"{coords['longitude']:.7f}"
            )
            if st.session_state.gps_applied_nonce != st.session_state.gps_capture_nonce:
                sensor_index = 0 if gps_target == "Sensor A" else 1
                st.session_state.sensor_capture_data.loc[sensor_index, "latitud"] = coords["latitude"]
                st.session_state.sensor_capture_data.loc[sensor_index, "longitud"] = coords["longitude"]
                st.session_state.gps_applied_nonce = st.session_state.gps_capture_nonce
                st.session_state.sensor_capture_editor_version += 1
                st.success(f"GPS aplicado a {gps_target}.")
                st.rerun()
        elif gps_target:
            st.info(f"Esperando lectura GPS para {gps_target}. Autorice el GPS del navegador si lo solicita.")
        else:
            st.info("Autorice el GPS del navegador para capturar la ubicacion del dispositivo.")

    captured_sensor_data = st.data_editor(
        st.session_state.sensor_capture_data,
        key=f"sensor_capture_editor_{st.session_state.sensor_capture_editor_version}",
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "sensor_id": st.column_config.TextColumn("Sensor"),
            "latitud": st.column_config.NumberColumn("Latitud", min_value=-90.0, max_value=90.0, format="%.7f"),
            "longitud": st.column_config.NumberColumn("Longitud", min_value=-180.0, max_value=180.0, format="%.7f"),
            "amplitud_rms": st.column_config.NumberColumn("Amplitud RMS", min_value=0.0, max_value=1.0, format="%.2f"),
            "ruido_base": st.column_config.NumberColumn("Ruido base", min_value=0.0, max_value=1.0, format="%.2f"),
            "frecuencia_dominante_hz": st.column_config.NumberColumn("Frecuencia dominante (Hz)", min_value=1, max_value=2000),
            "tiempo_llegada_s": st.column_config.NumberColumn("Tiempo llegada (s)", format="%.6f"),
        },
    )
    st.session_state.sensor_capture_data = captured_sensor_data

    sensor_summary_cols = st.columns(2)
    for sensor_index, sensor_column in enumerate(sensor_summary_cols):
        if sensor_index < len(captured_sensor_data):
            sensor = captured_sensor_data.iloc[sensor_index]
            lat = pd.to_numeric(sensor["latitud"], errors="coerce")
            lon = pd.to_numeric(sensor["longitud"], errors="coerce")
            amplitude = pd.to_numeric(sensor["amplitud_rms"], errors="coerce")
            arrival_time = pd.to_numeric(sensor["tiempo_llegada_s"], errors="coerce")
            gps_text = "Pendiente" if pd.isna(lat) or pd.isna(lon) else f"{lat:.7f}, {lon:.7f}"
            amplitude_text = "Pendiente" if pd.isna(amplitude) else f"{amplitude:.2f}"
            arrival_time_text = "Pendiente" if pd.isna(arrival_time) else f"{arrival_time:.6f} s"
            sensor_column.subheader(str(sensor["sensor_id"]))
            sensor_column.metric("GPS", gps_text)
            sensor_column.metric("Amplitud RMS", amplitude_text)
            sensor_column.metric("Tiempo llegada", arrival_time_text)

    loaded_sensor_data, csv_error = validate_sensor_data(captured_sensor_data)
    if csv_error:
        st.error(csv_error)
    else:
        save_sensor_data_copy(loaded_sensor_data, ACTIVE_SENSOR_CSV_PATH)
        save_sensor_data_copy(loaded_sensor_data, LAST_CAPTURE_CSV_PATH)
        st.caption(f"Datos activos guardados en {ACTIVE_SENSOR_CSV_PATH}")
        st.caption(f"Captura guardada en {LAST_CAPTURE_CSV_PATH}")

if loaded_sensor_data is None:
    st.header("Formato CSV requerido")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Columna": "sensor_id",
                    "Descripcion": "Identificador del sensor.",
                },
                {
                    "Columna": "latitud",
                    "Descripcion": "Latitud GPS del punto donde se instalo el sensor.",
                },
                {
                    "Columna": "longitud",
                    "Descripcion": "Longitud GPS del punto donde se instalo el sensor.",
                },
                {
                    "Columna": "amplitud_rms",
                    "Descripcion": "Amplitud RMS de la senal acustica capturada.",
                },
                {
                    "Columna": "ruido_base",
                    "Descripcion": "Nivel de ruido base registrado por el sensor.",
                },
                {
                    "Columna": "frecuencia_dominante_hz",
                    "Descripcion": "Frecuencia dominante de la senal en Hz.",
                },
                {
                    "Columna": "tiempo_llegada_s",
                    "Descripcion": "Tiempo de llegada de la senal en segundos.",
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.stop()

defaults = get_sensor_defaults(loaded_sensor_data)
sensor_locations = get_sensor_locations(loaded_sensor_data)

with st.sidebar:
    st.header("Derechos de autor")
    st.write(COPYRIGHT_DATA["Nombre"])
    st.caption(COPYRIGHT_DATA["Profesion"])
    st.caption(f"Matricula profesional: {COPYRIGHT_DATA['Matricula profesional']}")
    st.caption(COPYRIGHT_DATA["Entidad"])
    st.caption(COPYRIGHT_DATA["Pais"])

    st.header("Informe")
    network_type = st.selectbox(
        "Tipo de red",
        ["Linea matriz", "Red secundaria", "Otro"],
    )
    route_reference = st.text_input(
        "Referencia del trazado",
        value="Calzada / via publica",
    )

    st.header("Trazado de tuberia")
    route_mode = st.radio(
        "Metodo de distancia",
        ["Tramo linea recta", "Tramos con codos o pendiente"],
        horizontal=False,
    )

    if route_mode == "Tramo linea recta":
        simple_distance_m = st.number_input(
            "Longitud real entre sensores (m)",
            min_value=1.0,
            max_value=10000.0,
            value=200.0,
            step=10.0,
        )
        route_input = pd.DataFrame(
            [
                {
                    "tramo": "Tramo 1",
                    "longitud_horizontal_m": simple_distance_m,
                    "desnivel_m": 0.0,
                    "direccion": "Derecha",
                    "tipo": "Recto",
                    "latitud_fin": sensor_locations["sensor_b_lat"],
                    "longitud_fin": sensor_locations["sensor_b_lon"],
                }
            ]
        )
    else:
        st.caption(
            "Use un tramo por cada recta entre codos o cambios de pendiente. "
            "La direccion permite dibujar el recorrido real en planta."
        )
        st.caption(
            "El desnivel corresponde a topografia, planos de obra, levantamiento GPS, "
            "estacion total, nivelacion o medicion de campo; no viene del sensor."
        )
        st.caption(
            "Para geolocalizar la fuga con precision cuando hay codos, registre la "
            "latitud y longitud del final de cada tramo. El ultimo tramo debe terminar "
            "en el Sensor B."
        )
        route_input = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "tramo": "Tramo 1",
                        "longitud_horizontal_m": 100.0,
                        "desnivel_m": 0.0,
                        "direccion": "Derecha",
                        "tipo": "Recto",
                        "latitud_fin": None,
                        "longitud_fin": None,
                    },
                    {
                        "tramo": "Tramo 2",
                        "longitud_horizontal_m": 100.0,
                        "desnivel_m": 0.0,
                        "direccion": "Arriba",
                        "tipo": "Codo / cambio de direccion",
                        "latitud_fin": sensor_locations["sensor_b_lat"],
                        "longitud_fin": sensor_locations["sensor_b_lon"],
                    },
                ]
            ),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "tramo": st.column_config.TextColumn("Tramo"),
                "longitud_horizontal_m": st.column_config.NumberColumn(
                    "Longitud horizontal (m)",
                    min_value=0.0,
                    step=1.0,
                ),
                "desnivel_m": st.column_config.NumberColumn(
                    "Desnivel (m)",
                    step=1.0,
                ),
                "direccion": st.column_config.SelectboxColumn(
                    "Direccion en plano",
                    options=["Derecha", "Arriba", "Izquierda", "Abajo"],
                ),
                "tipo": st.column_config.SelectboxColumn(
                    "Tipo",
                    options=["Recto", "Codo / cambio de direccion", "Pendiente"],
                ),
                "latitud_fin": st.column_config.NumberColumn(
                    "Latitud fin tramo",
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.7f",
                ),
                "longitud_fin": st.column_config.NumberColumn(
                    "Longitud fin tramo",
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.7f",
                ),
            },
        )

    route_data, distance_m = calculate_total_route_length(route_input)
    st.metric("Longitud real calculada", f"{distance_m:.2f} m")
    if route_data.empty or distance_m <= 0:
        st.error("Ingrese al menos un tramo con longitud mayor a cero.")
        st.stop()

    st.header("Material y velocidad")
    material = st.selectbox("Material de tuberia", list(MATERIAL_SPEEDS.keys()))
    default_speed = MATERIAL_SPEEDS[material]
    acoustic_speed_m_s = st.number_input(
        "Velocidad acustica estimada (m/s)",
        min_value=1.0,
        max_value=5000.0,
        value=float(default_speed),
        step=10.0,
    )

    st.header("Datos de sensores")
    frequency_hz = st.slider("Frecuencia dominante (Hz)", 1, 2000, defaults["frequency_hz"])
    amplitude_a = st.slider("Amplitud RMS Sensor A", 0.0, 1.0, defaults["amplitude_a"], 0.01)
    amplitude_b = st.slider("Amplitud RMS Sensor B", 0.0, 1.0, defaults["amplitude_b"], 0.01)
    noise_a = st.slider("Ruido base Sensor A", 0.0, 1.0, defaults["noise_a"], 0.01)
    noise_b = st.slider("Ruido base Sensor B", 0.0, 1.0, defaults["noise_b"], 0.01)
    time_delta_s = st.number_input(
        "Desfase temporal entre senales dt (s)",
        min_value=-10.0,
        max_value=10.0,
        value=float(defaults["time_delta_s"]),
        step=0.001,
        format="%.6f",
    )

leak_from_a_m, raw_position_m = calculate_leak_position(
    distance_m, acoustic_speed_m_s, time_delta_s
)
leak_from_b_m = distance_m - leak_from_a_m
leak_lat, leak_lon, leak_gps_method = estimate_leak_gps_from_route(
    sensor_locations, route_data, leak_from_a_m, distance_m, route_mode
)
gps_report_points = build_gps_report_points(sensor_locations, route_data, leak_lat, leak_lon)
google_maps_urls = build_google_maps_urls(leak_lat, leak_lon)
municipality = locate_municipality_from_gps(leak_lat, leak_lon)
intensity = calculate_intensity(amplitude_a, amplitude_b, noise_a, noise_b)
severity, severity_color = classify_intensity(intensity)

metric_cols = st.columns(4)
metric_cols[0].metric("Distancia desde Sensor A", f"{leak_from_a_m:.2f} m")
metric_cols[1].metric("Distancia desde Sensor B", f"{leak_from_b_m:.2f} m")
metric_cols[2].metric("Intensidad relativa", f"{intensity:.2f}")
metric_cols[3].metric("Magnitud relativa", severity)

st.markdown(
    f"""
    <div style="border-left: 8px solid {severity_color}; padding: 12px 16px; background: #f8f9fa; margin: 8px 0 18px 0;">
        <div style="font-size: 14px; color: #495057;">Magnitud relativa de fuga</div>
        <div style="font-size: 28px; font-weight: 700; color: {severity_color};">{severity}</div>
        <div style="font-size: 13px; color: #495057;">Indice de intensidad: {intensity:.2f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_position_m != leak_from_a_m:
    st.warning(
        "El calculo produjo una posicion fuera del tramo entre sensores. "
        "Se muestra ajustada al limite fisico de la tuberia."
    )

travel_delta_m = acoustic_speed_m_s * time_delta_s
max_physical_delta_s = distance_m / acoustic_speed_m_s
if abs(time_delta_s) >= max_physical_delta_s:
    st.warning(
        "El desfase temporal cargado equivale al tiempo de recorrido acustico entre "
        "los dos sensores o lo supera. Con esos datos, la localizacion cae en un "
        "extremo de la tuberia. Revise el desfase temporal del CSV o la velocidad "
        "acustica estimada."
    )

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Visualizacion de la tuberia")
    st.pyplot(plot_pipe(route_data, distance_m, leak_from_a_m, severity_color, intensity))
    st.subheader("Perfil vertical")
    st.pyplot(plot_vertical_profile(route_data, leak_from_a_m, severity_color))
    st.subheader("Tramos usados para el calculo")
    st.dataframe(
        route_data[
            [
                "tramo",
                "tipo",
                "direccion",
                "longitud_horizontal_m",
                "desnivel_m",
                "longitud_real_m",
                "latitud_fin",
                "longitud_fin",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

with right:
    st.subheader("Resultado de localizacion")
    st.write(
        pd.DataFrame(
            {
                "Variable": [
                    "Punto probable de fuga",
                    "Municipio",
                    "Tipo de red",
                    "Referencia del trazado",
                    "GPS fuga estimada",
                    "Metodo GPS fuga",
                    "GPS Sensor A",
                    "GPS Sensor B",
                    "Distancia desde Sensor A",
                    "Distancia desde Sensor B",
                    "Longitud real del trazado",
                    "Velocidad acustica",
                    "Desfase temporal",
                    "Distancia equivalente por desfase",
                    "Frecuencia dominante",
                    "Indice de intensidad",
                    "Magnitud relativa de fuga",
                    "Derechos de autor - nombre",
                    "Derechos de autor - profesion",
                    "Derechos de autor - matricula",
                    "Derechos de autor - entidad",
                    "Derechos de autor - pais",
                ],
                "Valor": [
                    f"{leak_from_a_m:.2f} m",
                    municipality,
                    network_type,
                    route_reference if route_reference else "No registrada",
                    f"{leak_lat:.7f}, {leak_lon:.7f}",
                    leak_gps_method,
                    f"{sensor_locations['sensor_a_lat']:.7f}, {sensor_locations['sensor_a_lon']:.7f}",
                    f"{sensor_locations['sensor_b_lat']:.7f}, {sensor_locations['sensor_b_lon']:.7f}",
                    f"{leak_from_a_m:.2f} m",
                    f"{leak_from_b_m:.2f} m",
                    f"{distance_m:.2f} m",
                    f"{acoustic_speed_m_s:.2f} m/s",
                    f"{time_delta_s:.6f} s",
                    f"{travel_delta_m:.2f} m",
                    f"{frequency_hz} Hz",
                    f"{intensity:.2f}",
                    severity,
                    COPYRIGHT_DATA["Nombre"],
                    COPYRIGHT_DATA["Profesion"],
                    COPYRIGHT_DATA["Matricula profesional"],
                    COPYRIGHT_DATA["Entidad"],
                    COPYRIGHT_DATA["Pais"],
                ],
            }
        )
    )
    st.subheader("Puntos GPS para informe")
    st.dataframe(
        gps_report_points,
        width="stretch",
        hide_index=True,
    )
    st.subheader("Google Maps")
    st.iframe(google_maps_urls["embed"], height=360)
    st.link_button("Abrir punto de fuga en Google Maps", google_maps_urls["link"])
    st.caption(
        "Google Maps solo muestra el mapa base; no muestra la tuberia enterrada. "
        "Si el marcador parece caer sobre una cubierta, lote o fachada, revise que las "
        "coordenadas de sensores y codos hayan sido tomadas sobre el trazado real de la "
        "red de acueducto en calzada, anden o servidumbre. "
        "La coordenada GPS de fuga se estima segun la distancia calculada desde Sensor A. "
        "Si todos los tramos tienen coordenada final, se interpola sobre el trazado GPS "
        "con codos; si faltan coordenadas, se usa linea recta entre sensores."
    )

st.subheader("Senales acusticas")
st.pyplot(
    plot_acoustic_signals(
        frequency_hz,
        amplitude_a,
        amplitude_b,
        noise_a,
        noise_b,
        time_delta_s,
    )
)

st.subheader("Rangos de magnitud relativa de fuga")
severity_table = pd.DataFrame(
    [
        {"Rango": "0.00 - 0.25", "Clasificacion": "Baja"},
        {"Rango": "0.26 - 0.50", "Clasificacion": "Media"},
        {"Rango": "0.51 - 0.75", "Clasificacion": "Alta"},
        {"Rango": "0.76 - 1.00", "Clasificacion": "Critica"},
    ]
)
st.dataframe(severity_table, width="stretch", hide_index=True)
