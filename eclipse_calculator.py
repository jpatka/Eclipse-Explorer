# -*- coding: utf-8 -*-
"""
Uniwersalny Kalkulator i Eksplorator Zaćmień (Eclipse Explorer)
Wersja skryptowa (do uruchamiania w czystym Pythonie: python eclipse_calculator.py)
"""

import os
import sys
import math
import json
import webbrowser
import numpy as np
import pandas as pd

# Ustawienie nieinteraktywnego backendu Matplotlib przed importem pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skyfield.api import Loader, wgs84, Star
from skyfield import almanac
from skyfield.data import hipparcos
from skyfield.framelib import ecliptic_frame

# ==============================================================================
# KONFIGURACJA OBSERWACJI
# ==============================================================================

# 1. LOKALIZACJA OBSERWATORA
# ------------------------------------------------------------------------------
# Szerokość geograficzna (Latitude):
#   - Wartości DODATNIE (+) dla półkuli PÓŁNOCNEJ (N)  [np. 52.2297 dla Warszawy]
#   - Wartości UJEMNE   (-) dla półkuli POŁUDNIOWEJ (S) [np. -33.8688 dla Sydney]
OBSERVER_LATITUDE = 51.6174

# Długość geograficzna (Longitude):
#   - Wartości DODATNIE (+) dla półkuli WSCHODNIEJ (E) [np. 15.3082 dla Żagania]
#   - Wartości UJEMNE   (-) dla półkuli ZACHODNIEJ (W) [np. -74.0060 dla Nowego Jorku]
OBSERVER_LONGITUDE = 15.3082

# Nazwa lokalizacji (używana w nagłówkach wykresów, pliku ICS, nazwach plików itp.):
LOCATION_NAME = "Żagań"


# 2. OKRES WYSZUKIWANIA ZAĆMIEŃ
# ------------------------------------------------------------------------------
# Data rozpoczęcia poszukiwań (Rok, Miesiąc, Dzień):
START_YEAR  = 2000
START_MONTH = 1
START_DAY   = 1

# Data zakończenia poszukiwań (Rok, Miesiąc, Dzień):
END_YEAR    = 2036
END_MONTH   = 12
END_DAY     = 31

# ==============================================================================
# KONIEC SEKCJI KONFIGURACJI
# ==============================================================================


# Konfiguracja czytelnego wyświetlania tabeli Pandas w konsoli
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Clean/safe name for folders/files
loc_slug = LOCATION_NAME.lower().replace(" ", "_")

# --- KONFIGURACJA ŚCIEŻEK ---
output_dir = f"zacmienia_{loc_slug}"
astro_dir = "dane_astro"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(astro_dir, exist_ok=True)

load_astro = Loader(astro_dir)

print("1/7. Ładowanie efemeryd NASA oraz katalogu gwiazd...")
eph = load_astro('de421.bsp')
sun, moon, earth = eph['sun'], eph['moon'], eph['earth']

# Słownik planet
planets = {
    'Merkury': eph['mercury'],
    'Wenus': eph['venus'],
    'Mars': eph['mars'],
    'Jowisz': eph['jupiter barycenter'],
    'Saturn': eph['saturn barycenter']
}
ts = load_astro.timescale()

# Ładowanie gwiazd Hipparcos (mag <= 3.5)
with load_astro.open(hipparcos.URL) as f:
    stars_df = hipparcos.load_dataframe(f)
bright_stars = stars_df[stars_df['magnitude'] <= 3.5]
star_objects = Star.from_dataframe(bright_stars)

# Definicja obserwatora na podstawie danych z konfiguracji
user_geo = wgs84.latlon(OBSERVER_LATITUDE, OBSERVER_LONGITUDE)
user_obs = earth + user_geo

# Zakres czasowy na podstawie konfiguracji
t0 = ts.utc(START_YEAR, START_MONTH, START_DAY)
t1 = ts.utc(END_YEAR, END_MONTH, END_DAY)

# Stałe promienie fizyczne (km)
R_SUN_KM = 696340.0
R_MOON_KM = 1737.4
R_EARTH_KM = 6371.0


# ==========================================================
# FUNKCJE GEOMETRYCZNE I POMOCNICZE
# ==========================================================
def compute_obscuration(r_sun: float, r_moon: float, sep: float) -> float:
    """Wyznacza ułamek zakrytej powierzchni tarczy Słońca (0.0 do 1.0)."""
    if sep >= r_sun + r_moon:
        return 0.0
    if sep <= r_moon - r_sun:
        return 1.0
    if sep <= r_sun - r_moon:
        return (r_moon / r_sun) ** 2

    d1 = (r_sun ** 2 - r_moon ** 2 + sep ** 2) / (2.0 * sep)
    d2 = (r_moon ** 2 - r_sun ** 2 + sep ** 2) / (2.0 * sep)

    area_sun_part = r_sun ** 2 * np.arccos(np.clip(d1 / r_sun, -1.0, 1.0)) - d1 * np.sqrt(
        np.maximum(0.0, r_sun ** 2 - d1 ** 2))
    area_moon_part = r_moon ** 2 * np.arccos(np.clip(d2 / r_moon, -1.0, 1.0)) - d2 * np.sqrt(
        np.maximum(0.0, r_moon ** 2 - d2 ** 2))

    intersection_area = area_sun_part + area_moon_part
    sun_total_area = np.pi * r_sun ** 2

    return float(intersection_area / sun_total_area)


def get_angular_radius_deg(distance_km, body_radius_km):
    return np.degrees(np.arcsin(body_radius_km / distance_km))


def get_earth_shadow_radii_deg(t):
    d_sun = earth.at(t).observe(sun).distance().km
    d_moon = earth.at(t).observe(moon).apparent().distance().km
    r_sun = get_angular_radius_deg(d_sun, R_SUN_KM)
    r_earth_at_moon = get_angular_radius_deg(d_moon, R_EARTH_KM)
    r_umbra = r_earth_at_moon - (r_sun - (R_EARTH_KM / d_sun))
    r_penumbra = r_earth_at_moon + (r_sun + (R_EARTH_KM / d_sun))
    return r_umbra, r_penumbra


def get_apparent_altaz_with_refraction(observer, target, t):
    """Oblicza alt, az z uwzględnieniem refrakcji atmosferycznej."""
    obs = observer.at(t).observe(target).apparent()
    alt, az, dist = obs.altaz(temperature_C=10.0, pressure_mbar=1010.0)
    return alt.degrees, az.degrees, dist.km


# --- OBLICZENIA ASTRONOMICZNE ---
print(f"2/7. Obliczanie zjawisk dla lokalizacji: {LOCATION_NAME} ({OBSERVER_LATITUDE}°, {OBSERVER_LONGITUDE}°) w latach {START_YEAR}–{END_YEAR}...")
t_phases, phases = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))
t_new_moons = t_phases[phases == 0]
t_full_moons = t_phases[phases == 2]

tabela_danych = []

# Zaćmienia Słońca
for t_nm in t_new_moons:
    t_samples = ts.tt_jd(np.linspace(t_nm.tt - 0.166, t_nm.tt + 0.166, 121))
    s_alt, s_az, s_dist = get_apparent_altaz_with_refraction(user_obs, sun, t_samples)
    m_alt, m_az, m_dist = get_apparent_altaz_with_refraction(user_obs, moon, t_samples)

    s_obs = user_obs.at(t_samples).observe(sun).apparent()
    m_obs = user_obs.at(t_samples).observe(moon).apparent()
    seps = s_obs.separation_from(m_obs).degrees

    r_suns = get_angular_radius_deg(s_dist, R_SUN_KM)
    r_moons = get_angular_radius_deg(m_dist, R_MOON_KM)
    limit_seps = r_suns + r_moons

    visible_eclipsing = (seps < limit_seps) & (s_alt > -0.566)

    if np.any(visible_eclipsing):
        valid_indices = np.where(visible_eclipsing)[0]
        min_idx = valid_indices[np.argmin(seps[valid_indices])]
        t_max = t_samples[min_idx]
        sep_max = seps[min_idx]
        r_s, r_m = r_suns[min_idx], r_moons[min_idx]

        magnituda = max(0.0, (r_s + r_m - sep_max) / (2 * r_s))

        obs_val = compute_obscuration(r_s, r_m, sep_max)
        obscuration_pct = max(0.0, min(100.0, obs_val * 100.0))

        t_pocz, t_kon = t_samples[valid_indices[0]], t_samples[valid_indices[-1]]

        tabela_danych.append({
            'Typ': 'SŁOŃCE',
            'Maksimum (UTC)': t_max.utc_strftime('%Y-%m-%d %H:%M:%S'),
            'Magnituda': round(magnituda, 3),
            'Obscuration (%)': round(obscuration_pct, 1),
            'Wysokość (°)': round(s_alt[min_idx], 1),
            'Azymut (°)': round(s_az[min_idx], 1),
            'Początek (UTC)': t_pocz.utc_strftime('%H:%M:%S'),
            'Koniec (UTC)': t_kon.utc_strftime('%H:%M:%S'),
            'raw_time': t_max, 'sep': sep_max, 'alt_raw': s_alt[min_idx], 'az_raw': s_az[min_idx],
            't_pocz_raw': t_pocz, 't_kon_raw': t_kon, 'r_sun': r_s, 'r_moon': r_m
        })

# Zaćmienia Księżyca
for t_fm in t_full_moons:
    t_samples = ts.tt_jd(np.linspace(t_fm.tt - 0.208, t_fm.tt + 0.208, 151))
    e_sun = earth.at(t_samples).observe(sun).apparent()
    sun_lat, sun_lon, _ = e_sun.frame_latlon(ecliptic_frame)
    sh_lon = (sun_lon.degrees + 180.0) % 360.0
    sh_lat = -sun_lat.degrees

    e_moon = earth.at(t_samples).observe(moon).apparent()
    moon_lat, moon_lon, _ = e_moon.frame_latlon(ecliptic_frame)
    m_dist_km = e_moon.distance().km

    d_lon = (moon_lon.degrees - sh_lon + 180) % 360 - 180
    d_lat = moon_lat.degrees - sh_lat
    seps = np.sqrt(d_lon ** 2 + d_lat ** 2)

    r_umbra, r_penumbra = get_earth_shadow_radii_deg(t_samples)
    r_moons = get_angular_radius_deg(m_dist_km, R_MOON_KM)

    m_alt, m_az, _ = get_apparent_altaz_with_refraction(user_obs, moon, t_samples)
    visible_eclipsing = (seps < (r_penumbra + r_moons)) & (m_alt > -0.566)

    if np.any(visible_eclipsing):
        valid_indices = np.where(visible_eclipsing)[0]
        min_idx = valid_indices[np.argmin(seps[valid_indices])]
        t_max = t_samples[min_idx]
        sep_max = seps[min_idx]
        r_u, r_m = r_umbra[min_idx], r_moons[min_idx]

        magnituda_lunar = (r_u + r_m - sep_max) / (2 * r_m)
        faza_procent = max(0.0, magnituda_lunar * 100)
        t_pocz, t_kon = t_samples[valid_indices[0]], t_samples[valid_indices[-1]]

        tabela_danych.append({
            'Typ': 'KSIĘŻYC',
            'Maksimum (UTC)': t_max.utc_strftime('%Y-%m-%d %H:%M:%S'),
            'Magnituda': round(max(0.0, magnituda_lunar), 3),
            'Obscuration (%)': round(faza_procent, 1),
            'Wysokość (°)': round(m_alt[min_idx], 1),
            'Azymut (°)': round(m_az[min_idx], 1),
            'Początek (UTC)': t_pocz.utc_strftime('%H:%M:%S'),
            'Koniec (UTC)': t_kon.utc_strftime('%H:%M:%S'),
            'raw_time': t_max, 'sep': sep_max, 'alt_raw': m_alt[min_idx], 'az_raw': m_az[min_idx],
            't_pocz_raw': t_pocz, 't_kon_raw': t_kon, 'r_moon': r_m, 'r_umbra': r_u, 'r_penumbra': r_penumbra[min_idx]
        })

tabela_danych.sort(key=lambda x: x['Maksimum (UTC)'], reverse=True)

# ==================== 1. INDYWIDUALNE WYKRESY GEOMETRII (300 DPI) ====================
print(f"3/7. Generowanie indywidualnych wykresów statycznych 300 DPI (Liczba zjawisk: {len(tabela_danych)})...")

for idx, zj in enumerate(tabela_danych, start=1):
    ti = zj['raw_time']
    date_str = ti.utc_strftime('%Y-%m-%d_%H-%M-%S')
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    if zj['Typ'] == 'SŁOŃCE':
        m_alt, m_az, _ = get_apparent_altaz_with_refraction(user_obs, moon, ti)

        dx = -(m_az - zj['az_raw']) * np.cos(np.radians(zj['alt_raw']))
        dx = (dx + 180) % 360 - 180
        dy = m_alt - zj['alt_raw']

        ax.add_patch(plt.Circle((0, 0), zj['r_sun'], color='gold', label='Słońce', alpha=0.9, zorder=1))
        ax.add_patch(plt.Circle((dx, dy), zj['r_moon'], color='black', label='Księżyc', alpha=0.95, zorder=2))
        lim_min, lim_max = -0.6, 0.6
        info = f"ZAĆMIENIE SŁOŃCA\nMagnituda: {zj['Magnituda']}\nObscuration: {zj['Obscuration (%)']}%"
    else:
        e_sun = earth.at(ti).observe(sun).apparent()
        sun_lat, sun_lon, _ = e_sun.frame_latlon(ecliptic_frame)
        sh_lon = (sun_lon.degrees + 180.0) % 360.0
        sh_lat = -sun_lat.degrees
        e_moon = earth.at(ti).observe(moon).apparent()
        moon_lat, moon_lon, _ = e_moon.frame_latlon(ecliptic_frame)
        dx = -(moon_lon.degrees - sh_lon + 180) % 360 - 180
        dy = moon_lat.degrees - sh_lat

        ax.add_patch(plt.Circle((0, 0), zj['r_penumbra'], color='gray', alpha=0.15, label='Półcień Ziemi', zorder=1))
        ax.add_patch(
            plt.Circle((0, 0), zj['r_umbra'], color='#441111', alpha=0.7, label='Cień Ziemi (Umbra)', zorder=2))
        m_color = '#cc5533' if zj['Obscuration (%)'] >= 100 else '#dddddd'
        ax.add_patch(
            plt.Circle((dx, dy), zj['r_moon'], facecolor=m_color, edgecolor='white', alpha=0.9, label='Księżyc',
                       zorder=3))
        lim_min, lim_max = -1.7, 1.7
        ax.set_ylim(-1.9, lim_max)
        info = f"ZAĆMIENIE KSIĘŻYCA\nMagnituda: {zj['Magnituda']}\nFaza: {zj['Obscuration (%)']}%"

    ax.set_xlim(lim_min, lim_max)
    if zj['Typ'] == 'SŁOŃCE': ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.tick_params(colors='white', labelsize=7)
    ax.set_xlabel("Różnica azymutu [°]", color='white', fontsize=8)
    ax.set_ylabel("Różnica wysokości [°]", color='white', fontsize=8)
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.15)

    ax.set_title(f"[{idx}] {zj['Typ']} ECLIPSE ({LOCATION_NAME})\n{zj['Maksimum (UTC)']} UTC", color='white', fontsize=10, pad=12,
                 weight='bold')
    ax.text(0.04, 0.04, info, transform=ax.transAxes, color='lightgray', fontsize=8,
            bbox=dict(facecolor='#050510', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4'))
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, facecolor='#050510', edgecolor='gray',
                    borderpad=0.4)
    plt.setp(leg.get_texts(), color='white', fontsize=7)

    plt.savefig(f"{output_dir}/{zj['Typ'].lower()}_{date_str}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 2. ZBIORCZE WYKRESY PRZEGLĄDOWE ====================
print("4/7. Generowanie zbiorczych wykresów statycznych...")


def generuj_statyczne_wykresy(dane):
    if not dane:
        print("    Brak zjawisk do wygenerowania wykresu zbiorczego.")
        return
    df = pd.DataFrame(dane)
    df['Lata'] = df['raw_time'].apply(lambda x: x.utc_datetime().year)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), facecolor='#0d0d1a')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#141428')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(color='#282848', linestyle='--', alpha=0.7)

    sun_df = df[df['Typ'] == 'SŁOŃCE']
    moon_df = df[df['Typ'] == 'KSIĘŻYC']

    ax1.scatter(sun_df['Lata'], sun_df['Obscuration (%)'], color='#ffcc00', s=70, label='Słońce (Obscuration %)',
                edgecolors='black', zorder=3)
    ax1.scatter(moon_df['Lata'], moon_df['Obscuration (%)'], color='#ff5555', s=70, label='Księżyc (Faza %)',
                edgecolors='black', zorder=3)
    ax1.set_title(f'Pokrycie / Faza Maksymalna Zaćmień Widocznych z: {LOCATION_NAME} ({START_YEAR}-{END_YEAR})', fontsize=12, pad=10,
                  weight='bold')
    ax1.set_ylabel('Obscuration / Faza (%)')
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor='#1e1e38', edgecolor='none', labelcolor='white')

    ax2.scatter(sun_df['Azymut (°)'], sun_df['Wysokość (°)'], color='#ffcc00', s=sun_df['Obscuration (%)'] * 1.5,
                label='Słońce', alpha=0.8, edgecolors='black')
    ax2.scatter(moon_df['Azymut (°)'], moon_df['Wysokość (°)'], color='#ff5555', s=moon_df['Obscuration (%)'] * 1.5,
                label='Księżyc', alpha=0.8, edgecolors='black')
    ax2.set_title('Położenie Zjawisk na Niebie w Momentach Maksimum (Wysokość vs Azymut)', fontsize=12, pad=10,
                  weight='bold')
    ax2.set_xlabel('Azymut (°) [0=N, 90=E, 180=S, 270=W]')
    ax2.set_ylabel('Wysokość nad Horyzontem (°)')
    ax2.set_xlim(0, 360)
    ax2.set_ylim(0, 90)
    ax2.legend(facecolor='#1e1e38', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    stat_chart_path = f"{output_dir}/statystyka_zacmien_{loc_slug}.png"
    plt.savefig(stat_chart_path, dpi=200, bbox_inches='tight')
    plt.close()


generuj_statyczne_wykresy(tabela_danych)

# ==================== 3. MAPY NIEBA (.PNG) ====================
print("5/7. Generowanie map nieba w projekcji polarnej...")


def generuj_mape_nieba(zj, file_path):
    t = zj['raw_time']
    fig = plt.figure(figsize=(7, 7), facecolor='#0b0d19')
    ax = fig.add_subplot(111, polar=True, facecolor='#050714')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    st_alt, st_az, _ = get_apparent_altaz_with_refraction(user_obs, star_objects, t)
    visible_stars = st_alt > 0
    r_stars = 90 - st_alt[visible_stars]
    theta_stars = np.radians(st_az[visible_stars])
    mags = bright_stars['magnitude'][visible_stars]
    sizes = np.clip((4.5 - mags) ** 2, 1, 25)
    ax.scatter(theta_stars, r_stars, s=sizes, color='white', alpha=0.8, zorder=2)

    for p_name, p_obj in planets.items():
        p_alt, p_az, _ = get_apparent_altaz_with_refraction(user_obs, p_obj, t)
        if p_alt > 0:
            ax.scatter(np.radians(p_az), 90 - p_alt, color='#ffaa33', s=35, zorder=3)
            ax.text(np.radians(p_az), 90 - p_alt + 3.5, p_name, color='#b0b0b0', fontsize=7.5, ha='center', zorder=3)

    s_alt, s_az, _ = get_apparent_altaz_with_refraction(user_obs, sun, t)
    m_alt, m_az, _ = get_apparent_altaz_with_refraction(user_obs, moon, t)

    if s_alt > 0:
        ax.scatter(np.radians(s_az), 90 - s_alt, color='gold', s=80, label='Słońce', zorder=4)
    if m_alt > 0:
        ax.scatter(np.radians(m_az), 90 - m_alt, color='lightblue', s=60, label='Księżyc', zorder=4)

    ax.set_rlim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(['90°', '60°', '30°', '0° (Horyzont)'], color='gray', fontsize=7)

    kierunki_rad = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ax.set_xticks(kierunki_rad)
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], color='orange', fontsize=9, weight='bold')
    ax.grid(color='#222b45', linestyle='--', linewidth=0.6)

    title_text = f"MAPA NIEBA - {LOCATION_NAME.upper()}\nZaćmienie {zj['Typ']} | {zj['Maksimum (UTC)']} UTC"
    ax.set_title(title_text, color='white', fontsize=10, pad=15, weight='bold')

    plt.savefig(file_path, dpi=200, bbox_inches='tight')
    plt.close()


for idx, zj in enumerate(tabela_danych[:5], start=1):
    date_str = zj['raw_time'].utc_strftime('%Y-%m-%d_%H-%M-%S')
    mapa_path = f"{output_dir}/mapa_nieba_{zj['Typ'].lower()}_{date_str}.png"
    generuj_mape_nieba(zj, mapa_path)

# ==================== 4. EKSPORT DO KALENDARZA .ICS ====================
print("6/7. Tworzenie pliku kalendarza iCS...")


def eksportuj_do_ics(lista_zjawisk, nazwa_pliku_ics):
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:-//Astronomia {LOCATION_NAME}//Katalog Zacmien//PL",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH"
    ]

    for zj in lista_zjawisk:
        t_max = zj['raw_time']
        t_pocz = zj['t_pocz_raw']
        t_kon = zj['t_kon_raw']

        dt_start = t_pocz.utc_strftime('%Y%m%dT%H%M%SZ')
        dt_end = t_kon.utc_strftime('%Y%m%dT%H%M%SZ')
        dt_stamp = t_max.utc_strftime('%Y%m%dT%H%M%SZ')

        summary = f"Zaćmienie {zj['Typ'].capitalize()} - {LOCATION_NAME} (Obscuration {zj['Obscuration (%)']}%)"
        description = (f"Zaćmienie {zj['Typ']} widoczne z lokalizacji: {LOCATION_NAME}.\\n"
                       f"Maksimum: {zj['Maksimum (UTC)']} UTC\\n"
                       f"Magnituda: {zj['Magnituda']}\\n"
                       f"Obscuration / Powierzchnia: {zj['Obscuration (%)']}%\\n"
                       f"Wysokość: {zj['Wysokość (°)']}° | Azymut: {zj['Azymut (°)']}°")

        uid = f"eclipse-{t_max.utc_strftime('%Y%m%dT%H%M%S')}@{loc_slug}.astro"

        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dt_stamp}",
            f"DTSTART:{dt_start}",
            f"DTEND:{dt_end}",
            f"SUMMARY:{summary}",
            f"DESCRIPTION:{description}",
            f"LOCATION:{LOCATION_NAME} ({OBSERVER_LATITUDE}°N\\, {OBSERVER_LONGITUDE}°E)",
            "BEGIN:VALARM",
            "TRIGGER:-P1D",
            "ACTION:DISPLAY",
            f"DESCRIPTION:Jutro zaćmienie {zj['Typ']} - {LOCATION_NAME}!",
            "END:VALARM",
            "BEGIN:VALARM",
            "TRIGGER:-PT1H",
            "ACTION:DISPLAY",
            f"DESCRIPTION:Za godzinę rozpocznie się zaćmienie {zj['Typ']}!",
            "END:VALARM",
            "END:VEVENT"
        ])

    lines.append("END:VCALENDAR")

    path_ics = f"{output_dir}/{nazwa_pliku_ics}"
    with open(path_ics, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


eksportuj_do_ics(tabela_danych, f"zacmienia_{loc_slug}_kalendarz.ics")

# ==================== 5. KATALOG ANIMOWANY HTML ====================
print("7/7. Generowanie interaktywnego odtwarzacza HTML Canvas...")


def wygeneruj_katalog_html(lista_zjawisk, nazwa_pliku_html):
    skompilowane_dane = []

    for idx, zj in enumerate(lista_zjawisk):
        jd_vector = np.linspace(zj['t_pocz_raw'].tt, zj['t_kon_raw'].tt, 40)
        t_klatki = ts.tt_jd(jd_vector)
        dx_list, dy_list, czas_list = [], [], []

        if zj['Typ'] == 'SŁOŃCE':
            s_alt, s_az, _ = get_apparent_altaz_with_refraction(user_obs, sun, t_klatki)
            m_alt, m_az, _ = get_apparent_altaz_with_refraction(user_obs, moon, t_klatki)
            dx_all = -(m_az - s_az) * np.cos(np.radians(s_alt))
            dx_all = (dx_all + 180) % 360 - 180
            dy_all = m_alt - s_alt
        else:
            e_sun = earth.at(t_klatki).observe(sun).apparent()
            s_lat, s_lon, _ = e_sun.frame_latlon(ecliptic_frame)
            sh_lon = (s_lon.degrees + 180.0) % 360.0
            sh_lat = -s_lat.degrees

            e_moon = earth.at(t_klatki).observe(moon).apparent()
            m_lat, m_lon, _ = e_moon.frame_latlon(ecliptic_frame)
            dx_all = -(m_lon.degrees - sh_lon + 180) % 360 - 180
            dy_all = m_lat.degrees - sh_lat

        for k in range(len(t_klatki)):
            dx_list.append(round(float(dx_all[k]), 4))
            dy_list.append(round(float(dy_all[k]), 4))
            czas_list.append(t_klatki[k].utc_strftime('%H:%M:%S'))

        skompilowane_dane.append({
            'idx': idx + 1, 'typ': zj['Typ'], 'data_max': zj['Maksimum (UTC)'],
            'magnituda': zj['Magnituda'], 'obscuration': zj['Obscuration (%)'],
            'wysokosc': zj['Wysokość (°)'], 'azymut': zj['Azymut (°)'],
            'r_moon': round(float(zj['r_moon']), 4),
            'r_sun': round(float(zj.get('r_sun', 0)), 4),
            'r_umbra': round(float(zj.get('r_umbra', 0)), 4),
            'r_penumbra': round(float(zj.get('r_penumbra', 0)), 4),
            'dx': dx_list, 'dy': dy_list, 'czasy': czas_list
        })

    json_data = json.dumps(skompilowane_dane)

    html_code = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Katalog Zaćmień - {LOCATION_NAME} ({START_YEAR}-{END_YEAR})</title>
    <style>
        body {{ background-color: #0d0d1a; color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 0; padding: 20px; }}
        .card {{ background: #16162a; border-radius: 12px; padding: 20px; max-width: 520px; margin: auto; box-shadow: 0 6px 20px rgba(0,0,0,0.6); border: 1px solid #2a2a4a; }}
        canvas {{ background-color: #050510; border-radius: 8px; margin-top: 10px; border: 1px solid #222244; }}
        .nav-btns {{ margin: 12px 0; display: flex; justify-content: space-between; gap: 8px; }}
        .ctrl-panel {{ background: #0f0f22; border-radius: 8px; padding: 12px; margin-top: 12px; border: 1px solid #222244; }}
        .btn-row {{ display: flex; justify-content: center; gap: 8px; margin-bottom: 10px; align-items: center; }}
        button {{ background: #2a2a4a; color: white; border: 1px solid #4a4a7a; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: bold; flex: 1; transition: 0.2s; }}
        button:hover {{ background: #3b3b6b; border-color: #6a6ahf; }}
        .btn-action {{ background: #1e3a5f; border-color: #2b5b96; }}
        .btn-action:hover {{ background: #2b5b96; }}
        .slider-box {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; font-size: 12px; color: #aaa; margin-top: 8px; }}
        input[type=range] {{ flex: 1; cursor: pointer; }}
        .info-box {{ background: #090915; padding: 12px; border-radius: 6px; text-align: left; font-size: 13.5px; margin-top: 12px; line-height: 1.6; border: 1px solid #1c1c38; }}
        .badge {{ background: #e67e22; color: black; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .speed-label {{ font-size: 12px; color: #00d2ff; font-weight: bold; font-family: monospace; }}
    </style>
</head>
<body>

<div class="card">
    <h2 id="title" style="margin-top:0; font-size: 20px;">Zaćmienie</h2>

    <div class="nav-btns">
        <button onclick="prevEvent()">&#9668; Poprzednie</button>
        <button onclick="nextEvent()">Następne &#9658;</button>
    </div>

    <canvas id="eclipseCanvas" width="380" height="380"></canvas>

    <div class="ctrl-panel">
        <div class="btn-row">
            <button onclick="adjustSpeed(20)" class="btn-action" title="Zwolnij animację">Szybciej (+)</button>
            <button onclick="togglePlay()" id="playBtn" style="flex: 1.2;">&#10074;&#10074; Pauza</button>
            <button onclick="adjustSpeed(-20)" class="btn-action" title="Przyspiesz animację">Wolniej (-)</button>
        </div>

        <div class="slider-box">
            <span>Szybko</span>
            <input type="range" id="speedSlider" min="20" max="250" value="90" oninput="onSliderChange(this.value)">
            <span>Wolno</span>
        </div>
        <div style="margin-top: 6px;" class="speed-label" id="speedStatus">Opóźnienie: 90 ms (ok. 11 FPS)</div>
    </div>

    <div class="info-box" id="infoBox"></div>
</div>

<script>
    const data = {json_data};
    let currentIdx = 0;
    let currentFrame = 0;
    let isPlaying = true;
    let timer = null;
    let delayMs = 90;

    const canvas = document.getElementById('eclipseCanvas');
    const ctx = canvas.getContext('2d');

    function draw() {{
        if (!data || data.length === 0) return;
        const item = data[currentIdx];
        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const scale = item.typ === 'SŁOŃCE' ? 250 : 90;

        ctx.clearRect(0, 0, w, h);

        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
        ctx.moveTo(0, cy); ctx.lineTo(w, cy);
        ctx.stroke();

        const dx = item.dx[currentFrame] * scale;
        const dy = -item.dy[currentFrame] * scale;

        if (item.typ === 'SŁOŃCE') {{
            ctx.fillStyle = '#ffcc00';
            ctx.beginPath();
            ctx.arc(cx, cy, item.r_sun * scale, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#050510';
            ctx.beginPath();
            ctx.arc(cx + dx, cy + dy, item.r_moon * scale, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#666';
            ctx.stroke();
        }} else {{
            ctx.fillStyle = 'rgba(200,200,200,0.15)';
            ctx.beginPath();
            ctx.arc(cx, cy, item.r_penumbra * scale, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = 'rgba(100,20,20,0.7)';
            ctx.beginPath();
            ctx.arc(cx, cy, item.r_umbra * scale, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = item.obscuration >= 100 ? '#cc5533' : '#dddddd';
            ctx.beginPath();
            ctx.arc(cx + dx, cy + dy, item.r_moon * scale, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.stroke();
        }}

        document.getElementById('title').innerHTML = `<span class="badge">[${{item.idx}} / ${{data.length}}]</span> ZAĆMIENIE ${{item.typ}}`;
        document.getElementById('infoBox').innerHTML = `
            <b>Lokalizacja:</b> {LOCATION_NAME}<br>
            <b>Maksimum (UTC):</b> ${{item.data_max}}<br>
            <b>Czas klatki (UTC):</b> ${{item.czasy[currentFrame]}}<br>
            <b>Magnituda:</b> ${{item.magnituda}} | <b>Obscuration:</b> ${{item.obscuration}}%<br>
            <b>Wysokość (z refrakcją):</b> ${{item.wysokosc}}° | <b>Azymut:</b> ${{item.azymut}}°
        `;
    }}

    function nextFrame() {{
        if (!data || data.length === 0) return;
        currentFrame = (currentFrame + 1) % data[currentIdx].dx.length;
        draw();
    }}

    function resetTimer() {{
        if (timer) clearInterval(timer);
        if (isPlaying) {{
            timer = setInterval(nextFrame, delayMs);
        }}
    }}

    function updateSpeedUI() {{
        document.getElementById('speedSlider').value = delayMs;
        const fps = Math.round(1000 / delayMs);
        document.getElementById('speedStatus').innerText = `Opóźnienie: ${{delayMs}} ms (ok. ${{fps}} FPS)`;
        resetTimer();
    }}

    function onSliderChange(val) {{
        delayMs = parseInt(val, 10);
        updateSpeedUI();
    }}

    function adjustSpeed(delta) {{
        delayMs = Math.max(20, Math.min(250, delayMs - delta));
        updateSpeedUI();
    }}

    function togglePlay() {{
        isPlaying = !isPlaying;
        const btn = document.getElementById('playBtn');
        if (isPlaying) {{
            btn.innerHTML = '&#10074;&#10074; Pauza';
        }} else {{
            btn.innerHTML = '&#9654; Odtwarzaj';
        }}
        resetTimer();
    }}

    function nextEvent() {{
        if (!data || data.length === 0) return;
        currentIdx = (currentIdx + 1) % data.length;
        currentFrame = 0;
        draw();
    }}

    function prevEvent() {{
        if (!data || data.length === 0) return;
        currentIdx = (currentIdx - 1 + data.length) % data.length;
        currentFrame = 0;
        draw();
    }}

    document.addEventListener('keydown', function(e) {{
        if (e.key === "ArrowRight") nextEvent();
        if (e.key === "ArrowLeft") prevEvent();
        if (e.code === "Space") {{ e.preventDefault(); togglePlay(); }}
        if (e.key === "+" || e.key === "=" || e.code === "NumpadAdd") adjustSpeed(20);
        if (e.key === "-" || e.key === "_" || e.code === "NumpadSubtract") adjustSpeed(-20);
    }});

    draw();
    resetTimer();
</script>

</body>
</html>"""

    path_html = f"{output_dir}/{nazwa_pliku_html}"
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html_code)


wygeneruj_katalog_html(tabela_danych, f"katalog_zacmien_{loc_slug}.html")

# ==================== 6. CZYSZCZENIE KLUCZY, ZAPIS DO CSV I EKRAN ====================
for item in tabela_danych:
    for klucz in ['raw_time', 'sep', 'alt_raw', 'az_raw', 't_pocz_raw', 't_kon_raw', 'r_sun', 'r_moon', 'r_umbra',
                  'r_penumbra']:
        item.pop(klucz, None)

df_wyniki = pd.DataFrame(tabela_danych)
csv_filename = f"{output_dir}/podsumowanie_zacmien_{loc_slug}.csv"
df_wyniki.to_csv(csv_filename, index=False, encoding='utf-8-sig')

# Wyświetlanie wyników w konsoli
print("\n" + "=" * 85)
print(f" PODSUMOWANIE ZACMIEN DLA: {LOCATION_NAME.upper()} ({START_YEAR}–{END_YEAR})")
print("=" * 85)
if not df_wyniki.empty:
    print(df_wyniki.to_string(index=False))
else:
    print("Nie znaleziono widocznych zaćmień w podanym zakresie dla tej lokalizacji.")

print("\n" + "=" * 85)
print("[OK] Generowanie plików zakończone powodzeniem!")
print(f" Pomyślnie utworzono w katalogu '{output_dir}/':")
print(f"  - podsumowanie_zacmien_{loc_slug}.csv")
print(f"  - zacmienia_{loc_slug}_kalendarz.ics")
print(f"  - katalog_zacmien_{loc_slug}.html")
print(f"  - statystyka_zacmien_{loc_slug}.png oraz wykresy i mapy nieba PNG")
print("=" * 85 + "\n")

# Automatyczne otwarcie wygenerowanego pliku HTML w domyślnej przeglądarce
html_path = os.path.abspath(f"{output_dir}/katalog_zacmien_{loc_slug}.html")
if os.path.exists(html_path):
    webbrowser.open(f"file://{html_path}")