# 🌒 Eclipse Explorer

**Eclipse Explorer** is an advanced yet easy-to-use Python script designed for precise calculation, analysis, and visualization of **Solar and Lunar eclipses** for any location on Earth within a chosen timeframe.

The program leverages NASA's high-precision **DE421 ephemerides** via the `Skyfield` library, taking into account atmospheric refraction and realistic shadow geometries.

---

## 🤖 Built with AI Collaboration

This project was developed in collaboration with **Gemini**, an AI assistant by Google, which helped write, optimize, and structure the Python codebase, HTML interactive canvas player, and project documentation.

---

## 🚀 Key Features

* 📍 **Fully Customizable Location & Timeframe:** Easily modify geographic coordinates (latitude/longitude) and observation years in the clear configuration section at the top of the script.
* ☀️ 🌙 **Solar & Lunar Eclipse Support:** Calculates exact maximum times, phase/obscuration percentages, magnitudes, as well as altitude and azimuth of celestial bodies.
* 📊 **Automated Data Export:**
  * **CSV Summary Table:** Detailed breakdown of all detected eclipse events.
  * **`.ics` Calendar File:** Import events with built-in alarms directly into Google Calendar, Apple Calendar, or Microsoft Outlook.
* 🎨 **Rich Visualization Suite:**
  * **300 DPI Geometry Diagrams:** High-resolution static renderings showing the coverage and alignment for each event.
  * **Polar Sky Maps:** Star charts (Hipparcos catalog) and positions of bright planets at the exact moment of eclipse maximum.
  * **Summary Charts:** Multi-year statistical analysis of obscuration and sky position.
* 🌐 **Interactive HTML Catalog (Canvas):** Generates a standalone, dependency-free HTML file with a built-in animation player, adjustable playback speed (FPS), pause controls, and keyboard navigation.

---

## 🛠️ Requirements & Installation

### 1. Prerequisites
* **Python** 3.8 or higher.

### 2. Required Libraries
Install the necessary dependencies using `pip`:

```bash
pip install numpy pandas matplotlib skyfield

```

---

## ⚙️ Configuration & Usage

### 1. Setting Up Parameters

Open `eclipse_calculator.py` in your code editor and adjust the parameters in the **`OBSERVATION CONFIGURATION`** section at the top:

```python
# ==============================================================================
# OBSERVATION CONFIGURATION
# ==============================================================================

# 1. OBSERVER LOCATION
OBSERVER_LATITUDE  = 51.6174   # (+) for North (N), (-) for South (S)
OBSERVER_LONGITUDE = 15.3082   # (+) for East (E), (-) for West (W)
LOCATION_NAME      = "Żagań"   # Location name (used in filenames, charts, and calendar)

# 2. TIMEFRAME FOR ECLIPSE SEARCH
START_YEAR  = 2000
START_MONTH = 1
START_DAY   = 1

END_YEAR    = 2036
END_MONTH   = 12
END_DAY     = 31
# ==============================================================================

```

### 2. Running the Script

Execute the script from your terminal:

```bash
python eclipse_calculator.py

```

*Note: On the first run, the script will automatically download the required NASA ephemeris file (`de421.bsp`) and star catalogs into the `dane_astro/` directory.*

---

## 📁 Output File Structure

Once execution finishes, an output directory named after your configured location (e.g., `zacmienia_żagań/`) will be generated:

```text
zacmienia_[location_name]/
 ├── katalog_zacmien_[location].html   # Interactive HTML animation player
 ├── zacmienia_[location]_kalendarz.ics # iCalendar file with events & alarms
 ├── podsumowanie_zacmien_[location].csv # CSV table containing calculated data
 ├── statystyka_zacmien_[location].png  # Overview summary charts (phase & sky altitude)
 ├── słońce_YYYY-MM-DD_HH-MM-SS.png       # Geometry charts for Solar eclipses
 ├── księżyc_YYYY-MM-DD_HH-MM-SS.png      # Geometry charts for Lunar eclipses
 └── mapa_nieba_...png                    # Polar sky projection maps

```

---

## ⌨️ HTML Player Controls

The interactive HTML catalog provides convenient navigation:

* **Left / Right Arrows (`←` / `→`):** Switch between eclipse events.
* **Spacebar (`Space`):** Play / Pause the animation.
* **Plus / Minus (`+` / `-`):** Increase or decrease playback speed.

---

## 📜 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
