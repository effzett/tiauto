# Toleranzintervall-Rechner

Standalone-Tool zur automatischen Berechnung von Toleranzintervallen.
Daten eingeben, Button drücken – der Rest wird automatisch entschieden.

## Entscheidungslogik

1. **Verteilungsfrei** – wenn n groß genug (z.B. n ≥ 93 für 95/95 zweiseitig)
2. **Normal** – Shapiro-Wilk auf Rohdaten bestanden → k-Faktor-Verfahren
3. **Lognormal** – Shapiro-Wilk auf log(x) bestanden → k-Faktor auf log-Skala
4. **Weibull** – Monte-Carlo-KS-Test bestanden → parametrischer Bootstrap
5. **Fallback** – Normal-TI mit deutlicher Warnung

## Setup in PyCharm

1. Projekt öffnen: **File → Open → diesen Ordner wählen**
2. Python-Interpreter einrichten: **Settings → Project → Python Interpreter**
3. venv anlegen und Abhängigkeiten installieren:
   ```
   pip install -r requirements.txt
   ```
4. Run-Konfigurationen sind vorkonfiguriert im Dropdown:
   - **GUI starten** – Startet die PySide6-Oberfläche
   - **Demo (Konsole)** – Führt die Demos in der Konsole aus
   - **Tests (pytest)** – Lässt die Tests laufen
   - **EXE bauen (PyInstaller)** – Erzeugt eine standalone Binary

## Standalone-Binary erzeugen

```bash
pip install -r requirements-dev.txt
python build.py
```

Ergebnis: `dist/Toleranzintervall-Rechner.exe` (Windows) bzw. `dist/Toleranzintervall-Rechner` (Linux/Mac)

## Projektstruktur

```
toleranzintervall-rechner/
├── .run/                           # PyCharm Run-Konfigurationen
│   ├── GUI starten.run.xml
│   ├── Demo (Konsole).run.xml
│   ├── Tests (pytest).run.xml
│   └── EXE bauen (PyInstaller).run.xml
├── src/
│   ├── __init__.py
│   ├── tolerance_intervals.py      # Rechenlogik (auch standalone nutzbar)
│   └── ti_gui.py                   # PySide6 GUI
├── tests/
│   └── test_tolerance.py
├── main.py                         # Einstiegspunkt
├── build.py                        # PyInstaller-Build
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── .gitignore
```

## Schnellstart ohne GUI

```python
from src.tolerance_intervals import auto_tolerance_interval

result = auto_tolerance_interval([4.2, 4.5, 4.1, 4.8, 4.3, 4.6, 4.4])
```
