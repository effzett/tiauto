#!/usr/bin/env python3
"""
Toleranzintervall-Rechner – PySide6 GUI

Standalone-Tool zur automatischen Berechnung von Toleranzintervallen.
Einfach Daten eingeben oder aus CSV laden, Parameter setzen, berechnen.

Autor: Frank / Claude
"""

import sys
import io
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QTextEdit, QPlainTextEdit, QPushButton,
    QComboBox, QDoubleSpinBox, QGroupBox, QFileDialog, QSplitter,
    QMessageBox, QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt, QLocale, QUrl
from PySide6.QtGui import QFont, QAction, QKeySequence, QDesktopServices

# Toleranzintervall-Logik importieren
from src.tolerance_intervals import (
    auto_tolerance_interval, print_min_n_table, min_n_distribution_free,
)


class ToleranceIntervalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toleranzintervall-Rechner v1.3.0")
        self.setMinimumSize(900, 700)
        self.setup_ui()
        self.setup_menu()
        self.statusBar().showMessage("Bereit – Daten eingeben und berechnen.")

    # ── Menü ──────────────────────────────────────────────────

    def setup_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&Datei")

        open_action = QAction("CSV &laden...", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.load_csv)
        file_menu.addAction(open_action)

        save_action = QAction("Daten &speichern...", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.save_data)
        file_menu.addAction(save_action)

        paste_action = QAction("Aus &Zwischenablage einfügen", self)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_action.triggered.connect(self.paste_clipboard)
        file_menu.addAction(paste_action)

        file_menu.addSeparator()

        report_action = QAction("&Report erstellen...", self)
        report_action.setShortcut(QKeySequence("Ctrl+R"))
        report_action.triggered.connect(self.generate_report)
        file_menu.addAction(report_action)

        file_menu.addSeparator()

        quit_action = QAction("&Beenden", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = menu.addMenu("&Hilfe")

        table_action = QAction("&Mindeststichprobengrößen", self)
        table_action.triggered.connect(self.show_min_n_table)
        help_menu.addAction(table_action)

        doc_action = QAction("&Dokumentation (PDF)...", self)
        doc_action.triggered.connect(self.open_documentation)
        help_menu.addAction(doc_action)

        help_menu.addSeparator()

        about_action = QAction("&Über...", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    # ── UI aufbauen ───────────────────────────────────────────

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # ── Oberer Bereich: Eingabe + Parameter ──
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Daten-Eingabe
        data_group = QGroupBox("Messwerte")
        data_layout = QVBoxLayout(data_group)

        hint = QLabel("Ein Wert pro Zeile, oder Komma-/Semikolon-/Tab-getrennt:")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        data_layout.addWidget(hint)

        self.data_input = QPlainTextEdit()
        self.data_input.setPlaceholderText(
            "4.2\n4.5\n4.1\n4.8\n4.3\n...\n\n"
            "oder: 4.2, 4.5, 4.1, 4.8, 4.3\n\n"
            "oder CSV laden (Strg+O)"
        )
        self.mono = QFont("Menlo")  # macOS
        if not self.mono.exactMatch():
            self.mono = QFont("Consolas")  # Windows
        if not self.mono.exactMatch():
            self.mono.setFamily("monospace")  # Linux Fallback
        self.mono.setPointSize(11)
        self.data_input.setFont(self.mono)
        data_layout.addWidget(self.data_input)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("CSV laden...")
        load_btn.clicked.connect(self.load_csv)
        btn_row.addWidget(load_btn)

        save_btn = QPushButton("Daten speichern...")
        save_btn.clicked.connect(self.save_data)
        btn_row.addWidget(save_btn)

        clear_btn = QPushButton("Leeren")
        clear_btn.clicked.connect(self.data_input.clear)
        btn_row.addWidget(clear_btn)

        example_btn = QPushButton("Beispieldaten")
        example_btn.clicked.connect(self.load_example)
        btn_row.addWidget(example_btn)

        btn_row.addStretch()
        data_layout.addLayout(btn_row)
        top_layout.addWidget(data_group, stretch=3)

        # Parameter
        param_group = QGroupBox("Parameter")
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(8)

        row = 0
        param_layout.addWidget(QLabel("Abdeckung (p):"), row, 0)
        self.p_spin = QDoubleSpinBox()
        self.p_spin.setRange(0.01, 0.999)
        self.p_spin.setSingleStep(0.01)
        self.p_spin.setDecimals(3)
        self.p_spin.setValue(0.90)
        self.p_spin.setToolTip("Mindestanteil der Population, den das Intervall abdecken soll")
        param_layout.addWidget(self.p_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Konfidenz:"), row, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.50, 0.999)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(3)
        self.conf_spin.setValue(0.95)
        self.conf_spin.setToolTip("Konfidenzniveau für das Toleranzintervall")
        param_layout.addWidget(self.conf_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Seite:"), row, 0)
        self.side_combo = QComboBox()
        self.side_combo.addItems(["Zweiseitig", "Einseitig unten", "Einseitig oben"])
        self.side_combo.setToolTip(
            "Zweiseitig: [untere, obere] Grenze\n"
            "Einseitig unten: [untere Grenze, ∞)\n"
            "Einseitig oben: (-∞, obere Grenze]"
        )
        param_layout.addWidget(self.side_combo, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Shapiro α:"), row, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.20)
        self.alpha_spin.setSingleStep(0.01)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setToolTip("Signifikanzniveau für die Verteilungstests")
        param_layout.addWidget(self.alpha_spin, row, 1)

        row += 1
        param_layout.addWidget(QLabel("Methode:"), row, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Automatisch",
            "Verteilungsfrei",
            "Normal (k-Faktor)",
            "Lognormal",
            "Weibull (Bootstrap)",
        ])
        self.method_combo.setToolTip(
            "Automatisch: Beste Methode wird gewählt\n"
            "Verteilungsfrei: Ordnungsstatistiken (braucht viele Daten)\n"
            "Normal: k-Faktor-Methode\n"
            "Lognormal: k-Faktor auf log-Skala\n"
            "Weibull: Parametrischer Bootstrap"
        )
        param_layout.addWidget(self.method_combo, row, 1)

        row += 1
        param_layout.addWidget(QLabel(""), row, 0)  # Spacer

        # Min-n Anzeige
        row += 1
        self.min_n_label = QLabel()
        self.min_n_label.setStyleSheet("color: #555; font-size: 11px;")
        self.min_n_label.setWordWrap(True)
        param_layout.addWidget(self.min_n_label, row, 0, 1, 2)
        self.update_min_n_label()

        # Signals für live min_n update
        self.p_spin.valueChanged.connect(self.update_min_n_label)
        self.conf_spin.valueChanged.connect(self.update_min_n_label)
        self.side_combo.currentIndexChanged.connect(self.update_min_n_label)
        self.method_combo.currentIndexChanged.connect(self.update_min_n_label)

        row += 1
        param_layout.setRowStretch(row, 1)

        # Berechnen-Button
        row += 1
        self.calc_btn = QPushButton("▶  Berechnen")
        self.calc_btn.setMinimumHeight(45)
        self.calc_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        self.calc_btn.clicked.connect(self.calculate)
        param_layout.addWidget(self.calc_btn, row, 0, 1, 2)

        top_layout.addWidget(param_group, stretch=1)
        splitter.addWidget(top_widget)

        # ── Unterer Bereich: Ergebnisse ──
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        result_tabs = QTabWidget()

        # Tab 1: Ergebnis-Log
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(self.mono)
        self.result_text.setPlaceholderText("Hier erscheint das Ergebnis nach der Berechnung...")
        result_tabs.addTab(self.result_text, "Ergebnis && Entscheidungslog")

        # Tab 2: Daten-Übersicht
        self.stats_text = QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(self.mono)
        self.stats_text.setPlaceholderText("Deskriptive Statistiken der eingegebenen Daten...")
        result_tabs.addTab(self.stats_text, "Deskriptive Statistik")

        result_layout.addWidget(result_tabs)

        # Report-Button
        report_row = QHBoxLayout()
        report_row.addStretch()
        self.report_btn = QPushButton("📄  Report als PDF speichern...")
        self.report_btn.setMinimumHeight(32)
        self.report_btn.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
                border: none;
                padding: 4px 16px;
            }
            QPushButton:hover { background-color: #047857; }
            QPushButton:pressed { background-color: #065f46; }
        """)
        self.report_btn.clicked.connect(self.generate_report)
        report_row.addWidget(self.report_btn)
        report_row.addStretch()
        result_layout.addLayout(report_row)

        splitter.addWidget(result_widget)

        splitter.setSizes([350, 350])

    # ── Hilfsfunktionen ───────────────────────────────────────

    def get_side_string(self) -> str:
        idx = self.side_combo.currentIndex()
        return ['two-sided', 'lower', 'upper'][idx]

    def get_method_string(self) -> str:
        idx = self.method_combo.currentIndex()
        return ['auto', 'distribution_free', 'normal',
                'lognormal', 'weibull'][idx]

    def update_min_n_label(self):
        p = self.p_spin.value()
        conf = self.conf_spin.value()
        side = self.get_side_string()
        method = self.get_method_string()

        # Shapiro-α nur bei automatischer Wahl relevant
        self.alpha_spin.setEnabled(method == 'auto')

        try:
            min_n = min_n_distribution_free(p, conf, side)
            side_text = "zweiseitig" if side == 'two-sided' else "einseitig"
            if method == 'auto':
                self.min_n_label.setText(
                    f"ℹ Verteilungsfrei (Schritt 4)\n"
                    f"  benötigt min. n = {min_n} ({side_text})\n"
                    f"  für p={p:.3f}, conf={conf:.3f}"
                )
            elif method == 'distribution_free':
                self.min_n_label.setText(
                    f"⚠ Verteilungsfrei erzwungen\n"
                    f"  Benötigt min. n = {min_n} ({side_text})\n"
                    f"  für p={p:.3f}, conf={conf:.3f}"
                )
            else:
                self.min_n_label.setText(
                    f"ℹ Methode erzwungen: {method}\n"
                    f"  Kein Verteilungstest, kein min-n nötig."
                )
        except Exception:
            self.min_n_label.setText("")

    def parse_data(self) -> np.ndarray:
        """Parst die Eingabe – flexibel: Zeilen, Komma, Semikolon, Tab."""
        text = self.data_input.toPlainText().strip()
        if not text:
            raise ValueError("Keine Daten eingegeben.")

        # Semikolon und Tab durch Komma ersetzen, dann splitten
        text = text.replace(';', ',').replace('\t', ',')
        # Deutsches Dezimalkomma → Punkt (nur wenn kein Punkt im Wert)
        # Zeilenweise verarbeiten
        values = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            for part in parts:
                if not part:
                    continue
                # Versuche deutsche Notation (Komma als Dezimaltrenner)
                # nur wenn kein Punkt vorhanden
                try:
                    values.append(float(part))
                except ValueError:
                    # Versuch: deutsches Format 4,2 → 4.2
                    try:
                        values.append(float(part.replace(',', '.')))
                    except ValueError:
                        raise ValueError(f"Kann '{part}' nicht als Zahl lesen.")

        if len(values) < 2:
            raise ValueError(f"Mindestens 2 Messwerte nötig, habe {len(values)}.")

        return np.array(values)

    # ── Aktionen ──────────────────────────────────────────────

    def calculate(self):
        try:
            data = self.parse_data()
        except ValueError as e:
            QMessageBox.warning(self, "Eingabefehler", str(e))
            return

        p = self.p_spin.value()
        conf = self.conf_spin.value()
        side = self.get_side_string()
        alpha = self.alpha_spin.value()
        method = self.get_method_string()

        # Ergebnis berechnen, verbose-Output abfangen
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        try:
            result = auto_tolerance_interval(
                data, p=p, confidence=conf, side=side,
                method=method, alpha_shapiro=alpha, verbose=True,
            )
        except Exception as e:
            sys.stdout = old_stdout
            QMessageBox.critical(self, "Berechnungsfehler", str(e))
            return
        finally:
            sys.stdout = old_stdout

        self.result_text.setPlainText(buf.getvalue())

        # Deskriptive Statistik
        stats_lines = [
            f"Anzahl Messwerte:  {len(data)}",
            f"Minimum:           {np.min(data):.6g}",
            f"Maximum:           {np.max(data):.6g}",
            f"Spannweite:        {np.ptp(data):.6g}",
            f"Mittelwert:        {np.mean(data):.6g}",
            f"Median:            {np.median(data):.6g}",
            f"Std.abw. (n-1):    {np.std(data, ddof=1):.6g}",
            f"Variationskoeff.:  {np.std(data, ddof=1)/np.mean(data)*100:.2f}%"
                if np.mean(data) != 0 else "Variationskoeff.:  n/a",
            f"",
            f"Quartile:",
            f"  Q1 (25%):        {np.percentile(data, 25):.6g}",
            f"  Q2 (50%):        {np.percentile(data, 50):.6g}",
            f"  Q3 (75%):        {np.percentile(data, 75):.6g}",
            f"  IQR:             {np.percentile(data, 75) - np.percentile(data, 25):.6g}",
            f"",
            f"Alle Werte positiv:  {'Ja' if np.all(data > 0) else 'Nein'}",
            f"",
            f"Sortierte Daten:",
        ]
        sorted_data = np.sort(data)
        # Kompakte Darstellung
        per_line = 8
        for i in range(0, len(sorted_data), per_line):
            chunk = sorted_data[i:i+per_line]
            stats_lines.append("  " + "  ".join(f"{v:.6g}" for v in chunk))

        self.stats_text.setPlainText("\n".join(stats_lines))

        self.statusBar().showMessage(
            f"Berechnung abgeschlossen – Methode: {result.method}", 5000
        )

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Datendatei laden", "",
            "CSV-Dateien (*.csv *.txt *.dat);;Alle Dateien (*)"
        )
        if not path:
            return

        try:
            # Versuche die Datei zu lesen
            text = Path(path).read_text(encoding='utf-8')
            # Prüfe ob Header vorhanden (erste Zeile nicht-numerisch)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                first = lines[0].replace(';', ',').replace('\t', ',')
                parts = first.split(',')
                try:
                    float(parts[0])
                except ValueError:
                    # Header überspringen
                    lines = lines[1:]

            self.data_input.setPlainText("\n".join(lines))
            self.statusBar().showMessage(f"Geladen: {path}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Fehler beim Laden", str(e))

    def paste_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if text:
            self.data_input.setPlainText(text)
            self.statusBar().showMessage("Daten aus Zwischenablage eingefügt.", 3000)

    def load_example(self):
        example = "50.3\n48.3\n49.6\n50.4\n51.9"
        self.data_input.setPlainText(example)
        self.statusBar().showMessage("Beispieldaten geladen.", 3000)

    def save_data(self):
        """Speichert die aktuellen Daten aus der Eingabebox als CSV."""
        try:
            data = self.parse_data()
        except ValueError as e:
            QMessageBox.warning(self, "Keine Daten", str(e))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Daten speichern", "messwerte.csv",
            "CSV-Dateien (*.csv);;Text-Dateien (*.txt);;Alle Dateien (*)"
        )
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['Messwert'])
                for val in data:
                    writer.writerow([f'{val:g}'])
            self.statusBar().showMessage(
                f"Gespeichert: {len(data)} Werte → {path}", 5000
            )
        except Exception as e:
            QMessageBox.warning(self, "Fehler beim Speichern", str(e))

    def generate_report(self):
        """Erzeugt einen PDF-Report mit Ergebnissen und deskriptiver Statistik."""
        # Prüfe ob Daten und Ergebnisse vorhanden
        result_text = self.result_text.toPlainText().strip()
        stats_text = self.stats_text.toPlainText().strip()

        if not result_text:
            QMessageBox.information(
                self, "Kein Ergebnis",
                "Bitte zuerst Berechnung durchführen."
            )
            return

        try:
            data = self.parse_data()
        except ValueError as e:
            QMessageBox.warning(self, "Keine Daten", str(e))
            return

        # Rohdaten optional?
        include_raw = QMessageBox.question(
            self, "Rohdaten einschließen?",
            "Sollen die Rohdaten (sortiert) im Report enthalten sein?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        ) == QMessageBox.Yes

        # Speicherort
        default_name = f"TI_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Report speichern", default_name,
            "PDF-Dateien (*.pdf);;Alle Dateien (*)"
        )
        if not path:
            return

        try:
            self._write_pdf_report(path, data, result_text, stats_text,
                                   include_raw)
            self.statusBar().showMessage(f"Report gespeichert: {path}", 5000)
            # Direkt öffnen
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        except ImportError:
            # reportlab nicht verfügbar → Fallback auf Text
            self._write_text_report(path, data, result_text, stats_text,
                                    include_raw)
        except Exception as e:
            QMessageBox.critical(self, "Fehler beim Report", str(e))

    def _write_pdf_report(self, path, data, result_text, stats_text,
                          include_raw):
        """Schreibt den Report als PDF (benötigt reportlab)."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Preformatted,
            HRFlowable, Table, TableStyle, PageBreak,
        )
        from reportlab.lib import colors

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            topMargin=18*mm, bottomMargin=15*mm,
            leftMargin=20*mm, rightMargin=20*mm,
        )

        styles = getSampleStyleSheet()
        sap_blue = HexColor("#1B6FB5")
        sap_dark = HexColor("#1A3E5C")
        sap_light = HexColor("#EDF4FA")

        styles.add(ParagraphStyle('RTitle', parent=styles['Title'],
                                  fontSize=18, leading=22, textColor=sap_dark,
                                  spaceAfter=2*mm, alignment=TA_CENTER))
        styles.add(ParagraphStyle('RSub', parent=styles['Normal'],
                                  fontSize=10, leading=13,
                                  textColor=HexColor("#666666"),
                                  spaceAfter=4*mm, alignment=TA_CENTER))
        styles.add(ParagraphStyle('RSect', parent=styles['Heading2'],
                                  fontSize=13, leading=16, textColor=sap_blue,
                                  spaceBefore=5*mm, spaceAfter=2*mm))
        styles.add(ParagraphStyle('RBody', parent=styles['Normal'],
                                  fontSize=9.5, leading=12.5,
                                  alignment=TA_JUSTIFY, spaceAfter=2*mm))
        styles.add(ParagraphStyle('RMono', parent=styles['Normal'],
                                  fontName='Courier', fontSize=8,
                                  leading=10.5, spaceAfter=1*mm,
                                  leftIndent=4*mm))
        styles.add(ParagraphStyle('RFooter', parent=styles['Normal'],
                                  fontSize=7.5, textColor=HexColor("#999999"),
                                  alignment=TA_CENTER))

        story = []

        # ── Titel ──
        story.append(Paragraph("Toleranzintervall – Report", styles['RTitle']))
        p_val = self.p_spin.value()
        conf_val = self.conf_spin.value()
        side_text = self.side_combo.currentText()
        method_text = self.method_combo.currentText()
        story.append(Paragraph(
            f"Erstellt: {datetime.now():%d.%m.%Y %H:%M} · "
            f"n={len(data)} · p={p_val:.3f} · Konfidenz={conf_val:.3f} · "
            f"{side_text} · Methode: {method_text}",
            styles['RSub']
        ))
        story.append(HRFlowable(width="100%", thickness=1.2, color=sap_blue,
                                spaceAfter=4*mm))

        # ── Ergebnis ──
        story.append(Paragraph("Ergebnis und Entscheidungslog", styles['RSect']))
        for line in result_text.split('\n'):
            story.append(Paragraph(line.replace(' ', '&nbsp;'),
                                   styles['RMono']))

        # ── Deskriptive Statistik ──
        story.append(Paragraph("Deskriptive Statistik", styles['RSect']))

        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        cv = std / mean * 100 if mean != 0 else float('nan')

        stat_data = [
            ['Kenngröße', 'Wert'],
            ['Anzahl (n)', f'{n}'],
            ['Minimum', f'{np.min(data):.6g}'],
            ['Maximum', f'{np.max(data):.6g}'],
            ['Spannweite', f'{np.ptp(data):.6g}'],
            ['Mittelwert', f'{mean:.6g}'],
            ['Median', f'{med:.6g}'],
            ['Std.abw. (n−1)', f'{std:.6g}'],
            ['Variationskoeff.', f'{cv:.2f}%'],
            ['Q1 (25%)', f'{q1:.6g}'],
            ['Q3 (75%)', f'{q3:.6g}'],
            ['IQR', f'{q3 - q1:.6g}'],
        ]

        t = Table(stat_data, colWidths=[55*mm, 50*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), sap_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, sap_light]),
            ('GRID', (0, 0), (-1, -1), 0.4, HexColor("#CCCCCC")),
        ]))
        story.append(t)

        # ── Rohdaten (optional) ──
        if include_raw:
            story.append(Paragraph("Rohdaten (sortiert)", styles['RSect']))
            sorted_data = np.sort(data)

            # Als Tabelle mit mehreren Spalten
            cols = 8
            rows_needed = int(np.ceil(len(sorted_data) / cols))
            raw_table = [['#', 'Wert'] * min(cols, 4)]  # Header

            # Einfacher: kompakte Textdarstellung
            per_line = 10
            for i in range(0, len(sorted_data), per_line):
                chunk = sorted_data[i:i + per_line]
                idx_str = f'[{i+1:>3}–{min(i+per_line, len(sorted_data)):>3}]'
                vals = '  '.join(f'{v:.6g}' for v in chunk)
                story.append(Paragraph(
                    f'{idx_str}&nbsp; {vals}', styles['RMono']
                ))

        # ── Footer ──
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=0.4,
                                color=HexColor("#CCCCCC"), spaceAfter=2*mm))
        story.append(Paragraph(
            f"Toleranzintervall-Rechner · {datetime.now():%d.%m.%Y %H:%M}",
            styles['RFooter']
        ))

        doc.build(story)

    def _write_text_report(self, path, data, result_text, stats_text,
                           include_raw):
        """Fallback: Report als Textdatei (wenn reportlab fehlt)."""
        path = Path(path).with_suffix('.txt')
        lines = [
            "=" * 60,
            "Toleranzintervall – Report",
            f"Erstellt: {datetime.now():%d.%m.%Y %H:%M}",
            "=" * 60,
            "",
            "ERGEBNIS UND ENTSCHEIDUNGSLOG",
            "-" * 40,
            result_text,
            "",
            "DESKRIPTIVE STATISTIK",
            "-" * 40,
            stats_text,
        ]
        if include_raw:
            lines += [
                "",
                "ROHDATEN (sortiert)",
                "-" * 40,
            ]
            sorted_data = np.sort(data)
            for i, v in enumerate(sorted_data, 1):
                lines.append(f"  {i:>4d}:  {v:.6g}")

        path.write_text('\n'.join(lines), encoding='utf-8')
        self.statusBar().showMessage(
            f"Report als Text gespeichert (reportlab nicht verfügbar): {path}",
            5000
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def open_documentation(self):
        """Öffnet das PDF-Paper im Standard-PDF-Viewer."""
        # Suche PDF: neben der App, im src-Ordner, oder im PyInstaller-Bundle
        candidates = [
            Path(__file__).parent / 'toleranzintervalle.pdf',
            Path(__file__).parent.parent / 'toleranzintervalle.pdf',
        ]
        # PyInstaller-Bundle
        if hasattr(sys, '_MEIPASS'):
            candidates.insert(0, Path(sys._MEIPASS) / 'toleranzintervalle.pdf')

        for pdf_path in candidates:
            if pdf_path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path.resolve())))
                return

        QMessageBox.warning(
            self, "Datei nicht gefunden",
            "Die Dokumentation (toleranzintervalle.pdf) wurde nicht gefunden.\n\n"
            "Bitte legen Sie die Datei neben die Anwendung."
        )

    def show_min_n_table(self):
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_min_n_table()
        sys.stdout = old_stdout

        msg = QMessageBox(self)
        msg.setWindowTitle("Mindeststichprobengrößen")
        msg.setText("Mindest-n für verteilungsfreie Toleranzintervalle:")
        msg.setDetailedText(buf.getvalue())
        msg.setFont(self.mono)
        msg.exec()

    def show_about(self):
        QMessageBox.about(
            self,
            "Über Toleranzintervall-Rechner",
            "<h3>Toleranzintervall-Rechner</h3>"
            "<p>Automatische Berechnung von Toleranzintervallen "
            "mit intelligenter Methodenwahl.</p>"
            "<p><b>Entscheidungslogik (parametrisch zuerst):</b></p>"
            "<ol>"
            "<li>Shapiro-Wilk OK → Normal (k-Faktor)</li>"
            "<li>Shapiro-Wilk auf log(x) OK → Lognormal</li>"
            "<li>Weibull-GoF OK → Weibull (parametr. Bootstrap)</li>"
            "<li>n ausreichend → Verteilungsfrei (Ordnungsstatistiken)</li>"
            "<li>Sonst → Normal als Fallback (mit Warnung)</li>"
            "</ol>"
            "<p><i>Frank Zimmermann – 2026</i></p>"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ToleranceIntervalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
