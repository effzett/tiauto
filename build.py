#!/usr/bin/env python3
"""
Build-Skript: Erzeugt eine eigenständige Anwendung.
    macOS:   dist/Toleranzintervall-Rechner.app
    Windows: dist/Toleranzintervall-Rechner.exe

Icon-Konvertierung:
    Lege eine icon.png (mind. 512x512, idealerweise 1024x1024) ins
    Projektverzeichnis. Das Skript konvertiert automatisch:
        macOS   → icon.icns  (via sips + iconutil, kein Pillow nötig)
        Windows → icon.ico   (via Pillow)

Voraussetzungen:
    pip install -r requirements-dev.txt

Aufruf:
    python build.py
"""

import subprocess
import sys
import shutil
from pathlib import Path


def convert_icon_macos(png_path: Path) -> Path:
    """Konvertiert PNG → ICNS via macOS sips + iconutil."""
    iconset = Path('build/icon.iconset')
    iconset.mkdir(parents=True, exist_ok=True)

    # macOS erwartet diese Größen im iconset
    sizes = [16, 32, 64, 128, 256, 512]
    for size in sizes:
        # Standard-Auflösung
        out = iconset / f'icon_{size}x{size}.png'
        subprocess.run([
            'sips', '-z', str(size), str(size),
            str(png_path), '--out', str(out)
        ], capture_output=True)
        # Retina (@2x) – doppelte Pixel, halbe Punktgröße
        size2 = size * 2
        if size2 <= 1024:
            out2 = iconset / f'icon_{size}x{size}@2x.png'
            subprocess.run([
                'sips', '-z', str(size2), str(size2),
                str(png_path), '--out', str(out2)
            ], capture_output=True)

    # 512@2x = 1024x1024
    out_1024 = iconset / 'icon_512x512@2x.png'
    subprocess.run([
        'sips', '-z', '1024', '1024',
        str(png_path), '--out', str(out_1024)
    ], capture_output=True)

    icns_path = Path('build/icon.icns')
    subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(icns_path)],
                    check=True)
    print(f"  ✓ Icon konvertiert: {icns_path}")
    return icns_path


def convert_icon_windows(png_path: Path) -> Path:
    """Konvertiert PNG → ICO via Pillow."""
    try:
        from PIL import Image
    except ImportError:
        print("  ⚠ Pillow nicht installiert (pip install Pillow)")
        print("    → Baue ohne Icon")
        return None

    Path('build').mkdir(parents=True, exist_ok=True)
    ico_path = Path('build/icon.ico')
    img = Image.open(png_path)

    # ICO braucht mehrere Größen
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format='ICO',
             sizes=[s for s in sizes if s[0] <= img.width])
    print(f"  ✓ Icon konvertiert: {ico_path}")
    return ico_path


def convert_icon(png_path: Path) -> Path:
    """Plattform-abhängige Icon-Konvertierung."""
    if sys.platform == 'darwin':
        return convert_icon_macos(png_path)
    elif sys.platform == 'win32':
        return convert_icon_windows(png_path)
    else:
        # Linux: kein App-Icon nötig
        return None


def build():
    app_name = 'Toleranzintervall-Rechner'

    # Aufräumen
    for d in ['build', 'dist']:
        if Path(d).exists():
            shutil.rmtree(d)

    sep = ';' if sys.platform == 'win32' else ':'

    # ── Icon vorbereiten ──
    png_path = Path('icon.png')
    icon_path = None
    if png_path.exists():
        print("Icon gefunden, konvertiere...")
        icon_path = convert_icon(png_path)
    else:
        print("Kein icon.png gefunden, baue ohne Icon.")
        print(f"  Tipp: Lege eine icon.png (512x512+) ins Projektverzeichnis.")

    # ── PyInstaller Kommando ──
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', app_name,
        '--windowed',               # Kein Konsolenfenster, .app auf macOS
        '--noconfirm',
        '--clean',
        # Versteckte Imports
        '--hidden-import', 'scipy.special._cdflib',
        '--hidden-import', 'scipy.stats',
        '--hidden-import', 'numpy',
        # src-Paket mitliefern
        '--add-data', f'src{sep}src',
        # Dokumentation (PDF) mitliefern
        '--add-data', f'toleranzintervalle.pdf{sep}.',
    ]

    # Plattform-spezifische Optionen
    if sys.platform == 'darwin':
        # macOS: .app Bundle
        cmd += [
            '--osx-bundle-identifier', 'de.zenmeister.toleranzintervall',
        ]
        if icon_path:
            cmd += ['--icon', str(icon_path)]
    elif sys.platform == 'win32':
        # Windows: einzelne .exe
        cmd += ['--onefile']
        if icon_path:
            cmd += ['--icon', str(icon_path)]
    else:
        # Linux
        cmd += ['--onefile']
        if icon_path:
            cmd += ['--icon', str(icon_path)]

    cmd.append('main.py')

    print(f"\nStarte Build für {sys.platform}...")
    print(f"  Kommando: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✓ Build erfolgreich!")
        if sys.platform == 'darwin':
            print(f"  Ausgabe: dist/{app_name}.app")
            print(f"  Starten: open dist/{app_name}.app")
            print()
            print(f"  Tipp: In Applications ziehen für permanente Installation")
        elif sys.platform == 'win32':
            print(f"  Ausgabe: dist/{app_name}.exe")
        else:
            print(f"  Ausgabe: dist/{app_name}")
        print("=" * 60)
    else:
        print("\n✗ Build fehlgeschlagen.")
        sys.exit(1)


if __name__ == '__main__':
    build()
