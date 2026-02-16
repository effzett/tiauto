#!/usr/bin/env python3
"""
Automatisierte Berechnung von Toleranzintervallen.

Entscheidungslogik (parametrisch zuerst):
1. Shapiro-Wilk auf Normalität → Normal-TI (k-Faktor)
2. Daten > 0 und Shapiro-Wilk auf log(x) → Lognormal-TI
3. Daten > 0 und Weibull-GoF (Monte-Carlo-KS) → Weibull-TI (parametr. Bootstrap)
4. n ausreichend für verteilungsfrei? → verteilungsfrei (Order Statistics)
5. Sonst: Warnung + Normal-TI als Fallback

Begründung: Das verteilungsfreie Verfahren kann bei n ≈ n_min nur die
Extremwerte verwenden und liefert die schwächste Aussage. Parametrische
Verfahren nutzen alle n Datenpunkte und sind daher informativer –
besonders bei kleinem n.

Autor: Frank / Claude
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class ToleranceResult:
    """Ergebnis einer Toleranzintervall-Berechnung."""
    method: str
    lower: Optional[float]
    upper: Optional[float]
    coverage: float          # Mindestanteil der Population (p)
    confidence: float        # Konfidenzniveau (1 - alpha)
    side: str                # 'two-sided', 'lower', 'upper'
    n: int
    details: dict

    def __repr__(self):
        if self.side == 'two-sided':
            interval = f'[{self.lower:.6g}, {self.upper:.6g}]'
        elif self.side == 'lower':
            interval = f'[{self.lower:.6g}, ∞)'
        else:
            interval = f'(-∞, {self.upper:.6g}]'
        return (
            f"Toleranzintervall ({self.method})\n"
            f"  Intervall:   {interval}\n"
            f"  Abdeckung:   {self.coverage*100:.1f}%\n"
            f"  Konfidenz:   {self.confidence*100:.1f}%\n"
            f"  n:           {self.n}\n"
            f"  Details:     {self.details}"
        )


# ─────────────────────────────────────────────────────────────
# 1. Verteilungsfreies Toleranzintervall
# ─────────────────────────────────────────────────────────────

def min_n_distribution_free(p: float, confidence: float, side: str = 'two-sided') -> int:
    """
    Minimales n für ein verteilungsfreies Toleranzintervall,
    das mindestens Anteil p der Population mit gegebener Konfidenz abdeckt.

    Für two-sided: verwendet x_(1) und x_(n) als Grenzen.
    Die Konfidenz, dass [x_(1), x_(n)] mindestens p abdeckt, ist:
        1 - n*p^(n-1) + (n-1)*p^n  (für r=1, s=n)

    Für one-sided: P(X_(n) >= x_p) = 1 - p^n >= confidence
        => n >= log(1-confidence) / log(p)
    """
    if side in ('lower', 'upper'):
        # Einseitig: n >= log(alpha) / log(p)
        alpha = 1 - confidence
        n = int(np.ceil(np.log(alpha) / np.log(p)))
        return max(n, 2)
    else:
        # Zweiseitig mit r=1, s=n (äußerste Ordnungsstatistiken)
        # Konfidenz = 1 - n*p^(n-1) + (n-1)*p^n
        for n in range(2, 10000):
            conf = 1 - n * p**(n-1) + (n-1) * p**n
            if conf >= confidence:
                return n
        return 10000  # Sollte nicht erreicht werden


def distribution_free_ti(data: np.ndarray, p: float = 0.95,
                         confidence: float = 0.95,
                         side: str = 'two-sided') -> ToleranceResult:
    """
    Verteilungsfreies Toleranzintervall basierend auf Ordnungsstatistiken.

    Zweiseitig: Wählt optimale innere Ordnungsstatistiken (l, u) nach
    Meeker/Hahn (2017, Kap. 5), sodass das Intervall [x_(l), x_(u)] die
    geforderte Konfidenz erreicht und dabei möglichst eng ist.

    Konfidenzformel (Wilks): CPTI(n, l, u, p) = pbinom(u - l - 1; n, p)

    Einseitig: Wählt die innerste Ordnungsstatistik, die die geforderte
    Konfidenz noch erreicht.
    """
    n = len(data)
    sorted_data = np.sort(data)

    if side == 'two-sided':
        # ── Finde minimales d = u - l, sodass stats.binom.cdf(d-1, n, p) >= confidence
        d_min = None
        for d in range(1, n):
            cpti = stats.binom.cdf(d - 1, n, p)
            if cpti >= confidence:
                d_min = d
                break

        if d_min is None:
            # Sollte nicht passieren wenn min_n korrekt geprüft wurde
            d_min = n - 1  # Fallback: x_(1) bis x_(n)

        # ── Für dieses d: finde l, das das engste Intervall liefert
        best_l, best_width = 1, np.inf
        for l in range(1, n - d_min + 1):
            u = l + d_min
            width = sorted_data[u - 1] - sorted_data[l - 1]  # 0-indexed
            if width < best_width:
                best_width = width
                best_l = l

        best_u = best_l + d_min
        actual_conf = stats.binom.cdf(d_min - 1, n, p)
        lower = sorted_data[best_l - 1]
        upper = sorted_data[best_u - 1]

        details = {
            'ordnungsstatistiken': f'x_({best_l}), x_({best_u})',
            'tatsächliche_konfidenz': f'{actual_conf:.4f}',
            'geforderte_konfidenz': f'{confidence:.4f}',
        }

    elif side == 'upper':
        # Oberes TI: (-∞, x_(u)]
        # P(F(x_(u)) >= p) = stats.binom.cdf(u-1, n, p)
        # Finde kleinstes u mit stats.binom.cdf(u-1, n, p) >= confidence
        best_u = n  # Fallback
        for u in range(1, n + 1):
            if stats.binom.cdf(u - 1, n, p) >= confidence:
                best_u = u
                break
        actual_conf = stats.binom.cdf(best_u - 1, n, p)
        lower = None
        upper = sorted_data[best_u - 1]

        details = {
            'ordnungsstatistik': f'x_({best_u})',
            'tatsächliche_konfidenz': f'{actual_conf:.4f}',
            'geforderte_konfidenz': f'{confidence:.4f}',
        }

    else:  # lower
        # Unteres TI: [x_(l), ∞)
        # P(1 - F(x_(l)) >= p) = 1 - stats.binom.cdf(l-1, n, 1-p)
        # Finde größtes l mit 1 - stats.binom.cdf(l-1, n, 1-p) >= confidence
        best_l = 1  # Fallback
        for l in range(n, 0, -1):
            if 1 - stats.binom.cdf(l - 1, n, 1 - p) >= confidence:
                best_l = l
                break
        actual_conf = 1 - stats.binom.cdf(best_l - 1, n, 1 - p)
        lower = sorted_data[best_l - 1]
        upper = None

        details = {
            'ordnungsstatistik': f'x_({best_l})',
            'tatsächliche_konfidenz': f'{actual_conf:.4f}',
            'geforderte_konfidenz': f'{confidence:.4f}',
        }

    return ToleranceResult(
        method='Verteilungsfrei (Ordnungsstatistiken)',
        lower=lower, upper=upper,
        coverage=p, confidence=actual_conf,
        side=side, n=n,
        details=details
    )


# ─────────────────────────────────────────────────────────────
# 2. Normalverteilungs-Toleranzintervall (k-Faktor)
# ─────────────────────────────────────────────────────────────

def k_factor(n: int, p: float, confidence: float,
             side: str = 'two-sided') -> float:
    """
    Berechnet den k-Faktor für ein Toleranzintervall unter Normalverteilung.

    Einseitig:  Exakte Lösung über nichtzentrale t-Verteilung.
                P(T_{n-1, delta=z_p*sqrt(n)} <= k*sqrt(n)) = confidence

    Zweiseitig: Approximation nach Howe (1969) / ISO 16269:
                Löse nach k, sodass die Abdeckung p mit Konfidenz (1-alpha) erreicht wird.
    """
    if side in ('lower', 'upper'):
        # Einseitig: exakt über nichtzentrale t-Verteilung
        z_p = stats.norm.ppf(p)
        delta = z_p * np.sqrt(n)
        # k * sqrt(n) = t_{n-1, 1-alpha, delta}
        k = stats.nct.ppf(confidence, df=n-1, nc=delta) / np.sqrt(n)
        return k
    else:
        # Zweiseitig: Approximation
        # Verwende die Wald-Wolfowitz-Approximation / Howe
        z_p = stats.norm.ppf((1 + p) / 2)
        chi2_val = stats.chi2.ppf(1 - confidence, df=n-1)  # unteres Quantil

        # Einfache Approximation (gut für n >= 10):
        # k = z_p * sqrt((n-1) * n) / sqrt(chi2_val * n)  ... nicht ganz
        # Besser: iterative Lösung
        # Verwende Howe's Approximation:
        # k = z_p * sqrt( (n-1) * (1 + 1/n) / chi2_val )
        k = z_p * np.sqrt((n - 1) * (1 + 1/n) / chi2_val)
        return k


def normal_ti(data: np.ndarray, p: float = 0.95,
              confidence: float = 0.95,
              side: str = 'two-sided') -> ToleranceResult:
    """Toleranzintervall unter Normalverteilungsannahme."""
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    k = k_factor(n, p, confidence, side)

    if side == 'two-sided':
        lower = x_bar - k * s
        upper = x_bar + k * s
    elif side == 'lower':
        lower = x_bar - k * s
        upper = None
    else:
        lower = None
        upper = x_bar + k * s

    # Shapiro-Wilk p-Wert für Dokumentation
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(data)
    else:
        sw_stat, sw_p = None, None

    return ToleranceResult(
        method='Normal (k-Faktor)',
        lower=lower, upper=upper,
        coverage=p, confidence=confidence,
        side=side, n=n,
        details={
            'k_faktor': f'{k:.4f}',
            'mittelwert': f'{x_bar:.6g}',
            'std_abw': f'{s:.6g}',
            'shapiro_wilk_p': f'{sw_p:.4f}' if sw_p is not None else 'n/a',
        }
    )


# ─────────────────────────────────────────────────────────────
# 3. Lognormal-Toleranzintervall
# ─────────────────────────────────────────────────────────────

def lognormal_ti(data: np.ndarray, p: float = 0.95,
                 confidence: float = 0.95,
                 side: str = 'two-sided') -> ToleranceResult:
    """
    Toleranzintervall unter Lognormalverteilungsannahme.
    Berechnet TI auf log-Skala und transformiert zurück.
    """
    if np.any(data <= 0):
        raise ValueError("Lognormal-TI erfordert strikt positive Daten.")

    log_data = np.log(data)
    n = len(log_data)
    mu_log = np.mean(log_data)
    s_log = np.std(log_data, ddof=1)
    k = k_factor(n, p, confidence, side)

    if side == 'two-sided':
        lower = np.exp(mu_log - k * s_log)
        upper = np.exp(mu_log + k * s_log)
    elif side == 'lower':
        lower = np.exp(mu_log - k * s_log)
        upper = None
    else:
        lower = None
        upper = np.exp(mu_log + k * s_log)

    # Shapiro-Wilk auf log-Daten
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(log_data)
    else:
        sw_stat, sw_p = None, None

    return ToleranceResult(
        method='Lognormal (k-Faktor auf log-Skala)',
        lower=lower, upper=upper,
        coverage=p, confidence=confidence,
        side=side, n=n,
        details={
            'k_faktor': f'{k:.4f}',
            'mu_log': f'{mu_log:.6g}',
            's_log': f'{s_log:.6g}',
            'shapiro_wilk_p_log': f'{sw_p:.4f}' if sw_p is not None else 'n/a',
        }
    )


# ─────────────────────────────────────────────────────────────
# 3b. Weibull-Toleranzintervall (parametrischer Bootstrap)
# ─────────────────────────────────────────────────────────────

def weibull_gof(data: np.ndarray, alpha: float = 0.05):
    """
    Weibull-Anpassungstest.

    Fittet Weibull (loc=0) per MLE und testet mit
    Anderson-Darling (via KS + Monte-Carlo-korrigiertem p-Wert).

    Returns: (shape_c, scale_lam, ad_stat, p_value)

    Da scipy.stats.anderson Weibull nicht direkt unterstützt,
    verwenden wir einen KS-Test mit Lilliefors-Korrektur via
    Monte-Carlo-Simulation (500 Replikationen).
    """
    n = len(data)
    if np.any(data <= 0):
        return None, None, None, 0.0  # Weibull braucht positive Daten

    # MLE-Fit mit loc=0 (Standard-Weibull)
    try:
        c, loc, scale = stats.weibull_min.fit(data, floc=0)
    except Exception:
        return None, None, None, 0.0

    if c <= 0 or scale <= 0:
        return None, None, None, 0.0

    # KS-Statistik gegen gefittete Verteilung
    ks_stat, _ = stats.kstest(data, 'weibull_min', args=(c, 0, scale))

    # Monte-Carlo p-Wert (Lilliefors-Korrektur):
    # Simuliere unter H0 und zähle wie oft KS >= beobachtet
    n_mc = 500
    count = 0
    rng = np.random.default_rng(seed=42)
    for _ in range(n_mc):
        sim = stats.weibull_min.rvs(c, loc=0, scale=scale, size=n,
                                     random_state=rng)
        try:
            c_sim, _, scale_sim = stats.weibull_min.fit(sim, floc=0)
            ks_sim, _ = stats.kstest(sim, 'weibull_min',
                                      args=(c_sim, 0, scale_sim))
            if ks_sim >= ks_stat:
                count += 1
        except Exception:
            count += 1  # Konservativ: Fehler zählt als "nicht abgelehnt"

    mc_p = count / n_mc
    return c, scale, ks_stat, mc_p


def weibull_ti(data: np.ndarray, p: float = 0.95,
               confidence: float = 0.95,
               side: str = 'two-sided',
               n_bootstrap: int = 2000) -> ToleranceResult:
    """
    Toleranzintervall unter Weibull-Annahme via parametrischem Bootstrap.

    Vorgehen:
    1. Fitte Weibull(c, λ) an Daten (MLE, loc=0)
    2. Generiere n_bootstrap Stichproben der Größe n aus Weibull(c, λ)
    3. Fitte jede Bootstrap-Stichprobe neu → c*, λ*
    4. Berechne die relevanten Quantile aus jeder Bootstrap-Verteilung
    5. Bestimme Konfidenzgrenzen aus der Bootstrap-Verteilung der Quantile

    Für Lebensdauerdaten das natürliche Modell.
    """
    n = len(data)
    if np.any(data <= 0):
        raise ValueError("Weibull-TI erfordert strikt positive Daten.")

    # MLE-Fit
    c, _, scale = stats.weibull_min.fit(data, floc=0)

    alpha_ci = 1 - confidence
    rng = np.random.default_rng(seed=123)

    if side == 'two-sided':
        q_lo_target = (1 - p) / 2      # z.B. 0.025 für p=0.95
        q_hi_target = (1 + p) / 2      # z.B. 0.975

        boot_lo = np.empty(n_bootstrap)
        boot_hi = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sim = stats.weibull_min.rvs(c, loc=0, scale=scale, size=n,
                                         random_state=rng)
            try:
                c_b, _, s_b = stats.weibull_min.fit(sim, floc=0)
                boot_lo[i] = stats.weibull_min.ppf(q_lo_target, c_b,
                                                     loc=0, scale=s_b)
                boot_hi[i] = stats.weibull_min.ppf(q_hi_target, c_b,
                                                     loc=0, scale=s_b)
            except Exception:
                boot_lo[i] = np.nan
                boot_hi[i] = np.nan

        boot_lo = boot_lo[~np.isnan(boot_lo)]
        boot_hi = boot_hi[~np.isnan(boot_hi)]

        # Konservative Grenzen: unteres Ende von lower, oberes Ende von upper
        lower = np.percentile(boot_lo, alpha_ci / 2 * 100)
        upper = np.percentile(boot_hi, (1 - alpha_ci / 2) * 100)

    elif side == 'upper':
        boot_q = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sim = stats.weibull_min.rvs(c, loc=0, scale=scale, size=n,
                                         random_state=rng)
            try:
                c_b, _, s_b = stats.weibull_min.fit(sim, floc=0)
                boot_q[i] = stats.weibull_min.ppf(p, c_b, loc=0, scale=s_b)
            except Exception:
                boot_q[i] = np.nan

        boot_q = boot_q[~np.isnan(boot_q)]
        lower = None
        upper = np.percentile(boot_q, confidence * 100)

    else:  # lower
        boot_q = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sim = stats.weibull_min.rvs(c, loc=0, scale=scale, size=n,
                                         random_state=rng)
            try:
                c_b, _, s_b = stats.weibull_min.fit(sim, floc=0)
                boot_q[i] = stats.weibull_min.ppf(1 - p, c_b,
                                                    loc=0, scale=s_b)
            except Exception:
                boot_q[i] = np.nan

        boot_q = boot_q[~np.isnan(boot_q)]
        lower = np.percentile(boot_q, (1 - confidence) * 100)
        upper = None

    # Punkt-Schätzungen der Quantile (aus Original-Fit)
    q_lo_point = stats.weibull_min.ppf((1-p)/2, c, loc=0, scale=scale)
    q_hi_point = stats.weibull_min.ppf((1+p)/2, c, loc=0, scale=scale)

    # GoF
    ks_stat, ks_p = stats.kstest(data, 'weibull_min', args=(c, 0, scale))

    return ToleranceResult(
        method='Weibull (parametrischer Bootstrap)',
        lower=lower, upper=upper,
        coverage=p, confidence=confidence,
        side=side, n=n,
        details={
            'shape_c': f'{c:.4f}',
            'scale_lambda': f'{scale:.6g}',
            'quantil_punktschätzung': f'[{q_lo_point:.6g}, {q_hi_point:.6g}]',
            'n_bootstrap': n_bootstrap,
            'ks_stat': f'{ks_stat:.4f}',
            'ks_p_raw': f'{ks_p:.4f}',
        }
    )


# ─────────────────────────────────────────────────────────────
# 4. Automatische Methoden-Auswahl
# ─────────────────────────────────────────────────────────────

def auto_tolerance_interval(
    data,
    p: float = 0.95,
    confidence: float = 0.95,
    side: str = 'two-sided',
    method: str = 'auto',
    alpha_shapiro: float = 0.05,
    verbose: bool = True,
) -> ToleranceResult:
    """
    Automatische Berechnung eines Toleranzintervalls.

    Entscheidungslogik (parametrisch zuerst, bei method='auto'):
    1. Shapiro-Wilk p >= alpha auf Rohdaten? → Normal-TI
    2. Daten > 0 und Shapiro-Wilk p >= alpha auf log(Daten)? → Lognormal-TI
    3. Daten > 0 und Weibull-GoF p >= alpha? → Weibull-TI (Bootstrap)
    4. n >= n_min für verteilungsfrei? → verteilungsfrei (Ordnungsstatistiken)
    5. Sonst → Normal-TI mit Warnung (Fallback)

    Begründung: Parametrische Verfahren nutzen alle n Datenpunkte und
    liefern engere Intervalle als das verteilungsfreie Verfahren, das
    bei n ≈ n_min nur die Extremwerte verwenden kann. Verteilungsfrei
    kommt erst zum Zug, wenn keine Verteilung passt UND genug Daten
    vorhanden sind, damit die Fensteroptimierung innere
    Ordnungsstatistiken wählen kann.

    Parameters
    ----------
    data : array-like
        Messwerte
    p : float
        Mindestabdeckung der Population (z.B. 0.95 = 95%)
    confidence : float
        Konfidenzniveau (z.B. 0.95 = 95%)
    side : str
        'two-sided', 'lower', oder 'upper'
    method : str
        'auto' (Standard), 'normal', 'lognormal', 'weibull',
        oder 'distribution_free' für erzwungene Methodenwahl
    alpha_shapiro : float
        Signifikanzniveau für Shapiro-Wilk-Test (Standard: 0.05)
    verbose : bool
        Ausgabe der Entscheidungsschritte

    Returns
    -------
    ToleranceResult
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]  # NaN entfernen
    n = len(data)

    if n < 2:
        raise ValueError(f"Mindestens 2 Messwerte nötig, habe {n}.")

    def log(msg):
        if verbose:
            print(f"  → {msg}")

    # ── Erzwungene Methodenwahl ──
    if method != 'auto':
        if verbose:
            print(f"\n{'='*60}")
            print(f"Toleranzintervall-Berechnung (Methode: {method})")
            print(f"  n={n}, p={p}, confidence={confidence}, side={side}")
            print(f"{'='*60}")

        if method == 'normal':
            log("Erzwungen: Normal (k-Faktor)")
            result = normal_ti(data, p, confidence, side)
        elif method == 'lognormal':
            if np.any(data <= 0):
                raise ValueError("Lognormal erfordert strikt positive Daten.")
            log("Erzwungen: Lognormal (k-Faktor auf log-Skala)")
            result = lognormal_ti(data, p, confidence, side)
        elif method == 'weibull':
            if np.any(data <= 0):
                raise ValueError("Weibull erfordert strikt positive Daten.")
            log("Erzwungen: Weibull (parametrischer Bootstrap)")
            result = weibull_ti(data, p, confidence, side)
        elif method == 'distribution_free':
            min_n = min_n_distribution_free(p, confidence, side)
            if n < min_n:
                raise ValueError(
                    f"Verteilungsfrei benötigt mindestens n={min_n}, "
                    f"vorhanden n={n}."
                )
            log("Erzwungen: Verteilungsfrei (Ordnungsstatistiken)")
            result = distribution_free_ti(data, p, confidence, side)
        else:
            raise ValueError(
                f"Unbekannte Methode '{method}'. "
                f"Erlaubt: auto, normal, lognormal, weibull, distribution_free"
            )

        if verbose:
            print(f"\n{result}")
        return result

    # ── Automatische Methodenwahl ──
    min_n = min_n_distribution_free(p, confidence, side)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Automatische Toleranzintervall-Berechnung")
        print(f"  n={n}, p={p}, confidence={confidence}, side={side}")
        print(f"{'='*60}")

    # ── Schritt 1: Normalverteilung? ──
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(data)
        log(f"Shapiro-Wilk auf Rohdaten: W={sw_stat:.4f}, p={sw_p:.4f}")

        if sw_p >= alpha_shapiro:
            log(f"✓ Normalverteilung nicht abgelehnt (p={sw_p:.4f} >= {alpha_shapiro})")
            result = normal_ti(data, p, confidence, side)
            if verbose:
                print(f"\n{result}")
            return result

        log(f"✗ Normalverteilung abgelehnt (p={sw_p:.4f} < {alpha_shapiro})")
    else:
        log(f"n < 3, Shapiro-Wilk nicht möglich")

    # ── Schritt 2: Lognormalverteilung? ──
    if np.all(data > 0) and n >= 3:
        sw_stat_log, sw_p_log = stats.shapiro(np.log(data))
        log(f"Shapiro-Wilk auf log(Daten): W={sw_stat_log:.4f}, p={sw_p_log:.4f}")

        if sw_p_log >= alpha_shapiro:
            log(f"✓ Lognormalverteilung nicht abgelehnt (p={sw_p_log:.4f} >= {alpha_shapiro})")
            result = lognormal_ti(data, p, confidence, side)
            if verbose:
                print(f"\n{result}")
            return result

        log(f"✗ Lognormalverteilung abgelehnt (p={sw_p_log:.4f} < {alpha_shapiro})")

    # ── Schritt 3: Weibull-Verteilung? ──
    if np.all(data > 0) and n >= 5:
        log(f"Teste Weibull-Anpassung (Monte-Carlo-KS-Test)...")
        wb_c, wb_scale, wb_ks, wb_p = weibull_gof(data, alpha_shapiro)

        if wb_c is not None:
            log(f"Weibull-Fit: shape={wb_c:.4f}, scale={wb_scale:.6g}, "
                f"KS={wb_ks:.4f}, MC-p={wb_p:.4f}")

            if wb_p >= alpha_shapiro:
                log(f"✓ Weibull-Verteilung nicht abgelehnt "
                    f"(MC-p={wb_p:.4f} >= {alpha_shapiro})")
                result = weibull_ti(data, p, confidence, side)
                if verbose:
                    print(f"\n{result}")
                return result

            log(f"✗ Weibull-Verteilung abgelehnt "
                f"(MC-p={wb_p:.4f} < {alpha_shapiro})")
        else:
            log(f"✗ Weibull-Fit fehlgeschlagen")

    # ── Schritt 4: Verteilungsfrei möglich? ──
    log(f"Verteilungsfrei benötigt min. n={min_n}, vorhanden n={n}")

    if n >= min_n:
        log(f"✓ Genug Daten für verteilungsfreies Verfahren")
        result = distribution_free_ti(data, p, confidence, side)
        if verbose:
            print(f"\n{result}")
        return result

    log(f"✗ Nicht genug Daten für verteilungsfreies Verfahren (n={n} < {min_n})")

    # ── Schritt 5: Fallback → Normal mit Warnung ──
    log(f"⚠ WARNUNG: Keine Verteilung passt gut. Verwende Normal-TI als Fallback.")
    log(f"  Ergebnis mit Vorsicht interpretieren! Ggf. mehr Daten sammeln.")

    result = normal_ti(data, p, confidence, side)
    result.method += ' [FALLBACK – Verteilung unklar!]'
    result.details['warnung'] = 'Weder Normal-, Lognormal- noch Weibull-Verteilung bestätigt'

    if verbose:
        print(f"\n{result}")
    return result


# ─────────────────────────────────────────────────────────────
# 5. Hilfsfunktion: Übersichtstabelle min_n
# ─────────────────────────────────────────────────────────────

def print_min_n_table():
    """Druckt eine Übersichtstabelle der Mindeststichprobengrößen."""
    print(f"\nMindeststichprobengrößen für verteilungsfreie Toleranzintervalle")
    print(f"{'p':>8} {'conf':>8} {'einseitig':>12} {'zweiseitig':>12}")
    print(f"{'-'*44}")
    for p in [0.90, 0.95, 0.99]:
        for conf in [0.90, 0.95, 0.99]:
            n1 = min_n_distribution_free(p, conf, 'lower')
            n2 = min_n_distribution_free(p, conf, 'two-sided')
            print(f"{p:>8.2f} {conf:>8.2f} {n1:>12d} {n2:>12d}")


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)

    print_min_n_table()

    # Demo 1: Genug Daten → verteilungsfrei
    print("\n\n" + "▶"*30 + " DEMO 1: Große Stichprobe (n=100)")
    data1 = np.random.normal(50, 5, size=100)
    auto_tolerance_interval(data1)

    # Demo 2: Wenig Daten, normalverteilt → Normal-TI
    print("\n\n" + "▶"*30 + " DEMO 2: Kleine Stichprobe, normalverteilt (n=20)")
    data2 = np.random.normal(100, 10, size=20)
    auto_tolerance_interval(data2)

    # Demo 3: Wenig Daten, lognormalverteilt → Lognormal-TI
    print("\n\n" + "▶"*30 + " DEMO 3: Kleine Stichprobe, lognormal (n=25)")
    data3 = np.random.lognormal(3, 0.5, size=25)
    auto_tolerance_interval(data3)

    # Demo 4: Wenig Daten, keine klare Verteilung → Fallback
    print("\n\n" + "▶"*30 + " DEMO 4: Kleine Stichprobe, bimodal (n=15)")
    data4 = np.concatenate([np.random.normal(10, 1, 8), np.random.normal(30, 1, 7)])
    auto_tolerance_interval(data4)

    # Demo 5: Weibull-verteilte Lebensdauerdaten (stark rechtsschief)
    print("\n\n" + "▶"*30 + " DEMO 5: Lebensdauer, Weibull (n=30)")
    # shape=0.8: abnehmende Ausfallrate (Frühausfälle / infant mortality)
    data5 = stats.weibull_min.rvs(0.8, loc=0, scale=500, size=30,
                                    random_state=np.random.RandomState(0))
    auto_tolerance_interval(data5)

    # Demo 6: Einseitig
    print("\n\n" + "▶"*30 + " DEMO 6: Einseitig, obere Grenze (n=30)")
    data6 = np.random.normal(0, 1, size=30)
    auto_tolerance_interval(data6, side='upper')

    # Demo 7: Eigene Daten eingeben
    print("\n\n" + "▶"*30 + " DEMO 7: Praxisbeispiel - Messwerte")
    messwerte = [4.2, 4.5, 4.1, 4.8, 4.3, 4.6, 4.4, 4.7, 4.2, 4.5,
                 4.3, 4.6, 4.4, 4.5, 4.3]
    auto_tolerance_interval(messwerte, p=0.95, confidence=0.95, side='two-sided')

    # Demo 8a: Meeker Daten: upper
    print("\n\n" + "▶"*30 + " DEMO 8a: Praxisbeispiel - Messwerte Meeker Buch 3.2, Ex. 4.4")
    messwerte = [50.3, 48.3, 49.6, 50.4, 51.9]
    auto_tolerance_interval(messwerte, p=0.90, confidence=0.95, side='upper')

    # Demo 8b: Meeker Daten: lower
    print("\n\n" + "▶"*30 + " DEMO 8b: Praxisbeispiel - Messwerte Meeker Buch 3.2, Ex. 4.6")
    messwerte = [50.3, 48.3, 49.6, 50.4, 51.9]
    auto_tolerance_interval(messwerte, p=0.90, confidence=0.95, side='lower')

    # Demo 8c: Meeker Daten: two sided
    print("\n\n" + "▶"*30 + " DEMO 8c: Praxisbeispiel - Messwerte Meeker Buch 3.2, Ex. 4.9")
    messwerte = [50.3, 48.3, 49.6, 50.4, 51.9]
    auto_tolerance_interval(messwerte, p=0.90, confidence=0.95, side='two-sided')

    # Demo 9: Meeker verteilungsfreie Daten: two sided
    print("\n\n" + "▶"*30 + " DEMO 9: Praxisbeispiel - Messwerte Meeker Buch 5.3, Ex. 5.9")
    messwerte = [
        1.49, 1.66, 2.05, 2.24, 2.29, 2.69, 2.77, 2.77, 3.10, 3.23,
        3.28, 3.29, 3.31, 3.36, 3.84, 4.04, 4.09, 4.13, 4.14, 4.16,
        4.57, 4.63, 4.83, 5.06, 5.17, 5.19, 5.89, 5.97, 6.28, 6.38,
        6.51, 6.53, 6.54, 6.55, 6.83, 7.08, 7.28, 7.53, 7.54, 7.68,
        7.81, 7.87, 7.94, 8.43, 8.70, 8.97, 8.98, 9.13, 9.14, 9.22,
        9.24, 9.30, 9.44, 9.69, 9.86, 9.99, 11.28, 11.37, 12.03, 12.32,
        12.93, 13.03, 13.09, 13.43, 13.58, 13.70, 14.17, 14.36, 14.96, 15.89,
        16.57, 16.60, 16.85, 17.18, 17.46, 17.74, 18.40, 18.78, 19.84, 20.45,
        20.89, 22.28, 22.48, 23.66, 24.33, 24.72, 25.46, 25.67, 25.77, 26.64,
        28.28, 28.28, 29.07, 29.16, 31.14, 31.83, 33.24, 37.32, 53.43, 58.11,
    ]
    auto_tolerance_interval(messwerte, p=0.90, confidence=0.95, side='two-sided')

    # Demo 10: Gleiche Meeker-Daten, aber erzwungen verteilungsfrei
    print("\n\n" + "▶"*30 + " DEMO 10: Meeker 5.9 – erzwungen verteilungsfrei")
    auto_tolerance_interval(messwerte, p=0.90, confidence=0.95,
                        side='two-sided', method='distribution_free')