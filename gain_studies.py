"""
Gain estimate based on the Gen-II LAPPD paper (Shin et al., Sec. 3.3, Eq. 3.5):

    G * e = integral of  V_out(t) / 50 ohm  dt

For every event with a detectable pulse in the window, the baseline-subtracted
voltage is integrated over the pulse's own positive lobe:

    start = zero crossing just BEFORE the peak
    end   = first zero crossing AFTER the peak

The integral is divided by 50 ohm to get charge, and by the elementary charge
to get the gain in electrons. Because the integral stops at the first zero
crossing after the peak, the undershoot is never included, and because it
starts at the pulse's own zero crossing, baseline noise before the pulse is
not included either. The window is only used to find the pulse (highest
sample inside it, which must exceed the threshold).

Events with no pulse above threshold are left out. The reported gain is the
mean over the remaining pulses.

Threshold = 5 x RMS noise, from the pedestal file made by
    python waveformana.py --pedestal --lappd NAME laser_blocked.xml

Usage:
    python gain_studies.py run.xml --lappd 189 --channel CHN1 --window 30 40

Output (nothing else is written):
    gain/<xml name>_<channel>/gain_histogram.png
    gain/<xml name>_<channel>/gain_summary.txt
    gain/<xml name>_<channel>/gain_waveforms.pdf   (first 300 pulses, the
                                                    integrated part in red)
"""

import re
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ELEMENTARY_CHARGE_C = 1.602176634e-19
IMPEDANCE_OHM = 50.0
N_SIGMA = 5
N_PDF_WAVEFORMS = 300
PDF_PER_PAGE = 2

_SERIAL_RE = re.compile(r'<Serial>(\d+)</Serial>')
_DATA_RE = re.compile(r'<Data>([-\d.]+),([-\d.]+)</Data>')


def iter_events(filepath, channel):
    """Stream events one at a time from a DRSOSC XML file (one channel)."""
    tag_open, tag_close = f"<{channel}>", f"</{channel}>"
    buffer = []
    in_event = False
    with open(filepath, "r") as f:
        for line in f:
            if "<Event>" in line:
                in_event = True
                buffer = [line]
            elif in_event:
                buffer.append(line)
                if "</Event>" in line:
                    block = "".join(buffer)
                    buffer = []
                    in_event = False

                    serial_m = _SERIAL_RE.search(block)
                    if serial_m is None or tag_open not in block:
                        continue
                    section = block[block.find(tag_open):block.find(tag_close)]
                    pairs = _DATA_RE.findall(section)
                    if not pairs:
                        continue
                    arr = np.array(pairs, dtype=float)
                    yield {"serial": int(serial_m.group(1)), "t": arr[:, 0], "v": arr[:, 1]}


def load_pedestal(lappd_name):
    path = f"pedestal_lappd_{lappd_name}.txt"
    try:
        with open(path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"No {path} found. Make it first with:\n"
              f"  python waveformana.py --pedestal --lappd {lappd_name} laser_blocked.xml")
        sys.exit(1)
    return float(lines[0]), float(lines[1])


def _integrate(v, t):
    return np.trapezoid(v, t) if hasattr(np, "trapezoid") else np.trapz(v, t)


def pulse_gain(t, v_sub, window_ns, threshold):
    """
    Returns None if there is no pulse above threshold in the window.
    Otherwise a dict with:
        gain           gain in electrons
        found_crossing False if the signal never returned to zero after the peak
        seg_t, seg_v   the integrated part of the waveform (for plotting),
                       with its end points sitting exactly on zero

    v_sub is the baseline-subtracted voltage (mV), NOT clipped.
    """
    win_idx = np.where((t >= window_ns[0]) & (t <= window_ns[1]))[0]
    if len(win_idx) == 0:
        return None

    peak_idx = win_idx[np.argmax(v_sub[win_idx])]
    if v_sub[peak_idx] <= threshold:
        return None

    # --- start: last sample at or below zero before the peak ---
    before = np.where(v_sub[:peak_idx] <= 0)[0]
    if len(before) == 0:
        first_idx = 0              # positive all the way back to the trace start
        start_tri = 0.0
        head_t, head_v = [], []
    else:
        j = before[-1]
        first_idx = j + 1
        v0, v1 = v_sub[j], v_sub[j + 1]                     # v0 <= 0 < v1
        t_start = t[j] + (t[j + 1] - t[j]) * (-v0) / (v1 - v0)
        start_tri = 0.5 * v1 * (t[j + 1] - t_start)
        head_t, head_v = [t_start], [0.0]

    # --- end: first sample at or below zero after the peak ---
    after = np.where(v_sub[peak_idx + 1:] <= 0)[0]
    if len(after) == 0:
        last_idx = len(t) - 1      # never returned to zero: integrate to trace end
        end_tri = 0.0
        tail_t, tail_v = [], []
        found_crossing = False
    else:
        cross_idx = peak_idx + 1 + after[0]
        last_idx = cross_idx - 1
        v0, v1 = v_sub[last_idx], v_sub[cross_idx]          # v0 > 0 >= v1
        t_zc = t[last_idx] + (t[cross_idx] - t[last_idx]) * v0 / (v0 - v1)
        end_tri = 0.5 * v0 * (t_zc - t[last_idx])
        tail_t, tail_v = [t_zc], [0.0]
        found_crossing = True

    body = slice(first_idx, last_idx + 1)
    integral = _integrate(v_sub[body], t[body]) + start_tri + end_tri

    charge_pC = integral / IMPEDANCE_OHM          # (mV * ns) / ohm = pC
    gain = charge_pC * 1e-12 / ELEMENTARY_CHARGE_C

    return {
        "gain": float(gain),
        "found_crossing": found_crossing,
        "seg_t": np.array(head_t + list(t[body]) + tail_t),
        "seg_v": np.array(head_v + list(v_sub[body]) + tail_v),
    }


def save_waveform_pdf(events, filename, window_ns):
    """
    PDF of pulses, PDF_PER_PAGE per page. The full baseline-subtracted
    waveform is drawn in blue (nothing clipped); the part that was integrated
    is drawn on top in red. Dotted lines mark the search window.
    """
    xlim = (window_ns[0] - 9, window_ns[1] + 40)
    n = len(events)
    n_pages = int(np.ceil(n / PDF_PER_PAGE))
    with PdfPages(filename) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(PDF_PER_PAGE, 1, figsize=(8, 4 * PDF_PER_PAGE))
            axes = np.atleast_1d(axes)
            for i, ax in enumerate(axes):
                k = page * PDF_PER_PAGE + i
                if k >= n:
                    ax.set_visible(False)
                    continue
                e = events[k]
                m = (e["t"] >= xlim[0]) & (e["t"] <= xlim[1])
                ax.plot(e["t"][m], e["v_sub"][m], linewidth=0.7, color="steelblue")
                ax.plot(e["seg_t"], e["seg_v"], linewidth=1.3, color="crimson")
                ax.axhline(0, color="gray", linewidth=0.4)
                ax.axvline(window_ns[0], color="orange", linewidth=0.9, linestyle=":")
                ax.axvline(window_ns[1], color="orange", linewidth=0.9, linestyle=":")
                ax.set_xlim(*xlim)
                ax.tick_params(labelsize=6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {n} waveforms across {n_pages} page(s) to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gain estimate (Gen-II LAPPD paper, Eq. 3.5)")
    parser.add_argument("filepath", help="path to the DRSOSC XML file")
    parser.add_argument("--lappd", type=str, required=True,
                        help="LAPPD name, e.g. 189 (uses pedestal_lappd_189.txt)")
    parser.add_argument("--channel", type=str, default="CHN1",
                        help="DRS4 channel to analyze (default: CHN1)")
    parser.add_argument("--window", type=float, nargs=2, default=[30, 40],
                        metavar=("START_NS", "END_NS"),
                        help="pulse window in ns, used to find the pulse (default: 30 40)")
    args = parser.parse_args()

    window_ns = tuple(args.window)
    baseline_mean, rms_noise = load_pedestal(args.lappd)
    threshold = N_SIGMA * rms_noise

    tag = os.path.splitext(os.path.basename(args.filepath))[0]
    outdir = os.path.join("gain", f"{tag}_{args.channel}")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {outdir}/")
    print(f"Channel {args.channel}, window {window_ns} ns, "
          f"baseline={baseline_mean:.3f} mV, RMS={rms_noise:.3f} mV, threshold={threshold:.3f} mV")

    gains = []
    pdf_events = []
    n_total = 0
    n_no_crossing = 0
    for ev in iter_events(args.filepath, args.channel):
        n_total += 1
        v_sub = ev["v"] - baseline_mean
        result = pulse_gain(ev["t"], v_sub, window_ns, threshold)
        if result is None:
            continue
        gains.append(result["gain"])
        if not result["found_crossing"]:
            n_no_crossing += 1
        if len(pdf_events) < N_PDF_WAVEFORMS:
            pdf_events.append({"t": ev["t"], "v_sub": v_sub,
                               "seg_t": result["seg_t"], "seg_v": result["seg_v"]})

    gains = np.array(gains)
    n_pulses = len(gains)

    lines = [
        f"File:                      {args.filepath}",
        f"Channel:                   {args.channel}",
        f"Window (to find pulse):    {window_ns[0]} to {window_ns[1]} ns",
        f"Integral:                  zero crossing before the peak -> first zero crossing after it",
        f"Pedestal baseline:         {baseline_mean:.4f} mV",
        f"Pedestal RMS noise:        {rms_noise:.4f} mV",
        f"Threshold (5 sigma):       {threshold:.4f} mV",
        f"Events analyzed:           {n_total}",
        f"Pulses used:               {n_pulses}",
    ]
    if n_pulses > 0:
        std = gains.std(ddof=1) if n_pulses > 1 else 0.0
        lines += [
            f"Mean gain:                 {gains.mean():.4e} electrons",
            f"Std of gain:               {std:.4e} electrons",
            f"Pulses with no zero crossing after the peak (integrated to end of trace): {n_no_crossing}",
        ]
    else:
        lines.append("No pulses above threshold in this window -- no gain computed.")

    summary = "\n".join(lines)
    print("\n" + summary)
    summary_path = os.path.join(outdir, "gain_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved {summary_path}")

    if n_pulses > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(gains, bins=60, color="steelblue", edgecolor="none")
        ax.set_xlabel("Gain (electrons)")
        ax.set_ylabel("Number of pulses")
        ax.set_yscale("log")
        fig.tight_layout()
        hist_path = os.path.join(outdir, "gain_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Saved {hist_path}")

        if len(pdf_events) < N_PDF_WAVEFORMS:
            print(f"Only {len(pdf_events)} pulses available (fewer than {N_PDF_WAVEFORMS}) -- plotting all of them.")
        save_waveform_pdf(pdf_events, os.path.join(outdir, "gain_waveforms.pdf"), window_ns)