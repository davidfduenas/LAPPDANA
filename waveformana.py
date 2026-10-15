"""
DRS4 XML event parser and analyzer — memory-safe streaming version.

Reads the DRSOSC XML file ONE EVENT AT A TIME (iter_drs4_events), so 100,000+
event files don't crash.

UNDERSHOOT: the LAPPD pulse is positive; the negative undershoot after it is
an electronics artifact, not signal. Every calculation AND every plot works on
positive_signal() = baseline-subtracted voltage clipped at 0. The only thing
NOT clipped is the pedestal calibration, which needs both signs of the noise
to get a correct baseline and RMS.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_SERIAL_RE = re.compile(r'<Serial>(\d+)</Serial>')
_TRIGCELL_RE = re.compile(r'<Trigger_Cell>(-?\d+)</Trigger_Cell>')
_DATA_RE = re.compile(r'<Data>([-\d.]+),([-\d.]+)</Data>')


def positive_signal(v, baseline):
    """Baseline-subtract and remove the undershoot (everything below 0). Used
    only by the CALCULATIONS (hit/amplitude/charge), never by plotting."""
    return np.clip(v - baseline, 0, None)


def iter_drs4_events(filepath, channel="CHN1"):
    """Stream events one at a time from a DRSOSC XML file."""
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
                    if serial_m is None:
                        continue
                    trigcell_m = _TRIGCELL_RE.search(block)
                    trigger_cell = int(trigcell_m.group(1)) if trigcell_m else -1

                    if tag_open not in block:
                        continue
                    section = block[block.find(tag_open):block.find(tag_close)]
                    pairs = _DATA_RE.findall(section)
                    if not pairs:
                        continue
                    arr = np.array(pairs, dtype=float)

                    yield {
                        "serial": int(serial_m.group(1)),
                        "trigger_cell": trigger_cell,
                        "t": arr[:, 0],
                        "v": arr[:, 1],
                    }


def plot_event(event, ax=None, label=None, baseline_mean=0.0, window_ns=None):
    """Plot a single event's full raw waveform (nothing clipped or removed)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    v_raw = event["v"] - baseline_mean
    ax.plot(event["t"], v_raw, linewidth=0.9, label=label)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("voltage (mV)")
    ax.axhline(0, color="gray", linewidth=0.5)
    return ax


def compute_pedestal_calibration(filepath, channel="CHN1"):
    """
    Baseline and RMS noise from a pedestal run (laser blocked). NOT clipped —
    needs both signs of the noise. Running sums keep memory flat.
    """
    total_sum = 0.0
    total_sumsq = 0.0
    total_n = 0
    n_events = 0
    for ev in iter_drs4_events(filepath, channel=channel):
        v = ev["v"]
        total_sum += v.sum()
        total_sumsq += (v ** 2).sum()
        total_n += len(v)
        n_events += 1

    baseline_mean = total_sum / total_n
    variance = total_sumsq / total_n - baseline_mean ** 2
    rms_noise = np.sqrt(max(variance, 0.0))
    return baseline_mean, rms_noise, n_events


def pedestal_filepath(lappd_name):
    return f"pedestal_lappd_{lappd_name}.txt"


def save_pedestal_calibration(baseline_mean, rms_noise, lappd_name):
    filepath = pedestal_filepath(lappd_name)
    with open(filepath, "w") as f:
        f.write(f"{baseline_mean}\n{rms_noise}\n")
    print(f"Saved pedestal calibration to {filepath}")


def load_pedestal_calibration(lappd_name):
    filepath = pedestal_filepath(lappd_name)
    with open(filepath, "r") as f:
        lines = f.readlines()
    return float(lines[0]), float(lines[1])


def compute_baseline_and_threshold_fallback(filepath, channel="CHN1",
                                             baseline_window_ns=(150, 195), n_sigma=5):
    """
    FALLBACK ONLY (no pedestal file). Baseline/RMS from a fixed quiet window.
    NOT clipped. The window must not overlap the real pulse.
    """
    total_sum = 0.0
    total_sumsq = 0.0
    total_n = 0
    for ev in iter_drs4_events(filepath, channel=channel):
        mask = (ev["t"] >= baseline_window_ns[0]) & (ev["t"] <= baseline_window_ns[1])
        v = ev["v"][mask]
        if len(v) == 0:
            continue
        total_sum += v.sum()
        total_sumsq += (v ** 2).sum()
        total_n += len(v)

    baseline_mean = total_sum / total_n
    variance = total_sumsq / total_n - baseline_mean ** 2
    rms_noise = np.sqrt(max(variance, 0.0))
    threshold = n_sigma * rms_noise
    return baseline_mean, rms_noise, threshold


def save_waveform_grid_pdf(event_list, filename, ncols=5, nrows=4, xlim=None, tag="",
                            baseline_mean=0.0, window_ns=None):
    """
    Multi-page PDF of waveforms. Full raw signal is always drawn (nothing
    clipped/removed). No titles on the page or on individual waveforms.
    """
    per_page = ncols * nrows
    n_pages = int(np.ceil(len(event_list) / per_page)) if len(event_list) > 0 else 0
    if n_pages == 0:
        print(f"No events to plot for {filename} — skipping.")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    axes = np.array(axes).reshape(-1)
    lines = []
    for ax in axes:
        line, = ax.plot([], [], linewidth=0.7, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.4)
        if window_ns is not None:
            ax.axvline(window_ns[0], color="orange", linewidth=0.9, linestyle=":")
            ax.axvline(window_ns[1], color="orange", linewidth=0.9, linestyle=":")
        ax.tick_params(labelsize=6)
        if xlim is not None:
            ax.set_xlim(*xlim)
        lines.append(line)
    fig.tight_layout()

    with PdfPages(filename) as pdf:
        for page in range(n_pages):
            chunk = event_list[page * per_page: (page + 1) * per_page]

            for i, ax in enumerate(axes):
                if i < len(chunk):
                    ev = chunk[i]
                    v_raw = ev["v"] - baseline_mean
                    if xlim is not None:
                        mask = (ev["t"] >= xlim[0]) & (ev["t"] <= xlim[1])
                        lines[i].set_data(ev["t"][mask], v_raw[mask])
                        ax.relim()
                        ax.autoscale_view(scalex=False)
                        ax.set_xlim(*xlim)
                    else:
                        lines[i].set_data(ev["t"], v_raw)
                        ax.relim()
                        ax.autoscale_view()
                    ax.set_visible(True)
                else:
                    ax.set_visible(False)

            pdf.savefig(fig)

    plt.close(fig)
    print(f"Saved {len(event_list)} waveforms across {n_pages} page(s) to {filename}")


def plot_amplitude_histogram(amplitudes, filename, bins=60, tag="", hit_fraction=None, purity=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_edges = np.histogram_bin_edges(amplitudes, bins=bins)
    ax.hist(amplitudes, bins=bin_edges, color="steelblue", edgecolor="none")
    ax.set_xlabel("Peak amplitude (mV)")
    ax.set_ylabel("Number of events")
    ax.set_yscale("log")
    if hit_fraction is not None and purity is not None:
        text = (f"Events with a pulse: {hit_fraction:.2f}%\n"
                f"Purity (fraction of hits that are true 1 PE): {purity:.2f}%")
        ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8))
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")
    return bin_edges


def extract_integrated_charge(ev, baseline_mean, window_ns, impedance_ohm=50):
    """
    Integral of the positive signal (undershoot removed) within the window,
    divided by impedance. (mV*ns)/ohm = pC.
    """
    mask = (ev["t"] >= window_ns[0]) & (ev["t"] <= window_ns[1])
    t = ev["t"][mask]
    if len(t) < 2:
        return 0.0
    v_pos = positive_signal(ev["v"][mask], baseline_mean)
    integral_mV_ns = np.trapezoid(v_pos, t) if hasattr(np, "trapezoid") else np.trapz(v_pos, t)
    return float(integral_mV_ns / impedance_ohm)


def plot_charge_histogram(charges, filename, bins=60, tag=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_edges = np.histogram_bin_edges(charges, bins=bins)
    ax.hist(charges, bins=bin_edges, color="steelblue", edgecolor="none")
    ax.set_xlabel("Integrated charge (pC)")
    ax.set_ylabel("Number of events")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")
    return bin_edges


def debug_single_event(filepath, serial, baseline_mean, rms_noise, threshold, window_ns, channel="CHN1"):
    """Print and plot the classification decision for one event serial."""
    target = None
    for ev in iter_drs4_events(filepath, channel=channel):
        if ev["serial"] == serial:
            target = ev
            break
    if target is None:
        print(f"Event {serial} not found in {filepath}")
        return

    mask = (target["t"] >= window_ns[0]) & (target["t"] <= window_ns[1])
    v_win = positive_signal(target["v"][mask], baseline_mean)
    t_win = target["t"][mask]

    if len(v_win) == 0:
        amp, is_hit, peak_t = 0.0, False, None
    else:
        idx = int(np.argmax(v_win))
        amp = float(v_win[idx])
        peak_t = float(t_win[idx])
        is_hit = bool(np.any(v_win > threshold))

    print(f"--- Debug: event {serial} ---")
    print(f"Baseline mean:      {baseline_mean:.4f} mV")
    print(f"RMS noise:          {rms_noise:.4f} mV")
    print(f"Threshold (5 sig):  {threshold:.4f} mV")
    print(f"Pulse window:       {window_ns} ns")
    print(f"Max value in window (baseline-subtracted): {amp:.4f} mV, at t={peak_t} ns")
    print(f"Verdict: {'HIT' if is_hit else 'EMPTY'}  ({amp:.4f} {'>' if is_hit else '<='} {threshold:.4f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    v_raw = target["v"] - baseline_mean
    ax.plot(target["t"], v_raw, linewidth=0.9, color="steelblue", label="raw signal")
    ax.axhline(0, color="gray", linewidth=0.6, label="baseline")
    ax.axhline(threshold, color="red", linewidth=0.8, linestyle="--", label="threshold")
    ax.axvspan(window_ns[0], window_ns[1], color="yellow", alpha=0.15, label="pulse window")
    if peak_t is not None:
        ax.plot(peak_t, amp, "o", color="crimson", markersize=8,
                 label=f"detected peak ({amp:.2f} mV)")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("baseline-subtracted voltage (mV)")
    ax.set_xlim(window_ns[0] - 9, window_ns[1] + 40)
    ax.legend(fontsize=8)
    fig.tight_layout()
    tag = os.path.splitext(os.path.basename(filepath))[0]
    os.makedirs(tag, exist_ok=True)
    outpath = os.path.join(tag, f"debug_event_{serial}.png")
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


def find_afterpulse(ev, baseline_mean, threshold, afterpulse_window_ns, main_peak_t):
    """
    Search for a secondary (afterpulse) threshold crossing after the main
    pulse, within afterpulse_window_ns. Returns (found, delay_ns) where
    delay_ns is the time from the main pulse's peak to the afterpulse's
    peak, or (False, None) if no secondary crossing is found.
    """
    mask = (ev["t"] >= afterpulse_window_ns[0]) & (ev["t"] <= afterpulse_window_ns[1])
    v_win = positive_signal(ev["v"][mask], baseline_mean)
    t_win = ev["t"][mask]
    if len(v_win) == 0 or not np.any(v_win > threshold):
        return False, None
    idx = int(np.argmax(v_win))
    ap_peak_t = float(t_win[idx])
    return True, ap_peak_t - main_peak_t


def plot_afterpulse_histogram(delays, filename, bins=60, tag=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delays, bins=bins, color="steelblue", edgecolor="none")
    ax.set_xlabel("Delay from main pulse (ns)")
    ax.set_ylabel("Number of afterpulses")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}  (n_afterpulses={len(delays)})")


def purity_from_nbar(nbar):
    """Fraction of non-empty (hit) events that are true single photoelectrons."""
    if nbar <= 0:
        return float("nan")
    p0 = np.exp(-nbar)
    p1 = nbar * p0
    return p1 / (1 - p0) if p0 < 1 else float("nan")


def nbar_for_purity(target_purity, lo=1e-4, hi=10.0, tol=1e-8, max_iter=200):
    """Solve purity_from_nbar(Nbar) = target_purity for Nbar by bisection."""
    f_lo = purity_from_nbar(lo) - target_purity
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = purity_from_nbar(mid) - target_purity
        if abs(f_mid) < tol:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def print_purity_summary(nbar, n_total=None, n_hits=None):
    """
    Print the occupancy/purity picture for a measured Nbar: P(0), hit
    fraction, purity at this Nbar, and a reverse table of Nbar needed for
    standard purity targets (reference only -- no filter recommendation).
    """
    p0 = np.exp(-nbar) if np.isfinite(nbar) else 0.0
    hit_fraction = (1 - p0) * 100

    print(f"\n--- Purity / occupancy summary ---")
    print(f"  Nbar:                          {nbar:.4f}")
    print(f"  P(0) (empty-event fraction):   {p0:.4f}")
    print(f"  Events with a detectable pulse: {hit_fraction:.2f}%"
          + (f"  ({n_hits} of {n_total})" if n_total is not None else ""))

    current_purity = purity_from_nbar(nbar)
    print(f"  Purity at this Nbar (fraction of hits that are true 1 PE): {current_purity * 100:.2f}%")

    targets = [0.90, 0.80, 0.75]
    print(f"\n  Reverse table -- Nbar needed for a given purity target:")
    for target in targets:
        nbar_target = nbar_for_purity(target)
        print(f"    purity={target*100:.0f}%  ->  Nbar={nbar_target:.4f}")
    print(f"-----------------------------------")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="DRS4 waveform analysis (streaming, memory-safe)")
    parser.add_argument("filepath", help="path to the DRSOSC XML file")
    parser.add_argument("--preview", type=int, default=None,
                         help="just sanity-check N waveforms instead of the full analysis")
    parser.add_argument("--pedestal", action="store_true",
                         help="treat this file as a pedestal run (laser blocked): compute and "
                              "save baseline/noise calibration for this LAPPD")
    parser.add_argument("--lappd", type=str, default=None,
                         help="LAPPD name/serial, e.g. 189 (pedestal_lappd_189.txt)")
    parser.add_argument("--debug-event", type=int, default=None,
                         help="show the classification decision for one event serial")
    parser.add_argument("--max-events", type=int, default=None,
                         help="only process the first N events (full analysis mode only)")
    parser.add_argument("--channel", type=str, default="CHN1",
                         help="DRS4 channel to analyze, e.g. CHN1 or CHN2 (default: CHN1)")
    parser.add_argument("--window", type=float, nargs=2, default=[30, 40], metavar=("START_NS", "END_NS"),
                         help="main pulse search window in ns, e.g. --window 30 40 (default: 30 40)")
    args = parser.parse_args()

    TAG = os.path.splitext(os.path.basename(args.filepath))[0]
    OUTDIR = TAG
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Output directory: {OUTDIR}/")
    print(f"Input file tag: {TAG}")
    print(f"Channel: {args.channel}")

    PULSE_WINDOW_NS = tuple(args.window)
    # Search window for afterpulses (ion feedback): starts a couple ns after
    # the main pulse window ends, so the falling edge/tail of the main pulse
    # isn't mistaken for a separate afterpulse, and runs to the end of the
    # full DRS4 acquisition window (200 ns).
    AFTERPULSE_WINDOW_NS = (PULSE_WINDOW_NS[1] + 2, 200)
    MIN_PLOTS = 1000

    if args.debug_event is not None:
        if not args.lappd:
            print("--debug-event requires --lappd NAME so it can load the pedestal calibration")
            sys.exit(1)
        try:
            baseline_mean, rms_noise = load_pedestal_calibration(args.lappd)
        except FileNotFoundError:
            print(f"No {pedestal_filepath(args.lappd)} found.")
            sys.exit(1)
        threshold = 5 * rms_noise
        debug_single_event(args.filepath, args.debug_event, baseline_mean, rms_noise,
                            threshold, PULSE_WINDOW_NS, channel=args.channel)
        sys.exit(0)

    if args.pedestal:
        if not args.lappd:
            print("Error: --pedestal requires --lappd NAME (e.g. --lappd 189)")
            sys.exit(1)
        baseline_mean, rms_noise, n_events = compute_pedestal_calibration(args.filepath, channel=args.channel)
        print(f"Parsed {n_events} pedestal-run event(s)")
        print(f"Baseline mean: {baseline_mean:.3f} mV")
        print(f"RMS noise:     {rms_noise:.3f} mV")
        save_pedestal_calibration(baseline_mean, rms_noise, args.lappd)
        sys.exit(0)

    if args.preview:
        n = args.preview
        preview_baseline = 0.0
        if args.lappd:
            try:
                preview_baseline, _ = load_pedestal_calibration(args.lappd)
            except FileNotFoundError:
                print(f"No pedestal found for LAPPD {args.lappd} — using 0 mV baseline for this preview.")
        preview_events = []
        for ev in iter_drs4_events(args.filepath, channel=args.channel):
            mask = (ev["t"] >= PULSE_WINDOW_NS[0]) & (ev["t"] <= PULSE_WINDOW_NS[1])
            v_win = positive_signal(ev["v"][mask], preview_baseline)
            ev["amplitude"] = float(v_win.max()) if len(v_win) > 0 else 0.0
            preview_events.append(ev)
            if len(preview_events) >= n:
                break
        print(f"Preview: parsed {len(preview_events)} event(s) (stopped early)")
        save_waveform_grid_pdf(preview_events, os.path.join(OUTDIR, "preview.pdf"), tag=TAG,
                                xlim=(PULSE_WINDOW_NS[0] - 9, PULSE_WINDOW_NS[1] + 40), ncols=1, nrows=2, baseline_mean=preview_baseline,
                                window_ns=PULSE_WINDOW_NS)
        sys.exit(0)

    # --- FULL ANALYSIS MODE ---
    if args.lappd:
        try:
            baseline_mean, rms_noise = load_pedestal_calibration(args.lappd)
            print(f"Using pedestal calibration for LAPPD {args.lappd}: "
                  f"baseline={baseline_mean:.3f} mV, RMS noise={rms_noise:.3f} mV")
            threshold = 5 * rms_noise
        except FileNotFoundError:
            print(f"No {pedestal_filepath(args.lappd)} found — falling back to windowed baseline.")
            baseline_mean, rms_noise, threshold = compute_baseline_and_threshold_fallback(args.filepath, channel=args.channel)
    else:
        print("No --lappd given — falling back to windowed baseline.")
        baseline_mean, rms_noise, threshold = compute_baseline_and_threshold_fallback(args.filepath, channel=args.channel)

    print(f"Threshold (5 sigma): {threshold:.3f} mV")
    print(f"Searching for hits only within {PULSE_WINDOW_NS} ns")
    print(f"Searching for afterpulses within {AFTERPULSE_WINDOW_NS} ns (hit events only)")

    total = 0
    n_hits = 0
    n_afterpulse = 0
    amplitudes, charges, hits_list, afterpulse_delays = [], [], [], []
    hit_samples, empty_samples, overlay_events = [], [], []
    OVERLAY_MAX = MIN_PLOTS  # at least 1000 waveforms on the sanity overlay too

    for ev in iter_drs4_events(args.filepath, channel=args.channel):
        if args.max_events is not None and total >= args.max_events:
            break
        total += 1
        if len(overlay_events) < OVERLAY_MAX:
            overlay_events.append(ev)

        mask = (ev["t"] >= PULSE_WINDOW_NS[0]) & (ev["t"] <= PULSE_WINDOW_NS[1])
        v_win = positive_signal(ev["v"][mask], baseline_mean)
        t_win = ev["t"][mask]

        if len(v_win) == 0:
            amp, is_hit, main_peak_t = 0.0, False, None
        else:
            idx = int(np.argmax(v_win))
            amp = float(v_win[idx])
            main_peak_t = float(t_win[idx])
            is_hit = bool(np.any(v_win > threshold))

        amplitudes.append(amp)
        charges.append(extract_integrated_charge(ev, baseline_mean, PULSE_WINDOW_NS))
        hits_list.append(is_hit)
        ev["amplitude"] = amp

        if is_hit:
            n_hits += 1
            if len(hit_samples) < MIN_PLOTS:
                hit_samples.append(ev)
            # Afterpulses only make sense relative to a real main pulse, so
            # this search only runs on events already classified as hits.
            found, delay = find_afterpulse(ev, baseline_mean, threshold,
                                            AFTERPULSE_WINDOW_NS, main_peak_t)
            if found:
                n_afterpulse += 1
                afterpulse_delays.append(delay)
        elif len(empty_samples) < MIN_PLOTS:
            empty_samples.append(ev)

    print(f"\nParsed {total} event(s) from {args.filepath}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for ev in overlay_events:
        plot_event(ev, ax=ax, baseline_mean=baseline_mean, window_ns=PULSE_WINDOW_NS)
    ax.set_xlim(PULSE_WINDOW_NS[0] - 9, PULSE_WINDOW_NS[1] + 40)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "drs4_waveforms.png"), dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUTDIR}/drs4_waveforms.png")

    amplitudes = np.array(amplitudes)
    charges = np.array(charges)
    hits = np.array(hits_list)
    n_empty = total - n_hits
    p0 = n_empty / total if total > 0 else float("nan")
    n_bar = -np.log(p0) if p0 > 0 else float("inf")

    print(f"Total events:  {total}")
    print(f"Hits:          {n_hits}")
    print(f"Empty:         {n_empty}")
    print(f"P(0):          {p0:.4f}")
    print(f"N_bar = -ln(P(0)): {n_bar:.4f}")
    print_purity_summary(n_bar, n_total=total, n_hits=n_hits)

    afterpulse_rate = n_afterpulse / n_hits * 100 if n_hits > 0 else float("nan")
    print(f"\n--- Afterpulse summary ---")
    print(f"  Search window:           {AFTERPULSE_WINDOW_NS} ns")
    print(f"  Main-pulse hits:         {n_hits}")
    print(f"  Hits with an afterpulse: {n_afterpulse}")
    print(f"  Afterpulse rate:         {afterpulse_rate:.2f}%  (of main-pulse hits)")
    print(f"  P(afterpulse | hit):     {afterpulse_rate / 100:.4f}")
    if afterpulse_delays:
        delays_arr = np.array(afterpulse_delays)
        print(f"  Delay: mean={delays_arr.mean():.2f} ns, std={delays_arr.std():.2f} ns, "
              f"min={delays_arr.min():.2f} ns, max={delays_arr.max():.2f} ns")
        if delays_arr.max() > AFTERPULSE_WINDOW_NS[1] - 5:
            print(f"  NOTE: some delays sit right at the {AFTERPULSE_WINDOW_NS[1]} ns window edge -- "
                  f"the true afterpulse population may extend beyond what this window can see, "
                  f"so the rate above could be an underestimate.")
    print(f"-----------------------------------")

    # Save the per-event arrays so spe_fit.py (or anything else) can work
    # from them without re-parsing the XML file. Saved in a "spectra" folder
    # OUTSIDE the per-run output directory (not mixed in with the plots/PDFs),
    # named after the run tag so files from different runs don't collide.
    SPECTRA_DIR = "spectra"
    os.makedirs(SPECTRA_DIR, exist_ok=True)
    spectra_path = os.path.join(SPECTRA_DIR, f"{TAG}_spectra_data.npz")
    np.savez(spectra_path, amplitudes=amplitudes, charges=charges, hits=hits,
             baseline_mean=baseline_mean, rms_noise=rms_noise, threshold=threshold)
    print(f"Saved {spectra_path} (amplitudes, charges, hits — for spe_fit.py)")

    plot_amplitude_histogram(amplitudes, os.path.join(OUTDIR, "amplitude_histogram_all.png"), tag=TAG,
                              hit_fraction=(1 - p0) * 100, purity=purity_from_nbar(n_bar) * 100)
    plot_charge_histogram(charges, os.path.join(OUTDIR, "charge_histogram_all.png"), tag=TAG)

    if afterpulse_delays:
        plot_afterpulse_histogram(afterpulse_delays, os.path.join(OUTDIR, "afterpulse_delay_histogram.png"), tag=TAG)

    if len(hit_samples) < MIN_PLOTS:
        print(f"\nWarning: only {len(hit_samples)} hit events available — plotting all of them.")
    if len(empty_samples) < MIN_PLOTS:
        print(f"Warning: only {len(empty_samples)} empty events available — plotting all of them.")

    save_waveform_grid_pdf(empty_samples, os.path.join(OUTDIR, "empty_waveforms.pdf"),
                            xlim=(PULSE_WINDOW_NS[0] - 9, PULSE_WINDOW_NS[1] + 40), tag=TAG, ncols=1, nrows=2,
                            baseline_mean=baseline_mean, window_ns=PULSE_WINDOW_NS)
    save_waveform_grid_pdf(hit_samples, os.path.join(OUTDIR, "hit_waveforms.pdf"),
                            xlim=(PULSE_WINDOW_NS[0] - 9, PULSE_WINDOW_NS[1] + 40), tag=TAG, ncols=1, nrows=2,
                            baseline_mean=baseline_mean, window_ns=PULSE_WINDOW_NS)