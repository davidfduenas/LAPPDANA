"""
Two-channel DRS4 analysis: LAPPD channel + fast photodiode channel
(set with --lappd-channel and --diode-channel; defaults CHN1 and CHN2).

- 1PE classification on the LAPPD channel (5 sigma above pedestal baseline, pulse window only)
- Amplitude and charge distributions for both channels (1PE-classified events)
- Timing: Delta t = t2 (LAPPD centroid) - t1 (photodiode centroid)

UNDERSHOOT HANDLING: both pulses are positive. The negative undershoot after
the pulse is an electronics artifact (capacitive pickup of the resistive
anode), not signal. Every calculation below works ONLY on the positive part
of the baseline-subtracted waveform (positive_signal()). Negative samples
never contribute to hit finding, amplitude, charge, peak finding, or the
centroid weights.

NOTE: the pedestal calibration (drs4_analysis.py --pedestal) must NOT be
clipped — baseline and RMS need both signs of the noise. Clipping only
happens here, after baseline subtraction.

Streams events one at a time — safe for large files.
"""

import re
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit
from scipy.special import erf

_SERIAL_RE = re.compile(r'<Serial>(\d+)</Serial>')
_DATA_RE = re.compile(r'<Data>([-\d.]+),([-\d.]+)</Data>')

# The arrival time of a pulse is calculated from the samples within this many ns
# of its peak (see centroid()).
LOCAL_HALFWIDTH_NS = 1.5

# The waveform PDFs are zoomed onto each pulse: its peak +/- this many ns, so the
# whole time region (peak +/- LOCAL_HALFWIDTH_NS) is visible with a little margin.
PDF_ZOOM_HALFWIDTH_NS = LOCAL_HALFWIDTH_NS + 0.5

# The zoomed-OUT view in the waveform PDFs: this many ns before the start and after the
# end of the search window.
PDF_OUT_BEFORE_NS = 9
PDF_OUT_AFTER_NS = 40

# Delta t zoom plots: range = main peak +/- ZOOM_HALFWIDTH_NS; the drawn histogram uses
# ZOOM_BIN_PS bins, the fit uses finer FIT_BIN_PS bins (the narrow core needs enough bins).
ZOOM_HALFWIDTH_NS = 1.5
ZOOM_BIN_PS = 75
FIT_BIN_PS = 40

# CFD: how far past the delayed peak (ns) to look for the negative lobe of the shaped signal.
CFD_LOBE_MARGIN_NS = 0.5


def _channel_section(block, channel):
    tag_open, tag_close = f"<{channel}>", f"</{channel}>"
    if tag_open not in block:
        return None
    return block[block.find(tag_open):block.find(tag_close)]


def iter_drs4_events_two_channel(filepath, ch_a="CHN1", ch_b="CHN2"):
    """Stream events one at a time, yielding both channels' (t, v) arrays."""
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
                    sec_a = _channel_section(block, ch_a)
                    sec_b = _channel_section(block, ch_b)
                    if sec_a is None or sec_b is None:
                        continue
                    pairs_a = _DATA_RE.findall(sec_a)
                    pairs_b = _DATA_RE.findall(sec_b)
                    if not pairs_a or not pairs_b:
                        continue
                    arr_a = np.array(pairs_a, dtype=float)
                    arr_b = np.array(pairs_b, dtype=float)
                    yield {
                        "serial": int(serial_m.group(1)),
                        "t_a": arr_a[:, 0], "v_a": arr_a[:, 1],
                        "t_b": arr_b[:, 0], "v_b": arr_b[:, 1],
                    }


def positive_signal(t, v, baseline, window_ns):
    """
    Baseline-subtract, restrict to the pulse window, and zero out everything
    below baseline. This is the ONLY form of the waveform the analysis uses.
    Returns (t_window, v_positive).
    """
    mask = (t >= window_ns[0]) & (t <= window_ns[1])
    return t[mask], np.clip(v[mask] - baseline, 0, None)


def is_hit(t, v, baseline, threshold, window_ns):
    _, v_pos = positive_signal(t, v, baseline, window_ns)
    return len(v_pos) > 0 and bool(np.any(v_pos > threshold))


def extract_amplitude_and_charge(t, v, baseline, window_ns, impedance_ohm=50):
    """Peak amplitude (mV) and integrated charge (pC) of the positive pulse."""
    t_w, v_pos = positive_signal(t, v, baseline, window_ns)
    if len(v_pos) < 2:
        return 0.0, 0.0
    amplitude = float(v_pos.max())
    integral_mV_ns = (np.trapezoid(v_pos, t_w) if hasattr(np, "trapezoid")
                       else np.trapz(v_pos, t_w))
    return amplitude, float(integral_mV_ns / impedance_ohm)  # (mV*ns)/ohm = pC


def centroid(t, v, baseline, window_ns, local_halfwidth_ns=LOCAL_HALFWIDTH_NS):
    """
    Arrival time: find the positive maximum, then v^2-weighted mean time of
    the POSITIVE samples within +/- local_halfwidth_ns of it. Undershoot
    samples are zero after clipping, so they carry no weight.
    """
    t_w, v_pos = positive_signal(t, v, baseline, window_ns)
    if len(v_pos) == 0 or v_pos.max() <= 0:
        return None
    peak_t = t_w[np.argmax(v_pos)]
    local = (t_w >= peak_t - local_halfwidth_ns) & (t_w <= peak_t + local_halfwidth_ns)
    w = v_pos[local] ** 2
    if w.sum() <= 0:
        return None
    return float(np.sum(t_w[local] * w) / w.sum())


def centroid_region(t, v, baseline, window_ns, local_halfwidth_ns=LOCAL_HALFWIDTH_NS):
    """
    The time span of the samples that centroid() actually uses: within
    +/- local_halfwidth_ns of the peak, and inside the search window.
    Returns (t_start, t_end, peak_time) or None.
    """
    t_w, v_pos = positive_signal(t, v, baseline, window_ns)
    if len(v_pos) == 0 or v_pos.max() <= 0:
        return None
    peak_t = t_w[np.argmax(v_pos)]
    local = (t_w >= peak_t - local_halfwidth_ns) & (t_w <= peak_t + local_halfwidth_ns)
    if not np.any(local):
        return None
    return float(t_w[local].min()), float(t_w[local].max()), float(peak_t)


def edge_time(t, v, baseline, window_ns, fraction=0.5):
    """
    Rising-edge time: when the leading edge reaches `fraction` of the pulse peak
    (linear interpolation between the two samples around that level). The peak is
    the highest sample in the window. Returns the time in ns, or None.
    """
    w = np.where((t >= window_ns[0]) & (t <= window_ns[1]))[0]
    if len(w) == 0:
        return None
    vs = v - baseline
    p = w[np.argmax(vs[w])]
    peak = vs[p]
    if peak <= 0:
        return None
    level = fraction * peak
    j = p
    while j > 0 and vs[j] >= level:          # walk back down the leading edge
        j -= 1
    if vs[j] >= level:                        # never came back below the level
        return None
    return float(t[j] + (t[j + 1] - t[j]) * (level - vs[j]) / (vs[j + 1] - vs[j]))


def cfd_signals(t, v, baseline, fraction, delay_ns):
    """
    The steps of constant-fraction discrimination (Knoll, fig. 17.52) on a waveform.
    Returns four arrays: (a) the baseline-subtracted input, (b) the input attenuated
    by `fraction`, (c) the input inverted and delayed by `delay_ns`, and (d) the sum
    of (b) and (c), whose zero crossing is the CFD time.
    """
    vs = v - baseline
    attenuated = fraction * vs
    inverted_delayed = -np.interp(t - delay_ns, t, vs)
    return vs, attenuated, inverted_delayed, attenuated + inverted_delayed


def cfd_time(t, v, baseline, window_ns, fraction=0.3, delay_ns=1.0):
    """
    Constant-fraction discrimination, done in software the way the analog circuit
    does it (Knoll, fig. 17.52): scale the pulse by `fraction`, subtract a copy
    delayed by `delay_ns`, and take the zero crossing of the result:

        s(t) = fraction * v(t) - v(t - delay)

    The crossing time does not depend on the pulse amplitude for pulses of the same
    shape. It lies `delay_ns` after the point where the leading edge reaches
    `fraction` of the peak; use the same delay on both channels and it cancels in
    Delta t. The crossing is found next to the strongly negative lobe of s(t) that
    appears when the delayed copy sits on the peak, so noise at the foot of the pulse
    is not mistaken for it. Returns the time in ns, or None.
    """
    w = np.where((t >= window_ns[0]) & (t <= window_ns[1]))[0]
    if len(w) == 0:
        return None
    vs, _, _, s = cfd_signals(t, v, baseline, fraction, delay_ns)
    p = w[np.argmax(vs[w])]
    if vs[p] <= 0:
        return None
    near = np.where((t >= t[p]) & (t <= t[p] + delay_ns + CFD_LOBE_MARGIN_NS))[0]
    if len(near) == 0:
        return None
    k = near[np.argmin(s[near])]              # bottom of the negative lobe
    if s[k] >= 0:
        return None
    while k > 0 and s[k] < 0:                 # back up to where s was last >= 0
        k -= 1
    if s[k] < 0:
        return None
    return float(t[k] + (t[k + 1] - t[k]) * s[k] / (s[k] - s[k + 1]))


def diode_baseline(v):
    """
    Per-event photodiode baseline: median of the full 200 ns trace. The pulse
    occupies only a few ns of it, so the median is essentially the baseline.
    (Previously the photodiode channel used a hardcoded 0.0 mV baseline.)
    """
    return float(np.median(v))


def save_two_channel_pdf(events, t1_vals, t2_vals, amps_a, charges_a, amps_b, charges_b,
                          bases_b, filename, window_ns, lappd_baseline, events_per_page=2, tag="",
                          diode_window_ns=None,
                          lappd_channel="CHN1", diode_channel="CHN2", show_dt=False, marks=None):
    """
    Both channels per event, each shown twice side by side:
      left  = zoomed OUT around the search window (window start - PDF_OUT_BEFORE_NS to
              window end + PDF_OUT_AFTER_NS), so you see the pulse in context;
      right = zoomed IN on the pulse (peak +/- PDF_ZOOM_HALFWIDTH_NS), with a dot on
              every sample.
    The waveform is always drawn unclipped. Yellow = search window, green = the
    region the time is calculated from, dashed line = the calculated time (centroid).
    show_dt adds each event's Delta t to the panel titles.
    marks (optional): per event, {"lappd": [(label, time, color, linestyle), ...],
    "diode": [...]} -- extra times (rising edge, CFD) drawn as lines on the zoomed-in panels.
    """
    n = len(events)
    n_pages = int(np.ceil(n / events_per_page)) if n > 0 else 0
    if n_pages == 0:
        print(f"No events to plot for {filename} — skipping.")
        return
    if diode_window_ns is None:
        diode_window_ns = window_ns

    def fit_ylim(ax, t, y, x_lo, x_hi):
        vis = (t >= x_lo) & (t <= x_hi)
        if np.any(vis):
            y_lo, y_hi = y[vis].min(), y[vis].max()
            pad = 0.08 * max(y_hi - y_lo, 1e-6)
            ax.set_ylim(y_lo - pad, y_hi + pad)

    with PdfPages(filename) as pdf:
        for page in range(n_pages):
            lo = page * events_per_page
            fig, axes = plt.subplots(events_per_page * 2, 2, figsize=(16, 3.5 * events_per_page * 2))
            axes = np.asarray(axes).reshape(events_per_page * 2, 2)
            for i in range(events_per_page):
                k = lo + i
                if k >= n:
                    for r in (2 * i, 2 * i + 1):
                        axes[r, 0].set_visible(False)
                        axes[r, 1].set_visible(False)
                    continue
                ev = events[k]
                panels = [
                    (2 * i, ev["t_a"], ev["v_a"], lappd_baseline,
                     f"evt {ev['serial']} — {lappd_channel} (LAPPD)", t2_vals[k], amps_a[k], charges_a[k], window_ns, "lappd"),
                    (2 * i + 1, ev["t_b"], ev["v_b"], bases_b[k],
                     f"evt {ev['serial']} — {diode_channel} (photodiode)", t1_vals[k], amps_b[k], charges_b[k], diode_window_ns, "diode"),
                ]
                for row, t, v, base, label, tc, amp, charge, win, chan in panels:
                    y = v - base
                    region = centroid_region(t, v, base, win)
                    out_lo, out_hi = win[0] - PDF_OUT_BEFORE_NS, win[1] + PDF_OUT_AFTER_NS
                    if region is not None:
                        in_lo, in_hi = region[2] - PDF_ZOOM_HALFWIDTH_NS, region[2] + PDF_ZOOM_HALFWIDTH_NS
                    else:
                        in_lo, in_hi = win[0] - 1.0, win[1] + 1.0

                    for col, (x_lo, x_hi) in enumerate([(out_lo, out_hi), (in_lo, in_hi)]):
                        ax = axes[row, col]
                        if col == 0:
                            ax.plot(t, y, linewidth=0.8, color="steelblue")
                        else:
                            ax.plot(t, y, linewidth=0.9, color="steelblue", marker=".", markersize=4)
                        ax.axhline(0, color="gray", linewidth=0.4)
                        ax.axvspan(win[0], win[1], color="yellow", alpha=0.15)
                        if region is not None:
                            ax.axvspan(region[0], region[1], color="tab:green", alpha=0.25,
                                       label=f"time region (peak ± {LOCAL_HALFWIDTH_NS:g} ns)")
                        if tc is not None:
                            ax.axvline(tc, color="crimson", linestyle="--", label=f"centroid={tc:.3f} ns")
                        if col == 1 and marks is not None and marks[k] is not None:
                            for m_label, m_time, m_color, m_style in marks[k].get(chan, []):
                                if m_time is not None:
                                    ax.axvline(m_time, color=m_color, linestyle=m_style, linewidth=1.1,
                                               label=f"{m_label}={m_time:.3f} ns")
                        ax.set_xlim(x_lo, x_hi)
                        fit_ylim(ax, t, y, x_lo, x_hi)
                        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                        if col == 0:
                            ax.set_ylabel("mV")
                            dt_txt = f" | Δt={t2_vals[k] - t1_vals[k]:.2f} ns" if show_dt else ""
                            ax.set_title(f"{label} | amp={amp:.2f} mV, Q={charge:.3f} pC{dt_txt}", fontsize=9)
                        else:
                            ax.set_title("zoom on the pulse", fontsize=8)
                            ax.legend(fontsize=7)
            axes[-1, 0].set_xlabel("time (ns)")
            axes[-1, 1].set_xlabel("time (ns)")
            fig.suptitle(f"{tag} (page {page + 1}/{n_pages})")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {n} events across {n_pages} page(s) to {filename}")


def save_records_pdf(records, filename, window_ns, lappd_baseline, diode_window_ns, tag,
                     lappd_channel, diode_channel, show_dt=False):
    save_two_channel_pdf(
        [r["ev"] for r in records], [r["t1"] for r in records], [r["t2"] for r in records],
        [r["amp_a"] for r in records], [r["charge_a"] for r in records],
        [r["amp_b"] for r in records], [r["charge_b"] for r in records],
        [r["base_b"] for r in records], filename, window_ns, lappd_baseline, tag=tag,
        diode_window_ns=diode_window_ns, lappd_channel=lappd_channel, diode_channel=diode_channel,
        show_dt=show_dt, marks=[r.get("marks") for r in records])


def save_timing_steps_pdf(records, filename, window_ns, lappd_baseline, diode_window_ns, tag,
                          lappd_channel, diode_channel, edge_fraction, cfd_fraction, cfd_delay):
    """
    One page per event, LAPPD on the left and photodiode on the right, with the steps of
    the two new timing methods drawn the way Knoll's fig. 17.52 draws them:
      (a) input signal, with the rising-edge level (edge_fraction of the peak) and the
          time at which the leading edge reaches it;
      (b) the signal attenuated by the CFD fraction;
      (c) the signal inverted and delayed by the CFD delay (thin gray = inverted, not delayed);
      (d) the sum of (b) and (c), whose zero crossing is the CFD time.
    """
    n = len(records)
    if n == 0:
        print(f"No events to plot for {filename} — skipping.")
        return
    edge_name = f"{edge_fraction * 100:g}% rising edge"
    # (channel name, time key, voltage key, baseline (None = the event's own), window, edge key, CFD key)
    cols = [
        (lappd_channel + " (LAPPD)", "t_a", "v_a", lappd_baseline, window_ns, "te_a", "tc_a"),
        (diode_channel + " (photodiode)", "t_b", "v_b", None, diode_window_ns, "te_b", "tc_b"),
    ]
    with PdfPages(filename) as pdf:
        for k, rec in enumerate(records):
            ev = rec["ev"]
            fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex="col")
            for c, (ch_name, tk, vk, base, win, te_key, tc_key) in enumerate(cols):
                t, v = ev[tk], ev[vk]
                base = rec["base_b"] if base is None else base
                vs, att, inv_del, shaped = cfd_signals(t, v, base, cfd_fraction, cfd_delay)
                w = np.where((t >= win[0]) & (t <= win[1]))[0]
                p = w[np.argmax(vs[w])]
                t_p, peak = t[p], vs[p]
                te, tc = rec[te_key], rec[tc_key]
                x_lo, x_hi = t_p - 2.5, t_p + cfd_delay + 2.5
                vis = (t >= x_lo) & (t <= x_hi)

                ax = axes[0, c]                                           # (a) input
                ax.plot(t, vs, linewidth=0.9, color="steelblue", marker=".", markersize=3)
                ax.axvspan(win[0], win[1], color="yellow", alpha=0.15)
                ax.axhline(peak, color="gray", linestyle=":", linewidth=0.8)
                ax.axhline(edge_fraction * peak, color="darkorange", linestyle=":", linewidth=0.9,
                           label=f"{edge_fraction * 100:g}% of the peak")
                if te is not None:
                    ax.axvline(te, color="darkorange", linestyle=":", linewidth=1.1)
                    ax.plot([te], [edge_fraction * peak], "o", color="darkorange", markersize=5,
                            label=f"{edge_name} = {te:.3f} ns")
                ax.set_title(f"evt {ev['serial']} — {ch_name}", fontsize=10)
                ax.set_ylabel("(a) input signal (mV)")

                ax = axes[1, c]                                           # (b) attenuated
                ax.plot(t, att, linewidth=0.9, color="tab:green", marker=".", markersize=3)
                ax.axhline(cfd_fraction * peak, color="gray", linestyle=":", linewidth=0.8,
                           label=f"{cfd_fraction:g} × peak")
                ax.set_ylabel(f"(b) attenuated, f = {cfd_fraction:g} (mV)")

                ax = axes[2, c]                                           # (c) inverted and delayed
                ax.plot(t, -vs, linewidth=0.7, color="gray", linestyle="--", label="inverted, not delayed")
                ax.plot(t, inv_del, linewidth=0.9, color="tab:red", marker=".", markersize=3,
                        label=f"inverted and delayed by {cfd_delay:g} ns")
                ax.set_ylabel("(c) inverted and delayed (mV)")

                ax = axes[3, c]                                           # (d) the sum
                ax.plot(t, shaped, linewidth=0.9, color="purple", marker=".", markersize=3)
                if tc is not None:
                    ax.axvline(tc, color="purple", linestyle="-.", linewidth=1.1)
                    ax.plot([tc], [0], "o", color="purple", markersize=5,
                            label=f"CFD zero crossing = {tc:.3f} ns")
                ax.set_ylabel("(d) sum of (b) and (c) (mV)")
                ax.set_xlabel("time (ns)")

                shown = [vs[vis], att[vis], np.concatenate([inv_del[vis], -vs[vis]]), shaped[vis]]
                for r in range(4):
                    a = axes[r, c]
                    a.axhline(0, color="gray", linewidth=0.5)
                    a.set_xlim(x_lo, x_hi)
                    if len(shown[r]):
                        lo_y, hi_y = shown[r].min(), shown[r].max()
                        pad = 0.08 * max(hi_y - lo_y, 1e-6)
                        a.set_ylim(lo_y - pad, hi_y + pad)
                    a.xaxis.set_minor_locator(AutoMinorLocator(5))
                    if a.get_legend_handles_labels()[0]:
                        a.legend(fontsize=7, loc="upper right")
            fig.suptitle(f"{tag} — steps of the rising-edge and constant-fraction timing "
                         f"(page {k + 1}/{n})", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {n} events to {filename}")


def _fit_window(counts, edges, lo, hi, mu0, sigma0):
    """Gaussian fit to the bins whose centres lie in [lo, hi]. Returns a dict or None."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    sel = (centers >= lo) & (centers <= hi)
    if sel.sum() < 5:
        return None
    lo_e, hi_e = edges[:-1][sel], edges[1:][sel]
    y = counts[sel]

    def model(_x, n, mu, sig):
        z = np.sqrt(2.0) * sig
        return n * 0.5 * (erf((hi_e - mu) / z) - erf((lo_e - mu) / z))

    err = np.sqrt(np.maximum(y, 1.0))
    popt = [y.sum(), mu0, sigma0]
    try:
        for _ in range(2):
            # 2nd pass weights by the fitted model instead of the noisy counts,
            # which avoids pulling the fit low when bins have few events
            popt, pcov = curve_fit(model, centers[sel], y, p0=popt, sigma=err, absolute_sigma=True,
                                   bounds=([0.0, lo, 1e-6], [np.inf, hi, hi - lo]),
                                   maxfev=20000)
            err = np.sqrt(np.maximum(model(None, *popt), 1.0))
    except (RuntimeError, ValueError):
        return None

    n, mu, sig = popt
    perr = np.sqrt(np.diag(pcov))
    ndf = int(sel.sum()) - 3
    chi2 = float(np.sum(((y - model(None, *popt)) / err) ** 2))
    return {"n": n, "mu": mu, "sigma": sig,
            "mu_err": perr[1], "sigma_err": perr[2],
            "chi2_ndf": chi2 / ndf if ndf > 0 else float("nan"),
            "fit_lo": lo, "fit_hi": hi, "n_bins": int(sel.sum()), "width": edges[1] - edges[0]}


def fit_gaussian_peak(counts, edges, n_left=1.25, n_right=0.25):
    """
    Fit a plain Gaussian to the CORE of the main peak of a histogram.

    The peak is located from a lightly smoothed copy of the histogram (tallest
    point and its width at half height). Only the bins from n_left sigma before
    the peak to n_right sigma after it are fitted. The window stops just after
    the peak on purpose: the tail of late events sits on that side, and the
    tails and any other bumps are ignored. The window is then re-centred on the
    fitted mean and width and the fit is repeated. The model is the Gaussian
    integrated over each bin.

    Returns a dict (mu, sigma, errors, chi2_ndf, fit range) or None.
    """
    counts = np.asarray(counts, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    if counts.sum() < 10:
        return None

    smooth = np.convolve(counts, np.ones(3) / 3.0, mode="same")
    i0 = int(np.argmax(smooth))
    half = smooth[i0] / 2.0
    left = i0
    while left > 0 and smooth[left - 1] >= half:
        left -= 1
    right = i0
    while right < len(smooth) - 1 and smooth[right + 1] >= half:
        right += 1
    sigma0 = max((right - left + 1) * width / 2.355, width)
    mu0 = centers[i0]

    fit = _fit_window(counts, edges, mu0 - n_left * sigma0, mu0 + n_right * sigma0, mu0, sigma0)
    if fit is None:
        return None
    refit = _fit_window(counts, edges, fit["mu"] - n_left * fit["sigma"],
                        fit["mu"] + n_right * fit["sigma"], fit["mu"], fit["sigma"])
    return refit if refit is not None else fit


def dt_zoom_plot(dt_ns, outpath, title_prefix, tag, print_name="Delta t"):
    """
    Delta t (ns) zoomed on its main peak, with a Gaussian fit to the core of the peak
    (the black curve and the box in the corner are the only statistics on the plot).
    Returns (fit_dict_or_None, zoom_center_ns).
    """
    dt_ns = np.asarray(dt_ns, dtype=float)
    coarse, coarse_edges = np.histogram(dt_ns, bins=max(int((dt_ns.max() - dt_ns.min()) / 0.1), 10))
    peak_guess = 0.5 * (coarse_edges[np.argmax(coarse)] + coarse_edges[np.argmax(coarse) + 1])
    near = dt_ns[np.abs(dt_ns - peak_guess) < 0.5]
    zoom_center = float(np.median(near)) if len(near) > 0 else float(peak_guess)
    zoom_ns = (zoom_center - ZOOM_HALFWIDTH_NS, zoom_center + ZOOM_HALFWIDTH_NS)
    zoom_bins = int(round(2 * ZOOM_HALFWIDTH_NS * 1000 / ZOOM_BIN_PS))
    in_zoom = dt_ns[(dt_ns >= zoom_ns[0]) & (dt_ns <= zoom_ns[1])]
    if len(in_zoom) == 0:
        print(f"No {print_name} values in ({zoom_ns[0]:.2f}, {zoom_ns[1]:.2f}) ns — skipping zoom plot.")
        return None, zoom_center
    print(f"{print_name} in ({zoom_ns[0]:.2f}, {zoom_ns[1]:.2f}) ns: n={len(in_zoom)} of {len(dt_ns)}, "
          f"mean={in_zoom.mean() * 1000:.1f} ps, std={in_zoom.std() * 1000:.1f} ps")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges, _ = ax.hist(in_zoom, bins=zoom_bins, range=zoom_ns, color="steelblue", alpha=0.8)
    fit = None
    for fit_bin_ps in (FIT_BIN_PS, FIT_BIN_PS / 2, FIT_BIN_PS / 4):
        # a very narrow core needs finer bins to have enough of them inside the fit window
        fit_counts, fit_edges = np.histogram(
            in_zoom, bins=int(round(2 * ZOOM_HALFWIDTH_NS * 1000 / fit_bin_ps)), range=zoom_ns)
        fit = fit_gaussian_peak(fit_counts, fit_edges)
        if fit is not None:
            break
    if fit is not None:
        xs = np.linspace(edges[0], edges[-1], 1000)
        # scaled to the DRAWN bin width (the fit itself used finer bins)
        ys = (fit["n"] * (edges[1] - edges[0]) / (fit["sigma"] * np.sqrt(2 * np.pi))
              * np.exp(-0.5 * ((xs - fit["mu"]) / fit["sigma"]) ** 2))
        ax.plot(xs, ys, color="black", linewidth=1.5)
        ax.text(0.98, 0.98,
                f"Gaussian fit (core):\nmean = {fit['mu']:.3f} ± {fit['mu_err']:.3f} ns\n"
                f"σ = {fit['sigma'] * 1000:.1f} ± {fit['sigma_err'] * 1000:.1f} ps",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8))
        print(f"{print_name} core Gaussian fit: mean={fit['mu']:.4f} +/- {fit['mu_err']:.4f} ns, "
              f"sigma={fit['sigma'] * 1000:.1f} +/- {fit['sigma_err'] * 1000:.1f} ps, "
              f"chi2/ndf={fit['chi2_ndf']:.2f} "
              f"(fitted {fit['fit_lo']:.2f} to {fit['fit_hi']:.2f} ns, {fit['n_bins']} bins of {fit_bin_ps:g} ps)")
    else:
        print(f"{print_name} zoom: Gaussian fit failed or too few events -- plot has no fit.")
    ax.set_xlim(*zoom_ns)
    ax.set_xlabel("$\\Delta t = t_2 - t_1$ (ns)")
    ax.set_ylabel(f"Events / {(zoom_ns[1] - zoom_ns[0]) / zoom_bins * 1000:.0f} ps")
    ax.set_title(f"{title_prefix}, zoom {zoom_ns[0]:.2f} to {zoom_ns[1]:.2f} ns — {tag}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")
    return fit, zoom_center


def plot_dt_vs_amplitude(amps, dts_ns, center_ns, outpath, title, name="Delta t"):
    """
    Delta t against LAPPD peak amplitude, one dot per event, with the median
    Delta t in amplitude bins (equal numbers of events per bin) on top. The
    median is used so the late tail doesn't drag it. If low-amplitude pulses sit
    at a different time from high-amplitude ones, that is amplitude walk /
    pulse-shape dependence of the time, not a real difference in arrival.
    """
    amps = np.asarray(amps, dtype=float)
    dts_ns = np.asarray(dts_ns, dtype=float)
    n = len(amps)
    half_range = 1.5      # same width as the zoom plot, so ps-level trends are visible
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(amps, dts_ns, s=6, color="steelblue", alpha=0.4, linewidths=0)

    n_bins = int(min(8, max(3, n // 60)))
    edges = np.quantile(amps, np.linspace(0, 1, n_bins + 1))
    rows = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = k == n_bins - 1
        sel = (amps >= lo) & ((amps <= hi) if last else (amps < hi))
        if sel.sum() < 5:
            continue
        d = dts_ns[sel]
        med = float(np.median(d))
        mad = 1.4826 * float(np.median(np.abs(d - med)))
        err = 1.2533 * mad / np.sqrt(sel.sum())          # error of the median
        rows.append((float(np.median(amps[sel])), med, err, lo, hi, int(sel.sum())))
    if rows:
        r = np.array(rows)
        ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt="o", color="crimson", capsize=3,
                    label="median per amplitude bin")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_ylim(center_ns - half_range, center_ns + half_range)
    ax.set_xlabel("LAPPD peak amplitude (mV)")
    ax.set_ylabel("$\\Delta t = t_2 - t_1$ (ns)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")

    n_out = int(np.sum(np.abs(dts_ns - center_ns) > half_range))
    print(f"{name} vs LAPPD amplitude, median per bin ({n_out} of {n} events are outside the plotted "
          f"+/-{half_range:g} ns):")
    for amp_med, med, err, lo, hi, cnt in rows:
        print(f"  amp {lo:6.1f} to {hi:6.1f} mV: n={cnt:4d}, median Delta t = {med:8.3f} ns "
              f"(+/- {err * 1000:.0f} ps)")


def side_by_side_hist(a, b, xlabel, title, outpath, log=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(a, bins=60, color="crimson", alpha=0.8)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Events")
    ax1.set_title("LAPPD")
    ax2.hist(b, bins=60, color="steelblue", alpha=0.8)
    ax2.set_xlabel(xlabel)
    ax2.set_title("Photodiode")
    if log:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-channel LAPPD/photodiode 1PE + timing analysis")
    parser.add_argument("filepath", help="path to the DRSOSC XML file (both channels present)")
    parser.add_argument("--lappd", type=str, required=True,
                         help="LAPPD name — loads pedestal_lappd_<name>.txt for the LAPPD channel baseline")
    parser.add_argument("--lappd-channel", type=str, default="CHN1",
                         help="DRS4 channel with the LAPPD signal (default: CHN1)")
    parser.add_argument("--diode-channel", type=str, default="CHN2",
                         help="DRS4 channel with the photodiode signal (default: CHN2)")
    parser.add_argument("--window", type=float, nargs=2, default=[10, 20],
                         help="pulse window in ns, e.g. --window 10 20")
    parser.add_argument("--diode-shift", type=float, default=10.0,
                         help="shift of the photodiode window to the right of --window, in ns "
                              "(default: 10, so --window 10 20 gives 20 30 for the photodiode)")
    parser.add_argument("--t1-range", type=float, nargs=2, action="append", default=None,
                         metavar=("LO_NS", "HI_NS"),
                         help="write a waveform PDF of the events whose PHOTODIODE time (t1) is in this "
                              "range, in ns. Can be given several times (one PDF each), e.g. to compare "
                              "a bump with the main peak: --t1-range 42.3 42.6 --t1-range 41.8 42.0")
    parser.add_argument("--min-amplitude", type=float, default=None, metavar="MV",
                         help="optional: also drop events whose LAPPD peak amplitude is below MV (mV). "
                              "The pedestal 5-sigma threshold is always applied first, so a value at or "
                              "below it has no effect. Default: no extra cut.")
    parser.add_argument("--edge-fraction", type=float, default=0.5, metavar="F",
                         help="rising-edge time: the fraction of the pulse peak at which the leading "
                              "edge is timed (default 0.5)")
    parser.add_argument("--cfd-fraction", type=float, default=0.3, metavar="F",
                         help="constant-fraction discrimination: the fraction f of the pulse that is "
                              "subtracted from the delayed copy (default 0.3)")
    parser.add_argument("--cfd-delay", type=float, default=1.0, metavar="NS",
                         help="constant-fraction discrimination: the delay in ns; it should be longer "
                              "than the pulse rise time (default 1.0)")
    parser.add_argument("--range", type=float, nargs=2, default=None, metavar=("LO_PS", "HI_PS"),
                         help="Delta t range (ps) of the events shown in the waveform PDF. "
                              "Default: the same window as the zoom plot (main peak +/- 1.5 ns).")
    args = parser.parse_args()

    window_ns = tuple(args.window)                                   # LAPPD
    diode_window_ns = (window_ns[0] + args.diode_shift,              # photodiode
                       window_ns[1] + args.diode_shift)
    explicit_range = args.range is not None
    MAX_RANGE_EVENTS = 100      # waveforms in the PDF
    MAX_OUTLIER_EVENTS = 100    # waveforms in the outlier PDF
    MAX_STEP_EVENTS = 100       # waveforms in the step-by-step timing PDF
    OUTLIER_NSIGMA = 3.0        # "outside the main peak" = more than this many core sigma from its mean
    STORE_MAX = 2000            # selected events kept in memory, so the PDF range can follow the peak
    zoom_center = None          # set below once the main peak is found
    core_fit = None             # the core Gaussian fit, set below

    pedestal_path = f"pedestal_lappd_{args.lappd}.txt"
    try:
        with open(pedestal_path) as f:
            lines = f.readlines()
        baseline_mean = float(lines[0])
        rms_noise = float(lines[1])
    except FileNotFoundError:
        print(f"No {pedestal_path} found. Run drs4_analysis.py --pedestal --lappd {args.lappd} "
              f"on a laser-blocked {args.lappd_channel} file first (with --channel {args.lappd_channel}).")
        sys.exit(1)
    threshold = 5 * rms_noise
    print(f"LAPPD channel {args.lappd_channel}, window {window_ns} ns")
    print(f"Photodiode channel {args.diode_channel}, window {diode_window_ns} ns")
    print(f"{args.lappd_channel} baseline={baseline_mean:.3f} mV, RMS={rms_noise:.3f} mV, threshold={threshold:.3f} mV")

    # Optional amplitude cut, on top of the pedestal threshold
    amp_cut = None
    if args.min_amplitude is not None:
        if args.min_amplitude <= threshold:
            print(f"Amplitude cut {args.min_amplitude:g} mV is at or below the pedestal threshold "
                  f"({threshold:.3f} mV), so it removes nothing -- running without an extra cut.")
        else:
            amp_cut = args.min_amplitude
            print(f"Amplitude cut: LAPPD peak amplitude >= {amp_cut:g} mV "
                  f"({amp_cut / rms_noise:.1f} sigma of the pedestal noise)")

    TAG = os.path.splitext(os.path.basename(args.filepath))[0]
    OUTDIR = f"{TAG}_{args.lappd_channel}_{args.diode_channel}"
    if amp_cut is not None:
        OUTDIR += f"_ampcut{amp_cut:g}mV"
    print(f"Output directory: {OUTDIR}/")
    os.makedirs(OUTDIR, exist_ok=True)

    lappd_amps, lappd_charges, diode_amps, diode_charges = [], [], [], []
    t1_list, t2_list, delta_ts = [], [], []
    dt_amps = []   # LAPPD amplitude of each event in delta_ts
    # the same events timed two more ways: rising edge, and constant-fraction discrimination
    edge_label = f"{args.edge_fraction * 100:g}% rising edge"
    cfd_label = f"CFD (f={args.cfd_fraction:g}, delay={args.cfd_delay:g} ns)"
    dt_edge, dt_edge_amps = [], []
    dt_cfd, dt_cfd_amps = [], []
    n_total = n_hits = n_cut = 0

    stored_all = []     # first STORE_MAX selected events: the waveform PDFs draw from these
    stored_range = []   # with --range: the first selected events inside that range

    # Centroid times for the centroid plots. With no cut these are ALL events (no
    # 1PE selection). With --min-amplitude they hold ONLY the selected events.
    all_t_lappd, all_t_diode = [], []
    cut_active = amp_cut is not None

    for ev in iter_drs4_events_two_channel(args.filepath, ch_a=args.lappd_channel, ch_b=args.diode_channel):
        n_total += 1
        base_b = diode_baseline(ev["v_b"])

        c_lappd = centroid(ev["t_a"], ev["v_a"], baseline_mean, window_ns)
        c_diode = centroid(ev["t_b"], ev["v_b"], base_b, diode_window_ns)
        if not cut_active:
            if c_lappd is not None:
                all_t_lappd.append(c_lappd)
            if c_diode is not None:
                all_t_diode.append(c_diode)

        if not is_hit(ev["t_a"], ev["v_a"], baseline_mean, threshold, window_ns):
            continue
        n_hits += 1

        amp_a, charge_a = extract_amplitude_and_charge(ev["t_a"], ev["v_a"], baseline_mean, window_ns)
        if amp_cut is not None and amp_a < amp_cut:
            n_cut += 1
            continue
        if cut_active:
            if c_lappd is not None:
                all_t_lappd.append(c_lappd)
            if c_diode is not None:
                all_t_diode.append(c_diode)
        amp_b, charge_b = extract_amplitude_and_charge(ev["t_b"], ev["v_b"], base_b, diode_window_ns)
        lappd_amps.append(amp_a)
        lappd_charges.append(charge_a)
        diode_amps.append(amp_b)
        diode_charges.append(charge_b)

        t2 = c_lappd  # LAPPD
        t1 = c_diode  # photodiode
        if t1 is None or t2 is None:
            continue
        t1_list.append(t1)
        t2_list.append(t2)
        dt = (t2 - t1) * 1000.0  # ns -> ps
        delta_ts.append(dt)
        dt_amps.append(amp_a)

        te_a = edge_time(ev["t_a"], ev["v_a"], baseline_mean, window_ns, args.edge_fraction)
        te_b = edge_time(ev["t_b"], ev["v_b"], base_b, diode_window_ns, args.edge_fraction)
        tc_a = cfd_time(ev["t_a"], ev["v_a"], baseline_mean, window_ns, args.cfd_fraction, args.cfd_delay)
        tc_b = cfd_time(ev["t_b"], ev["v_b"], base_b, diode_window_ns, args.cfd_fraction, args.cfd_delay)
        if te_a is not None and te_b is not None:
            dt_edge.append((te_a - te_b) * 1000.0)
            dt_edge_amps.append(amp_a)
        if tc_a is not None and tc_b is not None:
            dt_cfd.append((tc_a - tc_b) * 1000.0)
            dt_cfd_amps.append(amp_a)

        rec = {"ev": ev, "t1": t1, "t2": t2, "dt": dt, "base_b": base_b,
               "amp_a": amp_a, "charge_a": charge_a, "amp_b": amp_b, "charge_b": charge_b,
               "te_a": te_a, "te_b": te_b, "tc_a": tc_a, "tc_b": tc_b,
               "marks": {"lappd": [(edge_label, te_a, "darkorange", ":"), (cfd_label, tc_a, "purple", "-.")],
                         "diode": [(edge_label, te_b, "darkorange", ":"), (cfd_label, tc_b, "purple", "-.")]}}
        if len(stored_all) < STORE_MAX:
            stored_all.append(rec)
        if explicit_range and args.range[0] <= dt <= args.range[1] and len(stored_range) < MAX_RANGE_EVENTS:
            stored_range.append(rec)

    lappd_amps = np.array(lappd_amps)
    lappd_charges = np.array(lappd_charges)
    diode_amps = np.array(diode_amps)
    diode_charges = np.array(diode_charges)
    t1_arr = np.array(t1_list)
    t2_arr = np.array(t2_list)
    delta_ts = np.array(delta_ts)

    print(f"Total events: {n_total}, classified as 1PE hits: {n_hits}")
    if amp_cut is not None:
        print(f"Amplitude cut removed {n_cut} of {n_hits} hits, {n_hits - n_cut} left")

    # -- Centroid times, one separate plot per channel: all events, or only the
    #    selected ones when --min-amplitude is used --
    scope = "selected" if cut_active else "all"
    for times, name, color, fname, win in [
        (np.array(all_t_lappd), f"LAPPD ({args.lappd_channel})", "crimson",
         f"centroid_times_lappd_{scope}_events.png", window_ns),
        (np.array(all_t_diode), f"Photodiode ({args.diode_channel})", "steelblue",
         f"centroid_times_diode_{scope}_events.png", diode_window_ns),
    ]:
        if len(times) == 0:
            print(f"No centroids found for {name} — skipping plot.")
            continue
        print(f"{name} centroid, {scope} events: n={len(times)}, "
              f"mean={times.mean():.3f} ns, std={times.std() * 1000:.1f} ps")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(times, bins=100, range=win, color=color, alpha=0.8)
        ax.set_xlabel("Centroid time (ns)")
        ax.set_ylabel("Events")
        ax.set_title(f"{name} centroid time — {scope} events — {TAG}")
        fig.tight_layout()
        outpath = os.path.join(OUTDIR, fname)
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"Saved {outpath}")

    if len(lappd_amps) == 0:
        print("No events left to plot.")
        sys.exit(0)
    print(f"LAPPD amplitude: mean={lappd_amps.mean():.2f} mV, std={lappd_amps.std():.2f} mV")
    print(f"LAPPD charge:    mean={lappd_charges.mean():.4f} pC, std={lappd_charges.std():.4f} pC")
    print(f"Diode amplitude: mean={diode_amps.mean():.2f} mV, std={diode_amps.std():.2f} mV")
    print(f"Diode charge:    mean={diode_charges.mean():.4f} pC, std={diode_charges.std():.4f} pC")
    if len(delta_ts) > 0:
        print(f"t1 (diode):  mean={t1_arr.mean():.3f} ns, std={t1_arr.std() * 1000:.1f} ps")
        print(f"t2 (LAPPD):  mean={t2_arr.mean():.3f} ns, std={t2_arr.std() * 1000:.1f} ps")
        print(f"Delta t:     mean={delta_ts.mean():.2f} ps, std={delta_ts.std():.2f} ps")

    side_by_side_hist(lappd_amps, diode_amps, "Peak amplitude (mV)",
                      f"Amplitude distribution (1PE hits) — {TAG}",
                      os.path.join(OUTDIR, "amplitude_distribution.png"))
    side_by_side_hist(lappd_charges, diode_charges, "Integrated charge (pC)",
                      f"Charge distribution (1PE hits) — {TAG}",
                      os.path.join(OUTDIR, "charge_distribution.png"))

    if len(delta_ts) > 0:
        # Diagnostic: each channel's arrival time on its own, to see which
        # channel any multi-cluster structure in Delta t comes from.
        side_by_side_hist(t2_arr, t1_arr, "Centroid time (ns)",
                          f"Arrival time per channel (t2 = LAPPD, t1 = diode) — {TAG}",
                          os.path.join(OUTDIR, "t1_t2_distributions.png"), log=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(delta_ts / 1000.0, bins=60, color="steelblue", alpha=0.8)
        ax.set_xlabel("$\\Delta t = t_2 - t_1$ (ns)")
        ax.set_ylabel("Events")
        ax.set_title(f"Timing distribution (1PE hits) — {TAG}")
        fig.tight_layout()
        outpath = os.path.join(OUTDIR, "timing_distribution.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"Saved {outpath}")

        # -- Delta t zoomed on its main peak, with a Gaussian fit to the core. The same
        #    plot is made for each way of timing the pulses: centroid, rising edge, CFD --
        dt_ns = delta_ts / 1000.0
        core_fit, zoom_center = dt_zoom_plot(
            dt_ns, os.path.join(OUTDIR, "timing_distribution_zoom.png"),
            "Timing distribution (centroid)", TAG, "Delta t")
        plot_dt_vs_amplitude(dt_amps, dt_ns, zoom_center, os.path.join(OUTDIR, "timing_vs_amplitude.png"),
                             f"Delta t (centroid) vs LAPPD amplitude (1PE hits) — {TAG}", "Delta t")

        method_fits = [("centroid", core_fit, len(dt_ns), dt_ns, zoom_center)]
        for m_label, m_dt, m_amps, m_tag in [(edge_label, dt_edge, dt_edge_amps, "edge"),
                                              (cfd_label, dt_cfd, dt_cfd_amps, "cfd")]:
            if len(m_dt) == 0:
                print(f"{m_label}: no event could be timed this way -- no plots.")
                continue
            m_dt_ns = np.asarray(m_dt) / 1000.0
            print(f"{m_label}: timed {len(m_dt)} of {len(dt_ns)} events")
            m_fit, m_center = dt_zoom_plot(
                m_dt_ns, os.path.join(OUTDIR, f"timing_distribution_zoom_{m_tag}.png"),
                f"Timing distribution ({m_label})", TAG, f"Delta t ({m_label})")
            plot_dt_vs_amplitude(m_amps, m_dt_ns, m_center,
                                 os.path.join(OUTDIR, f"timing_vs_amplitude_{m_tag}.png"),
                                 f"Delta t ({m_label}) vs LAPPD amplitude (1PE hits) — {TAG}",
                                 f"Delta t ({m_label})")
            method_fits.append((m_label, m_fit, len(m_dt), m_dt_ns, m_center))

        print("\nDelta t by way of timing the pulses. 'core sigma' is the Gaussian fit to the core of the peak;")
        print("'std' is the spread of every event in the zoom window, so it also contains the tails and any walk:")
        for m_label, m_fit, m_n, m_arr, m_ctr in method_fits:
            in_win = m_arr[np.abs(m_arr - m_ctr) < ZOOM_HALFWIDTH_NS]
            std_txt = f"{in_win.std() * 1000:6.1f} ps" if len(in_win) > 1 else "   n/a"
            core_txt = (f"{m_fit['sigma'] * 1000:6.1f} +/- {m_fit['sigma_err'] * 1000:4.1f} ps"
                        if m_fit is not None else "   no fit        ")
            print(f"  {m_label:<44s} core sigma = {core_txt} | std = {std_txt} | {m_n} events timed")

    # -- Waveform PDF: events whose Delta t is in the chosen range --
    if explicit_range:
        lo_ps, hi_ps = args.range
        how = "from --range"
    elif zoom_center is not None:
        lo_ps = (zoom_center - ZOOM_HALFWIDTH_NS) * 1000.0
        hi_ps = (zoom_center + ZOOM_HALFWIDTH_NS) * 1000.0
        how = "centred on the main peak"
    else:
        lo_ps = hi_ps = None
    if lo_ps is None:
        print("Waveform PDF: no Delta t values, nothing to plot.")
    else:
        lo_ns, hi_ns = lo_ps / 1000.0, hi_ps / 1000.0
        pool = stored_range if explicit_range else stored_all
        chosen = [r for r in pool if lo_ps <= r["dt"] <= hi_ps][:MAX_RANGE_EVENTS]
        print(f"Waveform PDF: {len(chosen)} events with Delta t in [{lo_ns:.2f}, {hi_ns:.2f}] ns ({how})")
        if len(chosen) == 0:
            print("  No events in that range -- use --range LO_PS HI_PS (in ps) to choose one.")
        save_records_pdf(chosen, os.path.join(OUTDIR, f"waveforms_{lo_ns:.2f}_to_{hi_ns:.2f}ns.pdf"),
                         window_ns, baseline_mean, diode_window_ns,
                         f"{TAG}: Delta t in [{lo_ns:.2f}, {hi_ns:.2f}] ns",
                         args.lappd_channel, args.diode_channel)

    # -- Waveforms of the events OUTSIDE the main peak (both channels), with the
    #    region each time is calculated from and the calculated time --
    if core_fit is not None:
        c_mu, c_half = core_fit["mu"], OUTLIER_NSIGMA * core_fit["sigma"]
        how_out = f"core fit mean {c_mu:.3f} ns ± {OUTLIER_NSIGMA:g} sigma"
    elif zoom_center is not None:
        c_mu, c_half = zoom_center, 0.75
        how_out = f"peak {c_mu:.3f} ns ± 0.75 ns (the core fit failed)"
    else:
        c_mu = None
    if c_mu is None:
        print("Outlier waveforms: no Delta t values, nothing to plot.")
    else:
        out_lo, out_hi = c_mu - c_half, c_mu + c_half
        is_out = [abs(r["dt"] / 1000.0 - c_mu) > c_half for r in stored_all]
        outliers = [r for r, o in zip(stored_all, is_out) if o][:MAX_OUTLIER_EVENTS]
        print(f"Outlier waveforms: {sum(is_out)} of {len(stored_all)} stored events have Delta t outside "
              f"[{out_lo:.2f}, {out_hi:.2f}] ns ({how_out}); plotting {len(outliers)}")
        save_records_pdf(outliers, os.path.join(OUTDIR, "waveforms_outliers.pdf"),
                         window_ns, baseline_mean, diode_window_ns,
                         f"{TAG}: events outside Delta t = [{out_lo:.2f}, {out_hi:.2f}] ns",
                         args.lappd_channel, args.diode_channel, show_dt=True)

    # -- Waveforms of the events whose PHOTODIODE time (t1) is in a chosen range --
    for lo_t1, hi_t1 in (args.t1_range or []):
        chosen = [r for r in stored_all if lo_t1 <= r["t1"] <= hi_t1][:MAX_RANGE_EVENTS]
        n_found = sum(1 for r in stored_all if lo_t1 <= r["t1"] <= hi_t1)
        note = f" (searched the first {STORE_MAX} selected events)" if len(stored_all) >= STORE_MAX else ""
        print(f"Diode-time PDF: {n_found} of {len(stored_all)} stored events have t1 in "
              f"[{lo_t1:.2f}, {hi_t1:.2f}] ns{note}; plotting {len(chosen)}")
        if not chosen:
            print("  No events in that range -- check the t1 plot for the range.")
            continue
        save_records_pdf(chosen, os.path.join(OUTDIR, f"waveforms_diode_{lo_t1:.2f}_to_{hi_t1:.2f}ns.pdf"),
                         window_ns, baseline_mean, diode_window_ns,
                         f"{TAG}: photodiode time in [{lo_t1:.2f}, {hi_t1:.2f}] ns",
                         args.lappd_channel, args.diode_channel, show_dt=True)

    # -- The steps of the rising-edge and constant-fraction timing, on the first selected events --
    step_events = stored_all[:MAX_STEP_EVENTS]
    print(f"Timing-steps PDF: {len(step_events)} events")
    save_timing_steps_pdf(step_events, os.path.join(OUTDIR, "waveforms_timing_steps.pdf"),
                          window_ns, baseline_mean, diode_window_ns, TAG,
                          args.lappd_channel, args.diode_channel,
                          args.edge_fraction, args.cfd_fraction, args.cfd_delay)