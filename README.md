# LAPPDANA

Python scripts to analyze LAPPD data recorded with a DRS4 evaluation board (DRSOSC XML output). Events are streamed one at a time, so large files can be processed without loading them into memory.

## Requirements

- Python 3
- numpy, scipy, matplotlib

## Scripts

| Script | Purpose |
| --- | --- |
| `waveformana.py` | Pedestal calibration, single-channel waveform analysis: amplitude and charge distributions, afterpulses, purity estimate |
| `gain_studies.py` | Gain estimate from the integrated charge of each pulse (Gen-II LAPPD paper, Shin et al., Sec. 3.3, Eq. 3.5) |
| `timeresana.py` | Two-channel analysis (LAPPD and fast photodiode): 1PE selection, amplitude and charge, and timing (Delta t) |

## Workflow

1. Take a pedestal run with the laser blocked and compute the calibration for the LAPPD:

   ```
   python waveformana.py --pedestal --lappd 189 pedestal_run.xml
   ```

   This writes `pedestal_lappd_189.txt` (baseline and RMS noise). The other scripts read it and use 5 x RMS noise as the hit threshold.

2. Run the analysis on a data file:

   ```
   python waveformana.py run.xml --lappd 189 --channel CHN1 --window 30 40
   python gain_studies.py run.xml --lappd 189 --channel CHN1 --window 30 40
   python timeresana.py run.xml --lappd 189 --lappd-channel CHN1 --diode-channel CHN2 --window 10 20
   ```

Run any script with `--help` for the full list of options.

## Analysis notes

- The LAPPD pulse is positive. The negative undershoot that follows is treated as an electronics artifact, so calculations use only the baseline-subtracted signal clipped at zero. The pedestal calibration is not clipped.
- Gain: the baseline-subtracted voltage is integrated over the pulse's positive lobe, divided by 50 ohm to get charge, then divided by the elementary charge.
- Timing: Delta t is the LAPPD pulse time minus the photodiode pulse time. Pulse times can be taken from the centroid, the rising edge, or a constant-fraction discriminator (`--edge-fraction`, `--cfd-fraction`, `--cfd-delay`).

## Output

Each script writes its histograms (`.png`) and waveform plots (`.pdf`) to a directory named after the input file or, for `gain_studies.py`, under `gain/`. Data and output file types (`.xml`, `.root`, `.png`, `.pdf`, `.txt`, `.npz`) are listed in `.gitignore` and are not tracked.
