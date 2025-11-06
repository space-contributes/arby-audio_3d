# Arby Audio 3D — Cinematic Spatial Sound Engine - C++

### **Live 96 kHz / 32-bit Spatial Audio Conversion. GPU-Accelerated. Physically Accurate.**

##### **Arby Audio 3D** is a **next-generation, GPU-accelerated spatial sound engine** that delivers **live 96 kHz / 32-bit, ISO 9613-1–compliant, HRTF-accurate, multi-threaded, and privacy-safe 7.1.4 audio conversion** with **real-time reflections, sinc resampling, furniture-aware acoustics, and cross-platform AR/VR-ready performance** — all **optimized at the assembly level for true cinematic realism.**

*Made with ❤️ by Space-code* WITH *7 YEARS OF MAKING*!

![Version](https://img.shields.io/badge/version-3.2.1-blue) ![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen) ![Python Version](https://img.shields.io/badge/python-3.11-blue) ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![Last Commit](https://img.shields.io/github/last-commit/space-contributes/arby-audio_3d) ![Stars](https://img.shields.io/github/stars/space-contributes/arby-audio_3d?style=social) ![Forks](https://img.shields.io/github/forks/space-contributes/arby-audio_3d?style=social) ![Open Issues](https://img.shields.io/github/issues/space-contributes/arby-audio_3d) ![Closed Issues](https://img.shields.io/github/issues-closed/space-contributes/arby-audio_3d) ![Downloads](https://img.shields.io/github/downloads/space-contributes/arby-audio_3d/total) ![Maintenance](https://img.shields.io/badge/maintenance-active-brightgreen) ![Supported OS](https://img.shields.io/badge/os-windows%20|%20macOS%20|%20Linux-lightgrey) ![Top Language](https://img.shields.io/github/languages/top/space-contributes/arby-audio_3d) ![Repo Size](https://img.shields.io/github/repo-size/space-contributes/arby-audio_3d) ![Commits](https://img.shields.io/github/commit-activity/m/space-contributes/arby-audio_3d) ![Issues Closed](https://img.shields.io/github/issues-pr-closed/space-contributes/arby-audio_3d)
![Arby Audio Logo](https://raw.githubusercontent.com/space-contributes/arby-audio_3d/refs/heads/main/Arby%20Logo%20Design%20Proto.1\(1\).jpg)

Arby Audio is a next-generation 3D spatial sound engine designed for **live, real-time, and file-based audio and video conversion**. It's engineered from the ground up for **precision, performance, and realism** — using **sinc resampling, ISO 9613-1–compliant attenuation**, and **HRTF-based spatial rendering** that simulates how sound truly behaves in the real world.
And yes — it sounds **AMAZING.**

---

## Setup & Usage

### Download the Executable

You can **download the latest release** (recommended) — or clone the repository manually.

```bash
git clone https://github.com/space-code/arby-audio_3d.git
cd arby-audio_3d
```

Or download the files individually from GitHub.

### 🪟 Windows Live Audio Conversion

The Windows `.exe` allows you to **convert live audio** directly into Arby Audio's spatial format.

> ⚠️ Make sure your playback device is set to **96 kHz, 24-bit (or 32-bit if supported)** in your Windows Sound Control Panel.
> Place all `.dll` files in the same folder as the `.exe`.

### 🐍 Python Version

Run the Python version to process an audio file:

```bash
python "arby_audio.py" --music_url https://your-music-url.com/file.wav
```

### 🌐 HTML / Web Version

Open the HTML file in your browser — or visit the hosted version on GitHub Pages.

* Works **offline**
* Compatible with **Windows, macOS, Linux, Android, and iOS**
* 100% **local processing**, **no servers**, and **GDPR compliant**

---

## 🧩 Features

* **Sinc-based resampling** up to 96 kHz / 32-bit for unmatched clarity.
* **HRTF spatialization** with full 360° azimuth and elevation coverage.
* **Real-time reflections** up to 3rd order with per-wall frequency damping.
* **Distance- and frequency-dependent air absorption** (ISO 9613-1).
* **Automatic object hardness and absorption adaptation.**
* **Furniture detection** for improved realism — no sensors required.
* **Multi-threaded (up to 8 workers)** for parallel sound reflections.
* **GPU-mapped memory access** for maximum efficiency.
* **Multi-platform support** (Windows, macOS, Linux; live conversion in progress for macOS/Linux).
* **No SDKs, no dependencies — just pure performance.**
* **AR/VR compatibility and game-ready architecture.**

---

## 💥 Why Arby Audio Is Better Than Typical Audio Engines

Most audio frameworks rely on middleware layers, SDKs, and abstraction — which introduce **latency**, **limited control**, and **non-optimized paths**. Arby Audio takes a completely different approach: **pure, physics-accurate sound processing**, coded directly at the **assembly level** for ultimate performance and realism.

### 🚀 Performance

* **Every instruction hand-optimized in pure assembly**, outperforming traditional compiled languages.
* **Direct GPU-mapped memory access** — zero driver overhead, no context switching.
* **8-thread worker pool** for real-time reflection modeling and sinc-based upsampling at 96 kHz / 32-bit precision.

### 🎧 Acoustic Realism

* **Fully compliant with ISO 9613-1** for air absorption and distance-based attenuation.
* **True HRTF-based spatial simulation** with full azimuth and elevation coverage (complete 360° + vertical).
* **3rd-order reflection modeling**, dynamic room scaling, and frequency-dependent energy loss simulation.
* **Automatic object hardness and absorption detection** — surfaces react naturally to sound.
* **Furniture detection and adaptive reflection logic** — optimized for realistic spaces without sensors.

### 🌍 Multi-Platform & Privacy-First

* Runs fully **offline** on **Windows, macOS, and Linux**, with web and mobile support via a standalone HTML engine.
* **No SDKs, no telemetry, no servers.**
* 100% **privacy- and compliance-safe**, with all processing handled locally.
* **Web version** uses on-device compute through WebAssembly and WebAudio for real-time rendering.

### 🎮 Developer-Friendly

* **Game-ready and VR/AR compatible** with parallel EXE support for multi-instance workflows.
* **Drop-in executable** for live conversion — no setup, no integration overhead.
* **Automatic resampling support** from 44.1 kHz to 96 kHz, with sinc-based filtering and FFT spectral smoothing.
* **Multi-channel 7.1.4 layout compatible** — with *true* 360° HRTF spatialization.

### 🧠 Why It Sounds Better

* Typical engines approximate reflections; Arby Audio **physically simulates** them.
* Typical engines use linear filters; Arby Audio applies **multi-band sinc resampling** and **FFT spectral weighting**.
* Typical engines pre-render effects; Arby Audio performs **live time-domain convolution** with frequency-dependent delay and attenuation.
* And most importantly — it simply sounds **incredible** 🔊

---

## 🧪 Development Status

✅ Audio/Video Conversion — Completed
✅ 96 kHz / 32-bit Sinc Resampling — Completed
✅ ISO 9613-1 Attenuation and Frequency Loss — Completed
✅ Multi-threaded Reflection Engine — Completed
✅ GPU Direct Optimization — Completed
🔄 Linux/macOS Live Conversion — In Progress
🔄 AR/VR SDK Support — Planned
🔄 Real-time Object Hardness Toggle — In Progress

---

## 🔐 Architecture Overview

* Written in **modern C++**, with **line-by-line assembly optimization**.
* Combines compiler optimizations from multiple sources and custom assembly inspection for maximum efficiency.
* **Dual-compatible techniques** ensure stability on both new and older systems.
* **Memory-safe** and **multi-threaded** by design.

---

## 📜 License

Open-source. Free to use, modify, and redistribute. All code executes locally and respects user privacy.
LICENSE.md only valid in Main Branch.
All references to brand names and trademarks are for educational and research purposes only.

---
