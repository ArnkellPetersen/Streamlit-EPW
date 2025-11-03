# Streamlit‑EPW — Climate Data Explorer

A fast,  Streamlit app for exploring **EPW weather files** and derived climate metrics for building simulation. The app is developed by the **NMBU Building Physics — Climate & Buildings** group as a test of the usability of streamlit for such purposes. 

---

## Highlights

- **EPW loader** with basic validation and metadata preview (location, timezone, elevation, sources).
- **Interactive views**: Map, Time Series, XY Scatter, Heatmaps, Windrose, Monthly summaries, Tables.
- **Stateful navigation** (no “jump back to first tab”) using a persistent segmented control / radio.
- **Smart caching** of heavy computations for responsive interaction.
- **Reproducible**: pinned dependencies and clean function boundaries for each view.

---

## Quick Start

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run EPW-App.py
# (or if you use the stateful-nav version)
streamlit run EPW-App_stateful_tabs.py
```

> **Tip:** Streamlit apps generally feel snappiest when run **locally** with caching enabled. Avoid re-reading large files on every interaction—cache loaders and heavy transforms.

---

## Repository Structure  

```
Streamlit-EPW/
├─ EPW-App.py                 # Main app (tabs version)
├─ requirements.txt           # Python dependencies
├─ Logos/                    # Logos, icons
└─ README.md
```

---

## Configuration

- **Python**: 3.13+ recommended
- **Key packages**: `streamlit`, `pandas`, `numpy`, `plotly>=5.20`, `altair` (optional)
- **Large files**: keep EPWs outside the repo or use Git LFS

---

## How to Cite

If you use this repository or derivatives in a publication, please cite the relevant Zenodo record or project publication. If none exist, cite: **“NMBU Building Physics — Climate & Buildings.”**

- **Zenodo community**: https://zenodo.org/communities/nmbu-buildingphysicsgroup/
- **Klimadata for bygninger**: https://www.klimadataforbygninger.no
- **Climate Data for Buildings**: https://www.climatedataforbuildings.eu


---

## Contributing

Issues and pull requests are welcome. Please follow standard scientific computing practices: clear documentation, reproducible code, and dataset provenance.

---

## License

Unless otherwise noted, code in this repository is released under an open‑source license (e.g., MIT or Apache‑2.0).  

---

*Questions?* Open an issue or contact the team.
