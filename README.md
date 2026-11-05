# Ibiza Channel transport: satellite altimetry vs. gliders

Code for the paper *Satellite altimetry captures variability in the Ibiza Channel*.

## Contents

| File                    | Description                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `paper_figures.ipynb` | Rebuilds the manuscript figures (Figs. 1-4) and the Supporting Information figures (Figs. S1-S5).                                                       |
| `2vgeo_gliders.py`    | Computes the geostrophic velocity and transport from SOCIB glider L2 sections across the Ibiza Channel (39°N, 0.1-1.13°E). Output is used in Fig. S5. |

## Data

Paths in the code are absolute (`/media/hargous/MyPassport/OT_Ibiza/...`). Edit them to your local copy.

### European CMEMS products

| Product                                                         | Use                      | Link                                                             |
| --------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------- |
| SEALEVEL_EUR_PHY_L4_MY_008_068 (SAT_EUR, 1/16°)                | Main altimetry product   | [doi.org/10.48670/moi-00141](https://doi.org/10.48670/moi-00141)  |
| SEALEVEL_GLO_PHY_L4_MY_008_047 (SAT_GLO, 1/8°)                 | Global altimetry product | [doi.org/10.48670/moi-00148](https://doi.org/10.48670/moi-00148)  |
| SEALEVEL_GLO_PHY_L4_MY_008_057 (SAT_TWO, 1/4°, two satellites) | Two-satellite reference  | [doi.org/10.48670/moi-00145](https://doi.org/10.48670/moi-00145)  |

### SOCIB glider transport

| Dataset                                                                             | Use                           | Link                                                                                   |
| ----------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| Glider geostrophic transport estimates, Ibiza Channel 2011-2024 (Juza et al., 2025) | Reference glider transport    | [doi.org/10.25704/DSRT-TY44](https://doi.org/10.25704/DSRT-TY44)                        |
| SOCIB glider L2 sections (`canalesL2`)                                            | Input of `2vgeo_gliders.py` | [thredds.socib.es/thredds/catalog.html](https://thredds.socib.es/thredds/catalog.html)  |

### Model MedMFC data

| Dataset                                                         | Use             | Link                                                             |
| --------------------------------------------------------------- | --------------- | ---------------------------------------------------------------- |
| CMEMS MedMFC model (MEDSEA_MULTIYEAR_PHY_006_004): ADT, T, S, v | Figs. S2 and S4 | [doi.org/10.48670/mds-00375](https://doi.org/10.48670/mds-00375)  |

### Other

| Dataset            | Use                    | Link                                                                                |
| ------------------ | ---------------------- | ----------------------------------------------------------------------------------- |
| SWOT L3 (pass 348) | Fig. 4 and Fig. S2     | [doi.org/10.24400/527896/A01-2023.018](https://doi.org/10.24400/527896/A01-2023.018) |
| GEBCO bathymetry   | Maps and section depth | [download.gebco.net](https://download.gebco.net/)                                    |

## Requirements

Python with `numpy`, `pandas`, `xarray`, `scipy`, `astropy`, `gsw`, `pyproj`, `matplotlib`, `cartopy`.

## Usage

1. Download the datasets above and set the paths in the code.
2. Run `2vgeo_gliders.py` to produce the glider transport CSV used in Fig. S5.
3. Run `paper_figures.ipynb`.
