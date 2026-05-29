# Quantitative measurement of fluid inertial effects in confined Brownian motion

## Dependencies
Alongside the usual `numpy`, `scipy`, `matplotlib`, you would need the publicly available `lmfit` and `jpkfilè` libraries and the custom `TNC.py` library (given in the repository).

## Experimental data
The data used to run the code are available on the following link : https://drive.proton.me/urls/XC3940DP48#Wwupyn2PsxmS

## analysis.ipynb
The data are extracted from the two differents types of files retrieved from the AFM.
The notebook handles the calibration of the system, determination of the precise distance for the "thermal-curves" files (`.jpk-force`).
It then performs the spectral analysis on the traces, fit the PSD with the theory presented in the paper, and finally presents the results.
