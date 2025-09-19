"""
Configuración por defecto de la aplicación
"""

# Parámetros de procesamiento
DEFAULT_PARAMS = {
    "REBIN_FACTOR": 4,
    "SG_WINDOW": 61,
    "SG_POLY": 2,
    "MOVING_AVG_WINDOW": 35,
    "DO_CONTINUUM_NORM": True,
    "SNR_WINDOW": 150,
    "CONTINUUM_WINDOW": 701,
    "CONTINUUM_PERCENTILE": 90,
    "REDSHIFT_SIGMA_CLIP": 2.0
}

# Líneas de interés para análisis
SPECTRAL_LINES = {
    "Hβ": 4861.3,
    "Mg I 5172": 5172.7,
    "Fe I 5328": 5328.0
}