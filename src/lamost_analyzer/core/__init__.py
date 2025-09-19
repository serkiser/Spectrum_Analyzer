"""
Módulo core para procesamiento de espectros LAMOST
"""

from .fits_processor import read_fits_file, valid_mask, rebin_spectrum
from .utils import try_savgol, running_percentile, enhance_line_detection
from .spectral_analysis import generate_spectral_report

__all__ = [
    'read_fits_file',
    'valid_mask',
    'rebin_spectrum',
    'try_savgol',
    'running_percentile',
    'enhance_line_detection',
    'generate_spectral_report'
]