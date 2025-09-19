"""
Módulo para procesamiento de archivos FITS
"""

from astropy.io import fits
import numpy as np

def read_fits_file(file_path):
    """Lee un archivo FITS y extrae los datos espectrales"""
    try:
        with fits.open(file_path) as hdul:
            print("Extensiones disponibles en el archivo FITS:")
            for i, hdu in enumerate(hdul):
                print(f"{i}: {hdu.name} - {type(hdu)}")
            
            # Buscar la extensión correcta
            data = None
            for ext_name in ["COADD_B", "COADD_R", "FLUX", "COADD"]:
                if ext_name in hdul:
                    data = hdul[ext_name].data
                    print(f"Usando extensión: {ext_name}")
                    break
            
            if data is None:
                # Usar la primera extensión con datos
                for i, hdu in enumerate(hdul):
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        data = hdu.data
                        print(f"Usando extensión {i} por defecto")
                        break
            
            if data is None:
                raise ValueError("No se encuentra extensión válida en el archivo FITS")
            
            # Buscar las columnas correctas
            if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
                print("Columnas disponibles:", data.dtype.names)
                
                if "WAVELENGTH" in data.dtype.names:
                    wl = np.array(data["WAVELENGTH"][0], dtype=float)
                    flux = np.array(data["FLUX"][0], dtype=float)
                    ivar = np.array(data["IVAR"][0], dtype=float)
                elif "wavelength" in data.dtype.names:
                    wl = np.array(data["wavelength"][0], dtype=float)
                    flux = np.array(data["flux"][0], dtype=float)
                    ivar = np.array(data["ivar"][0], dtype=float)
                elif "lambda" in data.dtype.names:
                    wl = np.array(data["lambda"][0], dtype=float)
                    flux = np.array(data["flux"][0], dtype=float)
                    ivar = np.array(data["ivar"][0], dtype=float)
                else:
                    # Intentar usar las primeras tres columnas
                    wl = np.array(data[0][0], dtype=float)
                    flux = np.array(data[0][1], dtype=float)
                    ivar = np.array(data[0][2], dtype=float)
            else:
                # Para datos en formato diferente
                wl = np.array(data[0][0], dtype=float)
                flux = np.array(data[0][1], dtype=float)
                ivar = np.array(data[0][2], dtype=float)
                
            return wl, flux, ivar
                
    except Exception as e:
        print(f"Error leyendo archivo FITS: {e}")
        print("Revisa la estructura del archivo FITS.")
        raise

def valid_mask(flux, ivar):
    """Crea una máscara para valores válidos"""
    m = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
    return m

def rebin_spectrum(wl, flux, ivar, factor=2):
    """Rebinea el espectro para mejorar SNR"""
    if factor <= 1:
        return wl, flux, ivar
    n = len(wl) // factor
    wl_r = wl[:n*factor].reshape(n, factor).mean(axis=1)
    var = 1.0 / ivar
    var_r = var[:n*factor].reshape(n, factor).mean(axis=1)
    flux_r = flux[:n*factor].reshape(n, factor).mean(axis=1)
    ivar_r = 1.0 / var_r
    return wl_r, flux_r, ivar_r
def read_fits_file(file_path):
    try:
        with fits.open(file_path) as hdul:
            print("Extensiones disponibles en el archivo FITS:")
            for i, hdu in enumerate(hdul):
                print(f"{i}: {hdu.name} - {type(hdu)}")
            
            # Buscar la extensión correcta
            data = None
            for ext_name in ["COADD_B", "COADD_R", "FLUX", "COADD"]:
                if ext_name in hdul:
                    data = hdul[ext_name].data
                    print(f"Usando extensión: {ext_name}")
                    break
            
            if data is None:
                # Usar la primera extensión con datos
                for i, hdu in enumerate(hdul):
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        data = hdu.data
                        print(f"Usando extensión {i} por defecto")
                        break
            
            if data is None:
                raise ValueError("No se encuentra extensión válida en el archivo FITS")
            
            # Buscar las columnas correctas
            if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
                print("Columnas disponibles:", data.dtype.names)
                
                if "WAVELENGTH" in data.dtype.names and "FLUX" in data.dtype.names and "IVAR" in data.dtype.names:
                    wl = np.array(data["WAVELENGTH"][0], dtype=float)
                    flux = np.array(data["FLUX"][0], dtype=float)
                    ivar = np.array(data["IVAR"][0], dtype=float)
                elif "wavelength" in data.dtype.names and "flux" in data.dtype.names and "ivar" in data.dtype.names:
                    wl = np.array(data["wavelength"][0], dtype=float)
                    flux = np.array(data["flux"][0], dtype=float)
                    ivar = np.array(data["ivar"][0], dtype=float)
                elif "lambda" in data.dtype.names and "flux" in data.dtype.names and "ivar" in data.dtype.names:
                    wl = np.array(data["lambda"][0], dtype=float)
                    flux = np.array(data["flux"][0], dtype=float)
                    ivar = np.array(data["ivar"][0], dtype=float)
                else:
                    # Intentar usar las primeras tres columnas
                    wl = np.array(data[0][0], dtype=float)
                    flux = np.array(data[0][1], dtype=float)
                    ivar = np.array(data[0][2], dtype=float)
            else:
                # Para arrays simples
                wl = np.array(data[0][0], dtype=float)
                flux = np.array(data[0][1], dtype=float)
                ivar = np.array(data[0][2], dtype=float)
            
            # 🔍 NUEVO: Verificar si los datos están vacíos o son todos cero
            print("\n=== VERIFICACIÓN DE CALIDAD DE DATOS ===")
            print(f"Longitud de onda: {len(wl)} puntos")
            print(f"Flujo - Min: {np.nanmin(flux):.6f}, Max: {np.nanmax(flux):.6f}")
            print(f"IVAR - Min: {np.nanmin(ivar):.6f}, Max: {np.nanmax(ivar):.6f}")
            
            if np.all(flux == 0) or np.all(np.isnan(flux)):
                print("⚠️  ADVERTENCIA: Todos los valores de flujo son cero o NaN")
                
            if np.all(ivar == 0) or np.all(np.isnan(ivar)):
                print("⚠️  ADVERTENCIA: Todos los valores de IVAR son cero o NaN")
                print("   Esto hará que todos los puntos sean descartados")
                
            return wl, flux, ivar
                
    except Exception as e:
        print(f"Error leyendo archivo FITS: {e}")
        print("Revisa la estructura del archivo FITS.")
        raise