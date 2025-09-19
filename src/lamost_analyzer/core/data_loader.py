from astropy.io import fits

REQUIRED_KEYS = ["SIMPLE", "BITPIX", "NAXIS", "EXTEND"]

def validate_and_load_fits(filepath):
    """
    Valida y carga un archivo FITS, corrigiendo headers básicos si es necesario.
    
    Args:
        filepath (str): Ruta al archivo FITS
        
    Returns:
        tuple: (header, data) si es exitoso, (None, None) si hay error
    """
    try:
        with fits.open(filepath, mode='update') as hdul:
            header = hdul[0].header
            missing_keys = []
            
            # Validar y completar cabecera
            for key in REQUIRED_KEYS:
                if key not in header:
                    missing_keys.append(key)
                    if key == "SIMPLE":
                        header[key] = True
                    elif key == "BITPIX":
                        header[key] = 16
                    elif key == "NAXIS":
                        header[key] = 0
                    elif key == "EXTEND":
                        header[key] = True
            
            if missing_keys:
                print(f"[INFO] Corregidas keys faltantes: {missing_keys}")
                hdul.flush()
            
            # Verificar datos corruptos
            if data is not None:
                # Verificar valores extremos que indican corrupción
                if np.any(np.abs(data) > 1e20):  # Umbral para detectar valores absurdos
                    print(f"[WARNING] El archivo {filepath} contiene valores posiblemente corruptos")
                    # Opcional: puedes intentar reparar o rechazar el archivo
                    
            return header, data
            
    except Exception as e:
        print(f"[ERROR] Error procesando {filepath}: {str(e)}")
        return None, None