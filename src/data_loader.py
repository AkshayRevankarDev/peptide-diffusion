from pyteomics import mzml
import numpy as np

# Re-export key functions from preprocessing
from .preprocessing import load_labeled_spectra, build_dataset

def load_raw_spectra(mzml_path, max_spectra=5000):
    """
    Stream mzML and return a list of dicts without requiring xlsx labels.
    Returns: list of {scan_num, mz, intensity, charge, precursor_mz}
    """
    import re
    
    spectra_list = []
    count = 0
    with mzml.read(mzml_path) as reader:
        for spectrum in reader:
            scan_match = re.search(r'scan=(\d+)', spectrum.get('id', ''))
            if not scan_match:
                continue
            scan_num = int(scan_match.group(1))
            
            mzs = spectrum.get('m/z array', np.array([]))
            ints = spectrum.get('intensity array', np.array([]))
            
            prec_mz = None
            charge = None
            for p in spectrum.get('precursorList', {}).get('precursor', []):
                for ion in p.get('selectedIonList', {}).get('selectedIon', []):
                    prec_mz = ion.get('selected ion m/z', prec_mz)
                    charge = ion.get('charge state', charge)
            
            spectra_list.append({
                'scan_num': scan_num,
                'mz': mzs,
                'intensity': ints,
                'charge': charge,
                'precursor_mz': prec_mz
            })
            
            count += 1
            if count >= max_spectra:
                break
                
    return spectra_list
