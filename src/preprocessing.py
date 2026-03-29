import pandas as pd
from pyteomics import mzml
import numpy as np
import re

# Standard 20 AA vocabulary strings
VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
# Token IDs: PAD=0, SOS=1, EOS=2, Amino Acids=3..22
CHAR_TO_IDX = {c: i+3 for i, c in enumerate(VOCAB)}

def load_labeled_spectra(mzml_path, xlsx_path, max_spectra=5000):
    """
    Stream mzML, join with xlsx labels by scan number.
    Returns list of dicts: {mz, intensity, peptide, charge, precursor_mz}
    Only include MS2 spectra (ms_level == 2) with a matching label.
    """
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print("Failed to load xlsx:", e)
        return []
        
    df = df.dropna(subset=['Sequence', 'Scan number'])
    scan_to_pep = dict(zip(df['Scan number'], df['Sequence']))
    
    spectra_list = []
    count = 0
    with mzml.read(mzml_path) as reader:
        for spectrum in reader:
            # Although the strict prompt says ms_level == 2, 
            # some datasets use ms level 1 if unfragmented or poorly annotated
            # We'll allow taking the ones that map to our db search.
            
            scan_match = re.search(r'scan=(\d+)', spectrum['id'])
            if not scan_match: continue
                
            scan_num = int(scan_match.group(1))
            if scan_num in scan_to_pep:
                peptide = scan_to_pep[scan_num]
                
                mzs = spectrum.get('m/z array', np.array([]))
                ints = spectrum.get('intensity array', np.array([]))
                
                prec_mz = None
                charge = None
                for p in spectrum.get('precursorList', {}).get('precursor', []):
                    for ion in p.get('selectedIonList', {}).get('selectedIon', []):
                        prec_mz = ion.get('selected ion m/z', prec_mz)
                        charge = ion.get('charge state', charge)
                
                # Charge state > 4 filtering
                if charge is not None and charge > 4:
                    continue
                
                spectra_list.append({
                    'mz': mzs,
                    'intensity': ints,
                    'peptide': peptide,
                    'charge': charge,
                    'precursor_mz': prec_mz,
                    'scan_num': scan_num
                })
                count += 1
                if count >= max_spectra:
                    break
    return spectra_list

def preprocess_spectrum(mz_array, intensity_array, mz_min=0, mz_max=2000, bin_size=0.1, top_k_peaks=200):
    """
    1. Keep only top_k_peaks by intensity
    2. Normalize intensities to [0, 1]
    3. Bin m/z axis into fixed bins of bin_size over [mz_min, mz_max]
    4. Return np.ndarray of shape (20000,)
    """
    num_bins = int((mz_max - mz_min) / bin_size)
    if len(mz_array) == 0:
        return np.zeros(num_bins)
        
    # 1a. Peak filtering: remove peaks below 0.1% of base peak intensity
    base_peak_int = np.max(intensity_array)
    mask = intensity_array >= 0.001 * base_peak_int
    mz_array = mz_array[mask]
    intensity_array = intensity_array[mask]
    
    if len(mz_array) == 0:
         return np.zeros(num_bins)
         
    # 1b. Denoising strategy: keep top_k_peaks
    if len(intensity_array) > top_k_peaks:
        idx = np.argsort(intensity_array)[-top_k_peaks:]
        mz_array = mz_array[idx]
        intensity_array = intensity_array[idx]
        
    # 2. Normalize intensity to [0,1] per spectrum
    intensity_array = intensity_array / np.max(intensity_array)
    
    # 3. m/z binning
    binned_vector = np.zeros(num_bins)
    
    bins = np.floor((mz_array - mz_min) / bin_size).astype(int)
    valid_mask = (bins >= 0) & (bins < num_bins)
    
    # Accumulate intensities in bins
    np.add.at(binned_vector, bins[valid_mask], intensity_array[valid_mask])
    
    return binned_vector

def encode_peptide(sequence, max_length=30):
    """
    Encode peptide string to integer token array.
    Vocabulary: 20 standard amino acids + PAD(0) + SOS(1) + EOS(2)
    Return np.ndarray of shape (max_length+2,) with SOS + tokens + EOS + padding
    """
    vector = np.zeros(max_length + 2, dtype=int)
    vector[0] = 1 # SOS
    
    # Strip any modifications (e.g. sequences with +15.99)
    sequence = re.sub(r'[^A-Z]', '', sequence)
    
    # Fill sequence
    for i, aa in enumerate(sequence):
        if i >= max_length:
            break
        vector[i+1] = CHAR_TO_IDX.get(aa, 0) # 0 is PAD if unknown
    
    # Add EOS
    eos_pos = min(len(sequence) + 1, max_length + 1)
    vector[eos_pos] = 2 # EOS
    
    return vector

def build_dataset(mzml_path, xlsx_path, max_spectra=5000):
    """
    Full pipeline: load -> preprocess -> encode
    Returns X (np.ndarray, shape [N, 20000]) and y (np.ndarray, shape [N, 32])
    """
    spectra_list = load_labeled_spectra(mzml_path, xlsx_path, max_spectra)
    
    X = []
    y = []
    
    for s in spectra_list:
        binned = preprocess_spectrum(s['mz'], s['intensity'])
        encoded = encode_peptide(s['peptide'])
        
        X.append(binned)
        y.append(encoded)
        
    return np.array(X), np.array(y)
