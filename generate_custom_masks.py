import os
import pickle
import numpy as np
import random

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_NAME = "anon_4_4_hangzhou_real"
NUM_INTERSECTIONS = 16
NUM_STEPS = 3600  # Durée de la simulation (RUN_COUNTS)
HORIZON = 9       # Horizon configuré dans DiffLight
# Les nouveaux taux qu'on veut créer (en nombre d'intersections sur 16)
RATES = {
    "km_50": 8,   # 50% = 8 intersections sur 16
    "km_75": 12   # 75% = 12 intersections sur 16
}

# ==========================================
# GÉNÉRATION
# ==========================================
# Le dossier de destination (celui que tu as montré dans ton image)
dest_dir = "memory/fourphase/km"
os.makedirs(dest_dir, exist_ok=True)

# DiffLight attend un dictionnaire de la forme : 
# {"km_8": array_numpy_du_masque, "km_12": ...}
# Mais il est souvent plus simple de générer un fichier séparé par masque.

for rate_name, num_missing in RATES.items():
    print(f"Génération du masque pour {rate_name} ({num_missing}/{NUM_INTERSECTIONS} intersections en panne)...")
    
    # 1. On choisit quelles intersections seront en panne (True = en panne, False = marche)
    missing_intersections = [False] * NUM_INTERSECTIONS
    indices_en_panne = random.sample(range(NUM_INTERSECTIONS), num_missing)
    for idx in indices_en_panne:
        missing_intersections[idx] = True

    # 2. On construit le masque complet (1 = ok, 0 = panne)
    # Shape attendue par DiffLight (souvent) : [HORIZON, NUM_INTERSECTIONS, NUM_STEPS]
    # Ou [NUM_INTERSECTIONS, NUM_STEPS] selon comment ils ont structuré leur code.
    # D'après `model_test.py` ligne 13, ils font un transpose(1, 0, 2), la shape originelle est bizarre.
    
    # Pour être sûr du format, le plus sûr est de charger un masque existant et de copier sa "shape".
    try:
        with open(f"{dest_dir}/anon_4_4_hangzhou_real_km.pkl", "rb") as f: # Essaie de trouver un fichier de référence
            ref_data = pickle.load(f)
            ref_key = list(ref_data.keys())[0]
            ref_shape = np.array(ref_data[ref_key]).shape
            print(f"Shape de référence trouvée : {ref_shape}")
    except:
        # Si on n'a pas de référence, on crée un format générique (1, NUM_INTERSECTIONS, NUM_STEPS)
        # qui est compatible avec le [0, :, :horizon] de model_test.py
        ref_shape = (1, NUM_INTERSECTIONS, NUM_STEPS + 100) # On ajoute 100 de marge
        
    mask_array = np.ones(ref_shape, dtype=int)
    
    # On applique les pannes
    for inter_idx in range(NUM_INTERSECTIONS):
        if missing_intersections[inter_idx]:
            # On met toute l'intersection à 0
            mask_array[:, inter_idx, :] = 0
            
    # 3. On sauvegarde au format DiffLight
    # Le dictionnaire attendu a une clé bizarre : "%s_%s" % (pattern, str(rate))
    # Ex: "km_won_8"
    dict_to_save = {
        rate_name: mask_array
    }
    
    file_path = f"{dest_dir}/{DATASET_NAME}_{rate_name}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(dict_to_save, f)
        
    print(f"-> Fichier sauvegardé : {file_path}")

print("✅ Génération des masques personnalisés terminée !")
