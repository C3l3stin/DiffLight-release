import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_traffic_demand(json_filepath, bin_size_seconds=60):
    """
    Analyse le fichier de flux et trace la demande (véhicules générés) au cours du temps.
    """
    print(f"Chargement des données depuis {json_filepath}...")
    with open(json_filepath, 'r') as f:
        traffic_data = json.load(f)
    
    # Extraction des temps de départ
    start_times = [item['startTime'] for item in traffic_data]
    total_vehicles = len(start_times)
    
    if total_vehicles == 0:
        print("Aucun véhicule trouvé dans le fichier.")
        return

    max_time = max(start_times)
    print(f"Analyse de {total_vehicles} véhicules sur une durée de {max_time} secondes.")
    
    # Création des intervalles (bins)
    bins = np.arange(0, max_time + bin_size_seconds, bin_size_seconds)
    
    # Comptage des véhicules par intervalle
    counts, bin_edges = np.histogram(start_times, bins=bins)
    
    # Tracé du graphique
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Tracer la courbe de la demande
    plt.plot(bin_edges[:-1], counts, color='#1f77b4', linewidth=2, label='Véhicules insérés')
    plt.fill_between(bin_edges[:-1], counts, color='#1f77b4', alpha=0.3)
    
    # Calcul d'une moyenne mobile pour lisser la courbe et bien voir la multimodalité
    window = 5
    if len(counts) >= window:
        moving_avg = np.convolve(counts, np.ones(window)/window, mode='same')
        # Ajustement de l'axe X pour la moyenne mobile
        x_moving_avg = bin_edges[window//2 : -window//2]
        # S'assurer que les dimensions correspondent
        min_len = min(len(x_moving_avg), len(moving_avg))
        plt.plot(x_moving_avg[:min_len], moving_avg[:min_len], 
                 color='#ff7f0e', linestyle='--', linewidth=2.5, label=f'Moyenne mobile ({window} bins)')

    plt.title(f"Profil de Demande du Trafic (Multimodality) - Bins de {bin_size_seconds}s", fontsize=14, pad=15)
    plt.xlabel("Temps de simulation (secondes)", fontsize=12)
    plt.ylabel(f"Nombre de véhicules générés", fontsize=12)
    plt.legend(fontsize=12)
    plt.xlim(0, max_time)
    
    plt.tight_layout()
    plt.savefig("traffic_demand_profile.png", dpi=300)
    print("Graphique sauvegardé sous 'traffic_demand_profile.png'.")
    plt.show()

def plot_cumulative_traffic_demand(json_filepath):
    """
    Analyse le fichier de flux et trace la demande cumulée 
    (somme des véhicules insérés) au cours du temps.
    """
    print(f"Chargement des données depuis {json_filepath}...")
    try:
        with open(json_filepath, 'r') as f:
            traffic_data = json.load(f)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {json_filepath} est introuvable.")
        return

    # Extraction et tri des temps de départ (startTime)
    # Le tri est crucial pour calculer la somme cumulée correctement
    start_times = [item['startTime'] for item in traffic_data]
    start_times.sort()
    
    total_vehicles = len(start_times)
    
    if total_vehicles == 0:
        print("Aucun véhicule trouvé dans le fichier.")
        return

    max_time = start_times[-1]
    print(f"Analyse de la demande cumulée pour {total_vehicles} véhicules sur {max_time} secondes.")

    # Calcul de la somme cumulée
    # A chaque instant 't' dans start_times, le nombre total de véhicules entrés augmente de 1.
    cumulative_counts = np.arange(1, total_vehicles + 1)

    # Configuration du graphique (style académique)
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Tracé de la courbe en escalier (step plot) 
    # C'est plus précis pour des événements discrets comme l'insertion de véhicules
    plt.step(start_times, cumulative_counts, where='post', color='#d62728', linewidth=2, label='Insertions cumulées')
    
    # Remplissage sous la courbe (optionnel, pour l'esthétique)
    plt.fill_between(start_times, cumulative_counts, step='post', color='#d62728', alpha=0.15)

    # Ajout de repères visuels (optionnel)
    # Par exemple, souligner le moment où la moitié des véhicules sont entrés
    half_demand_index = next(i for i, v in enumerate(cumulative_counts) if v >= total_vehicles / 2)
    half_demand_time = start_times[half_demand_index]
    
    plt.axhline(y=total_vehicles/2, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=half_demand_time, color='gray', linestyle='--', alpha=0.7)
    plt.text(half_demand_time + (max_time * 0.02), total_vehicles/2 - (total_vehicles * 0.05), 
             f'50% de la demande atteinte à t={half_demand_time}s', color='dimgray', fontsize=10)

    # Esthétique et labels
    plt.title("Demande Cumulée : Somme des insertions de véhicules", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Temps de simulation (secondes)", fontsize=12)
    plt.ylabel("Nombre total de véhicules insérés", fontsize=12)
    plt.xlim(0, max_time * 1.05) # Un peu de marge à droite
    plt.ylim(0, total_vehicles * 1.05)
    plt.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Sauvegarde et affichage
    output_filename = "cumulative_demand_profile.png"
    plt.savefig(output_filename)
    print(f"Graphique sauvegardé sous '{output_filename}'.")
    plt.show()



if __name__ == "__main__":
    # Remplace par le nom exact de ton fichier
    file_name = "data/Jinan/3_4/anon_3_4_jinan_real.json" 
    # Agrégation par tranches de 60 secondes (1 minute)
    plot_traffic_demand(file_name, bin_size_seconds=60)
    plot_cumulative_traffic_demand(file_name)