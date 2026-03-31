import numpy as np
import os
import json
from siamese_encoder import SiameseEncoder
import time

class BaseEmpreintes:
    """
    Base de données d'empreintes de signatures
    Permet d'ajouter des clients SANS ré-entraînement
    """
    
    def __init__(self, base_path='models/base_empreintes.npz'):
        self.base_path = base_path
        
        print("📥 Chargement de l'encodeur Siamese...")
        self.encoder = SiameseEncoder()
        
        if not self.encoder.is_trained:
            print("❌ ERREUR : Encodeur non entraîné !")
            print("   Lancez d'abord : python siamese_encoder.py")
            self.clients = None
            return
        
        self.clients = {}
        
        if os.path.exists(base_path):
            self._charger()
            print(f"✅ Base chargée : {len(self.clients)} clients")
        else:
            print("🆕 Nouvelle base créée")
    
    def _charger(self):
        """Charge la base d'empreintes depuis le fichier"""
        try:
            data = np.load(self.base_path, allow_pickle=True)
            self.clients = data['clients'].item()
        except Exception as e:
            print(f"⚠️  Erreur de chargement : {e}")
            self.clients = {}
    
    def _sauvegarder(self):
        """Sauvegarde la base d'empreintes"""
        np.savez(self.base_path, clients=self.clients)
    
    def ajouter_client(self, client_id, signatures_paths):
        """
        Ajoute un client INSTANTANÉMENT (2-5 secondes)
        
        Args:
            client_id: ID du client (ex: "56")
            signatures_paths: Liste des chemins vers ses signatures
        
        Returns:
            bool: True si succès, False sinon
        """
        if self.clients is None:
            print("❌ Base non initialisée")
            return False
        
        print(f"\n⚡ Ajout du client {client_id}...")
        print(f"   📂 {len(signatures_paths)} signatures à encoder")
        
        start = time.time()
        
        # Encoder toutes les signatures en une fois
        empreintes, valid_paths = self.encoder.encode_batch(signatures_paths)
        
        if empreintes is None or len(empreintes) == 0:
            print(f"   ❌ Échec de l'encodage")
            return False
        
        # Stocker dans la base
        self.clients[client_id] = {
            'empreintes': empreintes,
            'nb_signatures': len(empreintes),
            'date_ajout': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Sauvegarder
        self._sauvegarder()
        
        temps = time.time() - start
        
        print(f"   ✅ {len(empreintes)} empreintes créées")
        print(f"   💾 Base sauvegardée")
        print(f"   ⏱️  Temps : {temps:.2f} secondes")
        
        return True
    
    def identifier(self, signature_path, top_k=3, seuil_distance=0.45):
        """
        Identifie à qui appartient une signature
        
        Args:
            signature_path: Chemin vers la signature à identifier
            top_k: Nombre de prédictions à retourner
            seuil_distance: Seuil de distance (plus bas = plus similaire)
        
        Returns:
            dict: Résultats de l'identification
        """
        if self.clients is None:
            return {'error': 'Base non initialisée'}
        
        # Encoder la signature inconnue
        empreinte = self.encoder.encode(signature_path)
        
        if empreinte is None:
            return {'error': 'Erreur d\'encodage de l\'image'}
        
        # Comparer avec tous les clients
        resultats = []
        
        for client_id, data in self.clients.items():
            empreintes_client = data['empreintes']
            
            # Calculer la distance avec chaque signature du client
            distances = np.linalg.norm(empreintes_client - empreinte, axis=1)
            
            dist_min = float(np.min(distances))
            dist_moy = float(np.mean(distances))
            
            # Convertir en score de similarité (0-1, plus haut = plus similaire)
            similarite = 1 / (1 + dist_min)
            
            resultats.append({
                'client_id': client_id,
                'distance': dist_min,
                'distance_moyenne': dist_moy,
                'similarite': similarite,
                'confiance_pct': similarite * 100
            })
        
        # Trier par distance (plus petite = plus similaire)
        resultats.sort(key=lambda x: x['distance'])
        
        top = resultats[:top_k]
        best = top[0]
        
        # Décision basée sur la distance
        if best['distance'] < seuil_distance:
            status = 'identifie'
        elif best['distance'] < seuil_distance * 1.5:
            status = 'incertain'
        else:
            status = 'inconnu'
        
        return {
            'status': status,
            'best_match': best,
            'top_predictions': top,
            'seuil_utilise': seuil_distance
        }
    
    def statistiques(self):
        """Affiche les statistiques de la base"""
        if self.clients is None:
            print("❌ Base non initialisée")
            return
        
        print("\n" + "="*70)
        print("📊 STATISTIQUES DE LA BASE D'EMPREINTES")
        print("="*70)
        
        nb_clients = len(self.clients)
        nb_total_signatures = sum(data['nb_signatures'] for data in self.clients.values())
        
        print(f"\n📋 Général :")
        print(f"   • Nombre de clients      : {nb_clients}")
        print(f"   • Total de signatures    : {nb_total_signatures}")
        print(f"   • Moyenne par client     : {nb_total_signatures/nb_clients:.1f}")
        
        # Liste des clients
        print(f"\n👥 Clients enregistrés :")
        clients_sorted = sorted(self.clients.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        
        for i, client_id in enumerate(clients_sorted[:10], 1):
            data = self.clients[client_id]
            print(f"   {i:2d}. Client {client_id:3s} : {data['nb_signatures']} signatures")
        
        if len(clients_sorted) > 10:
            print(f"   ... et {len(clients_sorted) - 10} autres clients")
        
        print("="*70)


# ============================================================================
# UTILISATION STANDALONE
# ============================================================================

if __name__ == "__main__":
    import glob
    
    print("="*70)
    print("🗄️  GESTION DE LA BASE D'EMPREINTES")
    print("="*70)
    
    # Créer/charger la base
    base = BaseEmpreintes()
    
    if base.clients is None:
        print("\n❌ Impossible de continuer sans encodeur entraîné")
        exit(1)
    
    # Initialiser avec les clients existants si la base est vide
    if len(base.clients) == 0:
        print("\n📂 INITIALISATION DE LA BASE AVEC LES CLIENTS EXISTANTS")
        print("="*70)
        
        dossier = "data/real"
        
        if not os.path.exists(dossier):
            print(f"❌ Dossier introuvable : {dossier}")
            exit(1)
        
        # Grouper les signatures par client
        clients = {}
        for f in os.listdir(dossier):
            if f.startswith('original_') and f.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    client_id = f.split('_')[1]
                    if client_id not in clients:
                        clients[client_id] = []
                    clients[client_id].append(os.path.join(dossier, f))
                except:
                    pass
        
        print(f"\n✅ Trouvé {len(clients)} clients à initialiser")
        
        # Ajouter chaque client
        for idx, (client_id, sigs) in enumerate(sorted(clients.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0), 1):
            print(f"\n[{idx}/{len(clients)}]", end=" ")
            base.ajouter_client(client_id, sigs)
        
        print("\n" + "="*70)
        print("✅ INITIALISATION TERMINÉE !")
        print("="*70)
    
    # Afficher les statistiques
    base.statistiques()
    
    # Test d'identification
    print("\n" + "="*70)
    print("🧪 TEST D'IDENTIFICATION")
    print("="*70)
    
    test_sig = "data/real/original_1_5.png"
    if os.path.exists(test_sig):
        print(f"\n📄 Signature de test : {os.path.basename(test_sig)}")
        
        result = base.identifier(test_sig)
        
        if 'error' not in result:
            print(f"\n🔍 Résultat :")
            print(f"   Status : {result['status'].upper()}")
            print(f"\n   🏆 Meilleure correspondance :")
            best = result['best_match']
            print(f"      • Client      : {best['client_id']}")
            print(f"      • Distance    : {best['distance']:.4f}")
            print(f"      • Confiance   : {best['confiance_pct']:.1f}%")
            
            print(f"\n   📊 Top 3 :")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"      {i}. Client {pred['client_id']:3s} : {pred['confiance_pct']:.1f}% (dist: {pred['distance']:.4f})")
        else:
            print(f"\n❌ Erreur : {result['error']}")
    else:
        print(f"\n⚠️  Fichier de test introuvable : {test_sig}")
    
    print("\n" + "="*70)
    print("💡 Prochaine étape : python add_client_instant.py <client_id>")
    print("="*70)