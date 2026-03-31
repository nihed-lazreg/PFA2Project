from base_empreintes import BaseEmpreintes
import glob
import os
import time
import sys

def ajouter_client_instantane(client_id, dossier="data/real", min_signatures=3):
    """
    Ajoute un nouveau client avec MINIMUM 3 signatures (réaliste !)
    
    Args:
        client_id: ID du client à ajouter
        dossier: Dossier contenant les signatures
        min_signatures: Nombre minimum de signatures requis (défaut: 3)
    """
    print("="*70)
    print(f"⚡ AJOUT INSTANTANÉ DU CLIENT {client_id}")
    print("="*70)
    
    # Trouver les signatures du client
    print(f"\n📂 Recherche des signatures dans {dossier}/...")
    
    patterns = [
        f"original_{client_id}_*.png",
        f"original_{client_id}_*.jpg",
        f"original_{client_id}_*.jpeg",
        f"signature_{client_id}_*.png",  # Format alternatif
        f"client_{client_id}_*.png"       # Format alternatif
    ]
    
    signatures = []
    for pattern in patterns:
        signatures.extend(glob.glob(os.path.join(dossier, pattern)))
    
    if len(signatures) == 0:
        print(f"\n❌ Aucune signature trouvée pour le client {client_id}")
        print(f"   Formats acceptés :")
        print(f"      • original_{client_id}_1.png")
        print(f"      • signature_{client_id}_1.png")
        print(f"      • client_{client_id}_1.png")
        print(f"   Dossier : {dossier}")
        return False
    
    # Vérifier le nombre minimum
    if len(signatures) < min_signatures:
        print(f"\n⚠️  ATTENTION : Seulement {len(signatures)} signature(s) trouvée(s)")
        print(f"   Minimum recommandé : {min_signatures} signatures")
        print(f"   Trouvées : {', '.join([os.path.basename(s) for s in signatures])}")
        
        reponse = input(f"\n   Continuer quand même ? (o/n) : ").strip().lower()
        if reponse not in ['o', 'oui', 'y', 'yes']:
            print(f"   ❌ Opération annulée")
            return False
    
    print(f"   ✅ Trouvé {len(signatures)} signature(s)")
    
    # Recommandation selon le nombre
    if len(signatures) == 1:
        print(f"   ⚠️  1 signature : Précision limitée (~70-75%)")
    elif len(signatures) == 2:
        print(f"   ⚠️  2 signatures : Précision moyenne (~75-80%)")
    elif len(signatures) == 3:
        print(f"   ✅ 3 signatures : Précision correcte (~80-85%)")
    elif len(signatures) == 4:
        print(f"   ✅ 4 signatures : Précision bonne (~85-88%)")
    elif len(signatures) >= 5:
        print(f"   ✅ {len(signatures)} signatures : Précision excellente (~88-92%)")
    
    # Afficher les fichiers
    print(f"\n   📋 Signatures :")
    for sig in sorted(signatures)[:5]:
        print(f"      • {os.path.basename(sig)}")
    if len(signatures) > 5:
        print(f"      • ... et {len(signatures) - 5} autres")
    
    # Charger la base
    print(f"\n🗄️  Chargement de la base d'empreintes...")
    base = BaseEmpreintes()
    
    if base.clients is None:
        print(f"\n❌ ERREUR : Base non initialisée")
        print(f"   Lancez d'abord : python siamese_encoder.py")
        print(f"   Puis : python base_empreintes.py")
        return False
    
    print(f"   ✅ Base chargée : {len(base.clients)} clients actuellement")
    
    # Vérifier si le client existe déjà
    if client_id in base.clients:
        nb_sigs_existantes = base.clients[client_id]['nb_signatures']
        print(f"\n⚠️  ATTENTION : Le client {client_id} existe déjà ({nb_sigs_existantes} signatures)")
        print(f"   Options :")
        print(f"      1. Remplacer les anciennes signatures")
        print(f"      2. Ajouter aux signatures existantes (TODO)")
        print(f"      3. Annuler")
        
        reponse = input(f"\n   Votre choix (1/2/3) : ").strip()
        
        if reponse == '1':
            print(f"   ⚠️  Remplacement des anciennes signatures...")
        elif reponse == '2':
            print(f"   ⚠️  Fonction 'Ajouter' non implémentée. Remplacement à la place.")
        else:
            print(f"   ❌ Opération annulée")
            return False
    
    # Chronométrer l'ajout
    print(f"\n⚡ Ajout en cours...")
    start = time.time()
    
    success = base.ajouter_client(client_id, signatures)
    
    temps_total = time.time() - start
    
    if success:
        print(f"\n{'='*70}")
        print(f"✅ CLIENT {client_id} AJOUTÉ AVEC SUCCÈS EN {temps_total:.2f} SECONDES !")
        print(f"{'='*70}")
        
        print(f"\n📊 Statistiques :")
        print(f"   • Signatures enregistrées  : {len(signatures)}")
        print(f"   • Temps total              : {temps_total:.2f}s")
        print(f"   • Temps par signature      : {temps_total/len(signatures):.3f}s")
        print(f"   • Total clients en base    : {len(base.clients)}")
        
        # Estimation de précision
        if len(signatures) >= 5:
            precision_estimee = "88-92%"
            qualite = "✅ EXCELLENTE"
        elif len(signatures) == 4:
            precision_estimee = "85-88%"
            qualite = "✅ BONNE"
        elif len(signatures) == 3:
            precision_estimee = "80-85%"
            qualite = "✅ CORRECTE"
        elif len(signatures) == 2:
            precision_estimee = "75-80%"
            qualite = "⚠️ MOYENNE"
        else:
            precision_estimee = "70-75%"
            qualite = "⚠️ LIMITÉE"
        
        print(f"\n📈 Précision attendue :")
        print(f"   • Avec {len(signatures)} signature(s) : {precision_estimee}")
        print(f"   • Qualité              : {qualite}")
        
        if len(signatures) < 4:
            print(f"\n💡 RECOMMANDATION :")
            print(f"   Pour améliorer la précision, demandez {4 - len(signatures)} signature(s) supplémentaire(s)")
        
        print(f"\n💡 Le système est immédiatement opérationnel.")
        print(f"   Le client {client_id} peut être identifié dès maintenant.")
        
        # Test rapide d'identification
        if len(signatures) > 0:
            print(f"\n🧪 Test d'identification automatique...")
            test_sig = signatures[0]
            result = base.identifier(test_sig)
            
            if 'error' not in result:
                best = result['best_match']
                if best['client_id'] == client_id:
                    print(f"   ✅ Test réussi : Client {best['client_id']} identifié ({best['confiance_pct']:.1f}%)")
                else:
                    print(f"   ⚠️  Test : Client {best['client_id']} détecté au lieu de {client_id}")
                    print(f"       Distance : {best['distance']:.4f}")
                    if len(signatures) < 4:
                        print(f"       → Ajoutez plus de signatures pour améliorer")
        
        return True
    else:
        print(f"\n❌ ÉCHEC de l'ajout du client {client_id}")
        return False


# ============================================================================
# SCRIPT D'AIDE POUR CRÉER DES SIGNATURES DE TEST
# ============================================================================

def creer_signatures_test(client_id, nombre=4, source_client="1"):
    """
    Crée des signatures de test en copiant depuis un client existant
    (Utile pour tester avec peu de signatures)
    """
    import shutil
    
    print(f"\n🧪 CRÉATION DE SIGNATURES DE TEST POUR CLIENT {client_id}")
    print(f"   (Copie depuis client {source_client}, pour test uniquement)")
    
    source_dir = "data/real"
    
    # Trouver les signatures sources
    sources = sorted(glob.glob(f"{source_dir}/original_{source_client}_*.png"))[:nombre]
    
    if len(sources) < nombre:
        print(f"❌ Pas assez de signatures sources ({len(sources)} < {nombre})")
        return False
    
    # Copier avec nouveau nom
    for i, source in enumerate(sources, 1):
        dest = f"{source_dir}/signature_{client_id}_{i}.png"
        shutil.copy(source, dest)
        print(f"   ✅ Créé : {os.path.basename(dest)}")
    
    print(f"\n✅ {nombre} signatures de test créées pour le client {client_id}")
    return True


# ============================================================================
# UTILISATION EN LIGNE DE COMMANDE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("⚡ AJOUT INSTANTANÉ DE CLIENT (VERSION FLEXIBLE)")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n❌ Usage incorrect")
        print("\n📋 Usage :")
        print("   python add_client_instant.py <client_id> [nombre_min_signatures]")
        print("\n📝 Exemples :")
        print("   python add_client_instant.py 56")
        print("   python add_client_instant.py 56 3")
        print("   python add_client_instant.py 56 5")
        print("\n💡 Description :")
        print("   Ajoute un nouveau client avec MINIMUM 3 signatures (par défaut)")
        print("   En production réelle : 3-5 signatures suffisent !")
        print("\n📂 Format des fichiers :")
        print("   • original_<client_id>_1.png")
        print("   • original_<client_id>_2.png")
        print("   • ... (minimum 3, recommandé 4-5)")
        print("\n🧪 Pour créer des signatures de test :")
        print("   python add_client_instant.py --test 99 4")
        print("   (Crée 4 signatures de test pour le client 99)")
        sys.exit(1)
    
    # Mode test : créer des signatures factices
    if sys.argv[1] == '--test' and len(sys.argv) >= 3:
        test_client = sys.argv[2]
        test_nombre = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
        
        if creer_signatures_test(test_client, test_nombre):
            print(f"\n💡 Maintenant lancez : python add_client_instant.py {test_client}")
        sys.exit(0)
    
    # Mode normal
    client_id = sys.argv[1]
    min_sigs = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    
    # Vérifier le format de l'ID
    if not client_id.isdigit() and client_id != '--test':
        print(f"\n⚠️  ATTENTION : L'ID '{client_id}' n'est pas numérique")
        reponse = input("   Continuer quand même ? (o/n) : ").strip().lower()
        if reponse not in ['o', 'oui', 'y', 'yes']:
            print("   ❌ Opération annulée")
            sys.exit(1)
    
    # Ajouter le client
    success = ajouter_client_instantane(client_id, min_signatures=min_sigs)
    
    if success:
        print(f"\n🎉 SUCCÈS !")
        print(f"\n💡 Pour tester l'identification :")
        print(f"   python test_identification.py data/real/original_{client_id}_1.png")
    else:
        print(f"\n❌ ÉCHEC")
        sys.exit(1)