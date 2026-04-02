from base_empreintes import BaseEmpreintes
import sys
import os
import numpy as np

def tester_fausse_signature(signature_path, seuil_fraude=0.15):
    """
    Teste une FAUSSE signature (forgerie)
    Doit détecter :
    1. Que c'est une fraude
    2. Quel client est imité

    Le seuil est une distance cosinus (0 = identique, 2 = opposé).
    Valeur recommandée : 0.10–0.20 (plus bas = plus strict).
    """
    
    print("="*70)
    print("🔍 TEST DE FAUSSE SIGNATURE (FORGERIE)")
    print("="*70)
    
    if not os.path.exists(signature_path):
        print(f"❌ Fichier introuvable : {signature_path}")
        return
    
    filename = os.path.basename(signature_path)
    
    # Extraire le client imité depuis le nom du fichier
    try:
        if filename.startswith('forgeries_'):
            client_imite = filename.split('_')[1]
        elif filename.startswith('fake_'):
            client_imite = filename.split('_')[1]
        else:
            client_imite = input("Quel client est imité ? ")
    except:
        client_imite = input("Quel client est imité ? ")
    
    print(f"\n📄 Fichier          : {filename}")
    print(f"🎭 Client imité     : {client_imite}")
    print(f"⚙️  Seuil de fraude : {seuil_fraude}")
    
    # Charger la base
    print(f"\n🗄️  Chargement de la base...")
    base = BaseEmpreintes()
    
    if base.clients is None:
        print("❌ Base non initialisée")
        return
    
    print(f"✅ Base chargée : {len(base.clients)} clients")
    
    # Vérifier que le client imité est dans la base
    if client_imite not in base.clients:
        print(f"\n⚠️  Le client imité ({client_imite}) n'est pas dans la base")
    else:
        print(f"✅ Le client imité ({client_imite}) est dans la base")
    
    # Encoder la fausse signature
    print(f"\n🔍 Analyse de la fausse signature...")
    empreinte_fake = base.encoder.encode(signature_path)
    
    if empreinte_fake is None:
        print("❌ Erreur d'encodage")
        return
    
    # Calculer les distances cosinus avec TOUS les clients
    distances = []
    
    for client_id, data in base.clients.items():
        empreintes_client = data['empreintes']
        
        # Cosine similarity = dot product for L2-normalized vectors
        cos_sims = empreintes_client @ empreinte_fake  # shape: (N,)
        sim_max = float(np.max(cos_sims))
        sim_moy = float(np.mean(cos_sims))
        
        # Cosine distance: 0 = identical, smaller = more similar
        dist_min = 1.0 - sim_max
        dist_moy = 1.0 - sim_moy
        
        distances.append({
            'client_id': client_id,
            'dist_min': dist_min,
            'dist_moy': dist_moy
        })
    
    # Trier par distance minimale
    distances.sort(key=lambda x: x['dist_min'])
    
    # Client le plus proche
    plus_proche = distances[0]
    
    # Trouver la position du client imité
    rang_imite = None
    dist_imite = None
    
    for i, d in enumerate(distances, 1):
        if d['client_id'] == client_imite:
            rang_imite = i
            dist_imite = d['dist_min']
            break
    
    # ========================================================================
    # ANALYSE
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("🎯 RÉSULTAT DE L'ANALYSE")
    print(f"{'='*70}")
    
    # 1. Décision : Fraude ou non ?
    if plus_proche['dist_min'] < seuil_fraude:
        decision = "ACCEPTÉE (FAUX POSITIF ❌)"
        emoji = "⚠️"
        accepte = True
    else:
        decision = "REJETÉE (CORRECT ✅)"
        emoji = "🚨"
        accepte = False
    
    print(f"\n{emoji} Décision : {decision}")
    
    # 2. Client le plus proche
    print(f"\n🔍 Client le plus proche :")
    print(f"   • Client   : {plus_proche['client_id']}")
    print(f"   • Distance : {plus_proche['dist_min']:.4f}")
    
    if accepte:
        print(f"   ❌ LE SYSTÈME A ACCEPTÉ CETTE FRAUDE !")
        print(f"   💥 FAUX POSITIF - SÉCURITÉ COMPROMISE")
    else:
        print(f"   ✅ Le système a bien rejeté cette fraude")
    
    # 3. Position du client imité
    if rang_imite:
        print(f"\n🎭 Client imité ({client_imite}) :")
        print(f"   • Rang     : #{rang_imite}")
        print(f"   • Distance : {dist_imite:.4f}")
        
        if rang_imite == 1:
            print(f"   ✅ Le client imité est bien le plus proche")
            print(f"   💡 La fraude ressemble effectivement au client {client_imite}")
        else:
            print(f"   ❌ Le client imité N'EST PAS le plus proche")
            print(f"   💡 La fraude ressemble plus au client {plus_proche['client_id']}")
            print(f"      qu'au client {client_imite}")
    
    # 4. Top 5
    print(f"\n📊 Top 5 des clients les plus proches :")
    
    for i, d in enumerate(distances[:5], 1):
        emoji_rang = ["🥇", "🥈", "🥉", "  ", "  "][i-1]
        
        marker = ""
        if d['client_id'] == client_imite:
            marker = " ← CLIENT IMITÉ 🎭"
        
        status = "✅ Accepté" if d['dist_min'] < seuil_fraude else "❌ Rejeté"
        
        print(f"   {emoji_rang} {i}. Client {d['client_id']:3s} : dist={d['dist_min']:.4f} ({status}){marker}")
    
    # ========================================================================
    # ÉVALUATION
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("📊 ÉVALUATION")
    print(f"{'='*70}")
    
    # Critère 1 : La fraude est-elle détectée ?
    print(f"\n1️⃣  Détection de fraude :")
    
    if accepte:
        print(f"   ❌ ÉCHEC ! La fraude a été acceptée")
        print(f"   Distance : {plus_proche['dist_min']:.4f} < Seuil : {seuil_fraude}")
        print(f"\n   💡 Raisons possibles :")
        print(f"      a) Le faussaire est très doué (bonne imitation)")
        print(f"      b) Le seuil est trop permissif")
        print(f"      c) Le modèle n'est pas assez discriminant")
    else:
        print(f"   ✅ SUCCÈS ! La fraude a été rejetée")
        print(f"   Distance : {plus_proche['dist_min']:.4f} > Seuil : {seuil_fraude}")
    
    # Critère 2 : Le client imité est-il identifié ?
    print(f"\n2️⃣  Identification du client imité :")
    
    if rang_imite == 1:
        print(f"   ✅ CORRECT ! Le client {client_imite} est bien le plus proche")
    elif rang_imite and rang_imite <= 3:
        print(f"   ⚠️  PARTIEL ! Le client {client_imite} est dans le top 3 (rang {rang_imite})")
    elif rang_imite:
        print(f"   ❌ INCORRECT ! Le client {client_imite} est au rang {rang_imite}")
    else:
        print(f"   ❌ Le client {client_imite} n'est pas dans la base")
    
    # ========================================================================
    # RECOMMANDATIONS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("💡 RECOMMANDATIONS")
    print(f"{'='*70}")
    
    if accepte:
        # Fraude acceptée = PROBLÈME GRAVE
        print(f"\n🚨 PROBLÈME DE SÉCURITÉ DÉTECTÉ !")
        print(f"\n🔧 Solutions :")
        
        print(f"\n   Option 1 : Rendre le seuil plus strict")
        print(f"      • Seuil actuel : {seuil_fraude}")
        
        # Calculer un seuil qui rejetterait cette fraude
        seuil_suggere = round(plus_proche['dist_min'] * 0.9, 2)
        print(f"      • Seuil suggéré : {seuil_suggere}")
        print(f"      • Test : python test_forgery.py {signature_path} {seuil_suggere}")
        
        print(f"\n   Option 2 : Améliorer le modèle")
        print(f"      • Ré-entraîner avec plus d'epochs (30-40)")
        print(f"      • Ajouter plus de signatures authentiques du client {client_imite}")
        print(f"      • Utiliser les fausses signatures pendant l'entraînement")
    
    else:
        # Fraude rejetée = BON
        print(f"\n✅ Le système fonctionne correctement pour cette fraude")
        
        if rang_imite and rang_imite > 1:
            print(f"\n⚠️  Mais le client imité n'est pas #1")
            print(f"\n💡 Pour améliorer l'identification du client imité :")
            print(f"   • Ajouter plus de signatures du client {client_imite}")
    
    # Statistiques
    print(f"\n📈 Statistiques :")
    print(f"   • Distance fraude : {plus_proche['dist_min']:.4f}")
    print(f"   • Distance imité  : {dist_imite:.4f}" if dist_imite else "   • Client imité non trouvé")
    print(f"   • Écart           : {dist_imite - plus_proche['dist_min']:.4f}" if dist_imite else "")
    
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*70)
        print("🔍 TEST DE FAUSSE SIGNATURE (FORGERIE)")
        print("="*70)
        print("\n📋 Usage :")
        print("   python test_forgery.py <fausse_signature> [seuil]")
        print("\n📝 Exemples :")
        print("   python test_forgery.py data/fake/forgeries_6_1.png")
        print("   python test_forgery.py data/fake/forgeries_6_1.png 0.10")
        print("   python test_forgery.py data/fake/forgeries_6_1.png 0.20")
        print("\n💡 Ce script teste si le système :")
        print("   1. Détecte que c'est une FRAUDE")
        print("   2. Identifie quel client est IMITÉ")
        print("\n⚙️  Le seuil est une distance cosinus (0=identique) :")
        print("   • 0.10 : Très strict")
        print("   • 0.15 : Équilibré (RECOMMANDÉ)")
        print("   • 0.20 : Permissif")
        sys.exit(1)
    
    signature = sys.argv[1]
    seuil = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.15
    
    tester_fausse_signature(signature, seuil)