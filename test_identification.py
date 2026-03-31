from base_empreintes import BaseEmpreintes
import sys
import os

def tester_signature_avec_detection_fraude(signature_path, seuil_fraude=0.50):
    """
    Teste l'identification d'une signature AVEC détection de fraude
    
    Args:
        signature_path: Chemin vers la signature à identifier
        seuil_fraude: Seuil de distance pour détecter une fraude
                      Plus bas = plus strict
                      Plus haut = plus permissif
    
    Seuils recommandés :
        - 0.35 : Très strict (peu de faux positifs, plus de faux négatifs)
        - 0.45 : Équilibré (recommandé)
        - 0.60 : Permissif (moins de rejets, plus de risque)
    """
    print("="*70)
    print("🔍 TEST D'IDENTIFICATION AVEC DÉTECTION DE FRAUDE")
    print("="*70)
    
    # Vérifier que le fichier existe
    if not os.path.exists(signature_path):
        print(f"\n❌ Fichier introuvable : {signature_path}")
        return
    
    print(f"\n📄 Signature à tester : {os.path.basename(signature_path)}")
    print(f"   Chemin complet : {signature_path}")
    print(f"   ⚙️  Seuil de fraude : {seuil_fraude:.2f}")
    
    # Charger la base
    print(f"\n🗄️  Chargement de la base d'empreintes...")
    base = BaseEmpreintes()
    
    if base.clients is None:
        print("\n❌ ERREUR : Base non initialisée")
        return
    
    print(f"   ✅ Base chargée : {len(base.clients)} clients enregistrés")
    
    # Identifier avec seuil personnalisé
    print(f"\n🔍 Identification en cours...")
    result = base.identifier(signature_path, top_k=5)
    
    if 'error' in result:
        print(f"\n❌ Erreur : {result['error']}")
        return
    
    # ========================================================================
    # DÉTECTION DE FRAUDE
    # ========================================================================
    
    best = result['best_match']
    distance = best['distance']
    
    # DÉCISION BASÉE SUR LA DISTANCE
    if distance < seuil_fraude:
        decision = "AUTHENTIQUE"
        emoji_decision = "✅"
        couleur = "VERT"
        explication = "La signature correspond à un client enregistré"
    else:
        decision = "FRAUDE SUSPECTÉE"
        emoji_decision = "🚨"
        couleur = "ROUGE"
        explication = "La signature ne correspond à aucun client connu"
    
    # ========================================================================
    # AFFICHAGE DES RÉSULTATS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("🎯 DÉCISION FINALE")
    print(f"{'='*70}")
    
    print(f"\n{emoji_decision} {decision}")
    print(f"   {explication}")
    
    print(f"\n📊 Détails de l'analyse :")
    print(f"   • Distance minimale : {distance:.4f}")
    print(f"   • Seuil de fraude   : {seuil_fraude:.4f}")
    
    # Niveau de confiance dans la décision
    if distance < seuil_fraude * 0.5:
        niveau_confiance = "TRÈS HAUTE"
        emoji_conf = "🟢"
    elif distance < seuil_fraude * 0.8:
        niveau_confiance = "HAUTE"
        emoji_conf = "🟢"
    elif distance < seuil_fraude:
        niveau_confiance = "MOYENNE"
        emoji_conf = "🟡"
    elif distance < seuil_fraude * 1.3:
        niveau_confiance = "FAIBLE (zone grise)"
        emoji_conf = "🟠"
    else:
        niveau_confiance = "TRÈS HAUTE (rejet)"
        emoji_conf = "🔴"
    
    print(f"   • Confiance décision: {niveau_confiance} {emoji_conf}")
    
    # ========================================================================
    # SI AUTHENTIQUE : Afficher le client identifié
    # ========================================================================
    
    if decision == "AUTHENTIQUE":
        print(f"\n{'='*70}")
        print("👤 CLIENT IDENTIFIÉ")
        print(f"{'='*70}")
        
        print(f"\n🏆 Meilleure correspondance :")
        print(f"   • Client      : {best['client_id']}")
        print(f"   • Distance    : {distance:.4f}")
        print(f"   • Confiance   : {best['confiance_pct']:.1f}%")
        
        # Interprétation de la qualité de la correspondance
        if distance < 0.10:
            qualite = "Excellente correspondance ✅"
        elif distance < 0.25:
            qualite = "Bonne correspondance ✅"
        elif distance < seuil_fraude * 0.8:
            qualite = "Correspondance acceptable ✅"
        else:
            qualite = "Correspondance limite ⚠️"
        
        print(f"   • Qualité     : {qualite}")
        
        # Top 5
        print(f"\n📊 Top 5 des correspondances :")
        for i, pred in enumerate(result['top_predictions'], 1):
            if i == 1:
                emoji = "🥇"
            elif i == 2:
                emoji = "🥈"
            elif i == 3:
                emoji = "🥉"
            else:
                emoji = "  "
            
            # Marquer si au-dessus du seuil
            if pred['distance'] >= seuil_fraude:
                marker = " ⚠️ (au-dessus du seuil)"
            else:
                marker = ""
            
            print(f"   {emoji} {i}. Client {pred['client_id']:3s} : {pred['confiance_pct']:5.1f}% (dist: {pred['distance']:.4f}){marker}")
    
    # ========================================================================
    # SI FRAUDE : Afficher pourquoi c'est rejeté
    # ========================================================================
    
    else:
        print(f"\n{'='*70}")
        print("🚨 ANALYSE DE LA FRAUDE")
        print(f"{'='*70}")
        
        print(f"\n⚠️  Raison du rejet :")
        print(f"   La distance minimale ({distance:.4f}) dépasse le seuil ({seuil_fraude:.4f})")
        
        ecart_pourcent = ((distance - seuil_fraude) / seuil_fraude) * 100
        print(f"   Écart : +{ecart_pourcent:.1f}% au-dessus du seuil")
        
        print(f"\n🔍 Client le plus proche (mais rejeté) :")
        print(f"   • Client      : {best['client_id']}")
        print(f"   • Distance    : {distance:.4f}")
        print(f"   • Confiance   : {best['confiance_pct']:.1f}%")
        
        print(f"\n💡 Interprétation :")
        print(f"   Cette signature ne correspond à AUCUN client de la base.")
        print(f"   Scénarios possibles :")
        print(f"      1. Client non enregistré dans le système")
        print(f"      2. Tentative de fraude/falsification")
        print(f"      3. Signature mal scannée/illisible")
        print(f"      4. Signature trop différente de celles enregistrées")
        
        # Top 5 pour information
        print(f"\n📊 Top 5 (tous rejetés) :")
        for i, pred in enumerate(result['top_predictions'], 1):
            emoji = ["🥇", "🥈", "🥉", "  ", "  "][i-1]
            print(f"   {emoji} {i}. Client {pred['client_id']:3s} : {pred['confiance_pct']:5.1f}% (dist: {pred['distance']:.4f}) ❌")
    
    # ========================================================================
    # VÉRIFICATION AUTOMATIQUE (si possible)
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("✅ VÉRIFICATION")
    print(f"{'='*70}")
    
    try:
        filename = os.path.basename(signature_path)
        
        # Extraire le vrai client
        if filename.startswith('original_'):
            vrai_client = filename.split('_')[1]
        elif filename.startswith('signature_'):
            vrai_client = filename.split('_')[1]
        elif filename.startswith('fake_') or filename.startswith('forge_'):
            vrai_client = "FRAUDE"
        elif filename.startswith('inconnu_'):
            vrai_client = "INCONNU"
        else:
            vrai_client = None
        
        if vrai_client:
            print(f"   • Vérité terrain  : {vrai_client}")
            print(f"   • Décision système: {decision}")
            
            if vrai_client == "FRAUDE" or vrai_client == "INCONNU":
                # C'était vraiment une fraude
                if decision == "FRAUDE SUSPECTÉE":
                    print(f"   ✅ CORRECT ! Fraude bien détectée (Vrai Négatif)")
                else:
                    print(f"   ❌ ERREUR ! Fraude non détectée (Faux Positif)")
            
            elif decision == "AUTHENTIQUE":
                # Système a accepté
                if best['client_id'] == vrai_client:
                    print(f"   ✅ CORRECT ! Client bien identifié (Vrai Positif)")
                else:
                    print(f"   ❌ ERREUR ! Mauvais client identifié")
            
            else:
                # Système a rejeté
                print(f"   ❌ ERREUR ! Client légitime rejeté (Faux Négatif)")
                print(f"   💡 Suggestion : Augmenter le seuil ou ajouter plus de signatures d'entraînement")
        
        else:
            print(f"   ℹ️  Impossible de détecter la vérité terrain depuis le nom du fichier")
    
    except Exception as e:
        print(f"   ℹ️  Vérification automatique impossible")
    
    # ========================================================================
    # RECOMMANDATIONS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("💡 RECOMMANDATIONS")
    print(f"{'='*70}")
    
    if decision == "AUTHENTIQUE":
        if distance < 0.20:
            print(f"   ✅ Autoriser l'accès")
        elif distance < seuil_fraude * 0.8:
            print(f"   ⚠️  Autoriser avec vérification secondaire (ex: code PIN)")
        else:
            print(f"   ⚠️  Autoriser mais surveiller (zone limite)")
    else:
        print(f"   🚨 REFUSER l'accès")
        print(f"   📞 Contacter le client ou demander une vérification manuelle")
    
    print("="*70)


# ============================================================================
# UTILISATION EN LIGNE DE COMMANDE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🔐 TEST D'IDENTIFICATION AVEC DÉTECTION DE FRAUDE")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n❌ Usage incorrect")
        print("\n📋 Usage :")
        print("   python test_identification_with_fraud_detection.py <signature> [seuil]")
        print("\n📝 Exemples :")
        print("   # Test basique")
        print("   python test_identification_with_fraud_detection.py data/real/original_1_5.png")
        print("")
        print("   # Test avec seuil personnalisé")
        print("   python test_identification_with_fraud_detection.py data/real/signature_99_1.png 0.45")
        print("")
        print("   # Test d'une fraude (si vous avez créé le fichier)")
        print("   python test_identification_with_fraud_detection.py data/fake/forge_1_1.png")
        print("\n⚙️  Seuils de fraude :")
        print("   • 0.35 : Très strict (sécurité maximale, plus de rejets)")
        print("   • 0.45 : Équilibré (RECOMMANDÉ)")
        print("   • 0.60 : Permissif (moins de rejets, plus de risque)")
        print("\n💡 Le seuil détermine à partir de quelle distance on considère")
        print("   qu'une signature ne correspond à aucun client de la base.")
        
        sys.exit(1)
    
    signature_path = sys.argv[1]
    seuil = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.45
    
    tester_signature_avec_detection_fraude(signature_path, seuil_fraude=seuil)