import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from collections import defaultdict
import random

class SiameseEncoderV2:
    """
    Version AMÉLIORÉE avec :
    - Plus de paires d'entraînement
    - Utilisation des fausses signatures
    - Meilleure discrimination
    """
    
    def __init__(self, embedding_size=128, model_path='models/siamese_encoder_v2.weights.h5'):
        self.embedding_size = embedding_size
        self.model_path = model_path
        self.input_shape = (150, 150, 3)
        
        os.makedirs('models', exist_ok=True)
        
        self.encoder = self._build_encoder()
        
        if os.path.exists(model_path):
            try:
                self.encoder.load_weights(model_path)
                print("✅ Encodeur V2 chargé")
                self.is_trained = True
            except:
                print("🧠 Nouvel encodeur V2 créé")
                self.is_trained = False
        else:
            print("🧠 Nouvel encodeur V2 créé")
            self.is_trained = False
    
    def _build_encoder(self):
        """Architecture d'encodage"""
        model = models.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.embedding_size, activation=None),
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
        ], name='encoder')
        
        return model
    
    def _preprocess(self, image_path):
        """Prétraitement"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.resize(img, (150, 150))
            img = img.astype('float32') / 255.0
            return img
        except:
            return None
    
    def encode(self, image_path):
        """Encode une signature"""
        img = self._preprocess(image_path)
        if img is None:
            return None
        img_batch = np.expand_dims(img, axis=0)
        embedding = self.encoder.predict(img_batch, verbose=0)[0]
        return embedding
    
    def encode_batch(self, image_paths):
        """Encode plusieurs signatures"""
        images = []
        valid_paths = []
        
        for path in image_paths:
            img = self._preprocess(path)
            if img is not None:
                images.append(img)
                valid_paths.append(path)
        
        if len(images) == 0:
            return None, []
        
        images_batch = np.array(images)
        embeddings = self.encoder.predict(images_batch, verbose=0)
        
        return embeddings, valid_paths
    
    def train(self, dossier_real, dossier_fake, epochs=40, batch_size=64):
        """
        ENTRAÎNEMENT AMÉLIORÉ avec :
        - Plus de paires (8000+)
        - Utilisation des FAUSSES signatures
        - Plus d'epochs (40)
        """
        print("="*70)
        print("🔥 ENTRAÎNEMENT SIAMESE V2 (VERSION AMÉLIORÉE)")
        print("="*70)
        
        def extract_client_id(filepath):
            basename = os.path.basename(filepath)
            if basename.startswith('original_'):
                return basename.split('_')[1]
            elif basename.startswith('forgeries_'):
                return basename.split('_')[1]
            return None
        
        # 1. CHARGER LES VRAIES SIGNATURES
        print("\n📂 Chargement des VRAIES signatures...")
        
        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        files_real = [os.path.join(dossier_real, f) for f in os.listdir(dossier_real)
                     if f.lower().endswith(extensions)]
        
        clients_real = defaultdict(list)
        for f in files_real:
            client_id = extract_client_id(f)
            if client_id:
                clients_real[client_id].append(f)
        
        print(f"   ✅ {len(files_real)} signatures vraies")
        print(f"   ✅ {len(clients_real)} clients")
        
        # 2. CHARGER LES FAUSSES SIGNATURES
        print("\n📂 Chargement des FAUSSES signatures...")
        
        files_fake = [os.path.join(dossier_fake, f) for f in os.listdir(dossier_fake)
                     if f.lower().endswith(extensions)]
        
        clients_fake = defaultdict(list)
        for f in files_fake:
            client_id = extract_client_id(f)
            if client_id and client_id in clients_real:
                clients_fake[client_id].append(f)
        
        print(f"   ✅ {len(files_fake)} signatures fausses")
        print(f"   ✅ {len(clients_fake)} clients avec fausses signatures")
        
        # 3. GÉNÉRER LES PAIRES D'ENTRAÎNEMENT
        print("\n🔗 Génération des paires (VERSION AMÉLIORÉE)...")
        
        pairs = []
        labels = []
        
        # === PAIRES POSITIVES : même personne ===
        print("\n   Type 1 : Paires POSITIVES (même client)")
        
        for client_id, sigs in clients_real.items():
            if len(sigs) < 2:
                continue
            
            # AUGMENTÉ : 40 paires par client (au lieu de 15)
            for _ in range(min(40, len(sigs) * 2)):
                sig1, sig2 = random.sample(sigs, 2)
                pairs.append((sig1, sig2))
                labels.append(1)  # Similaire
        
        nb_positives = len(labels)
        print(f"      ✅ {nb_positives} paires positives")
        
        # === PAIRES NÉGATIVES TYPE 1 : personnes différentes ===
        print("\n   Type 2 : Paires NÉGATIVES (clients différents)")
        
        client_list = list(clients_real.keys())
        
        for _ in range(nb_positives):
            c1, c2 = random.sample(client_list, 2)
            sig1 = random.choice(clients_real[c1])
            sig2 = random.choice(clients_real[c2])
            pairs.append((sig1, sig2))
            labels.append(0)  # Différent
        
        nb_negatives_inter = len(labels) - nb_positives
        print(f"      ✅ {nb_negatives_inter} paires négatives (inter-clients)")
        
        # === PAIRES NÉGATIVES TYPE 2 : vraie vs fausse (NOUVEAU !) ===
        print("\n   Type 3 : Paires NÉGATIVES (vraie vs FRAUDE) 🆕")
        
        nb_paires_fraude = 0
        
        for client_id in clients_real.keys():
            if client_id not in clients_fake:
                continue
            
            sigs_vraies = clients_real[client_id]
            sigs_fausses = clients_fake[client_id]
            
            # 20 paires vraie/fausse par client
            for _ in range(min(20, len(sigs_vraies) * len(sigs_fausses))):
                sig_vraie = random.choice(sigs_vraies)
                sig_fausse = random.choice(sigs_fausses)
                pairs.append((sig_vraie, sig_fausse))
                labels.append(0)  # Différent !
                nb_paires_fraude += 1
        
        print(f"      ✅ {nb_paires_fraude} paires vraie/fraude")
        
        print(f"\n   📊 TOTAL : {len(pairs)} paires")
        print(f"      • Similaires  : {nb_positives}")
        print(f"      • Différentes : {len(labels) - nb_positives}")
        print(f"         - Inter-clients : {nb_negatives_inter}")
        print(f"         - Vraie/Fraude  : {nb_paires_fraude} 🔥")
        
        # Mélanger
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs, labels = zip(*combined)
        
        # 4. PRÉTRAITER
        print("\n🖼️  Prétraitement des images...")
        
        X1, X2, y = [], [], []
        
        total = len(pairs)
        for idx, ((path1, path2), label) in enumerate(zip(pairs, labels)):
            if idx % 500 == 0:
                pct = idx / total * 100
                print(f"   📊 {idx}/{total} ({pct:.1f}%)")
            
            img1 = self._preprocess(path1)
            img2 = self._preprocess(path2)
            
            if img1 is not None and img2 is not None:
                X1.append(img1)
                X2.append(img2)
                y.append(label)
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        
        print(f"\n   ✅ {len(X1)} paires prêtes")
        
        # 5. CONSTRUIRE LE MODÈLE
        print("\n🏗️  Construction du modèle Siamese...")
        
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        embedding_a = self.encoder(input_a)
        embedding_b = self.encoder(input_b)
        
        distance = layers.Lambda(
            lambda x: tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1, keepdims=True)
        )([embedding_a, embedding_b])
        
        output = layers.Dense(1, activation='sigmoid')(distance)
        
        siamese_model = models.Model(inputs=[input_a, input_b], outputs=output)
        
        siamese_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("   ✅ Modèle créé")
        
        # 6. ENTRAÎNER
        print(f"\n{'='*70}")
        print("⚡ DÉMARRAGE DE L'ENTRAÎNEMENT AMÉLIORÉ")
        print(f"{'='*70}")
        print(f"   • Epochs          : {epochs}")
        print(f"   • Batch size      : {batch_size}")
        print(f"   • Paires totales  : {len(X1)}")
        print(f"   • Avec fraudes    : OUI ✅")
        print(f"   ⏱️  Temps estimé   : {epochs * 1.2:.0f}-{epochs * 1.5:.0f} minutes")
        print("")
        
        history = siamese_model.fit(
            [X1, X2], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    self.model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=0.00001,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
        )
        
        self.encoder.save_weights(self.model_path)
        self.is_trained = True
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT TERMINÉ !")
        print("="*70)
        
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        print(f"\n📊 Performances finales :")
        print(f"   • Précision entraînement : {final_acc*100:.2f}%")
        print(f"   • Précision validation   : {final_val_acc*100:.2f}%")
        
        if final_val_acc > 0.90:
            print(f"\n🏆 EXCELLENT ! Le modèle est très performant.")
        elif final_val_acc > 0.85:
            print(f"\n✅ BON ! Le modèle fonctionne bien.")
        
        print(f"\n🎯 Le modèle a appris à distinguer :")
        print(f"   ✅ Signatures du même client")
        print(f"   ✅ Signatures de clients différents")
        print(f"   ✅ Vraies signatures vs FRAUDES 🔥")
        
        print(f"\n💡 Prochaine étape : python base_empreintes_v2.py")
        
        return history


if __name__ == "__main__":
    print("="*70)
    print("🚀 ENTRAÎNEMENT SIAMESE V2 (AMÉLIORÉ)")
    print("="*70)
    
    encoder = SiameseEncoderV2()
    
    if not encoder.is_trained:
        print("\n⚡ Configuration améliorée :")
        print("   • 40 paires positives par client")
        print("   • 20 paires vraie/fraude par client")
        print("   • 40 epochs")
        print("   • Utilisation des fausses signatures ✅")
        print("")
        
        history = encoder.train(
            dossier_real="data/real",
            dossier_fake="data/fake",
            epochs=40,
            batch_size=64
        )
    else:
        print("\n✅ Encodeur V2 déjà entraîné")