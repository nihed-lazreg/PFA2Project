"""Siamese signature verification encoder with cosine similarity and triplet loss.

Architecture:
    - Shared CNN encoder: maps signature images to L2-normalized embeddings.
    - Training  : triplet model (anchor / positive / negative) + triplet loss
                  with cosine distance.
    - Inference : embedding model (single signature) + pairwise cosine similarity
                  model (two signatures).

Quick-start::

    # ── Training (triplet loss) ──────────────────────────────────────────────
    encoder = SiameseEncoderV2()
    history = encoder.train(dossier_real='data/real', dossier_fake='data/fake')

    # ── Inference (encode + cosine similarity) ───────────────────────────────
    emb_a = encoder.encode('path/to/sig_a.png')   # shape (128,)
    emb_b = encoder.encode('path/to/sig_b.png')
    sim   = SiameseEncoderV2.cosine_similarity(emb_a, emb_b)  # 1.0 = identical

    # ── Two-input Keras model for batch inference ────────────────────────────
    siamese = encoder.build_siamese_model()
    score   = siamese.predict([img_a_batch, img_b_batch])   # shape (N, 1)
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from collections import defaultdict
import random


# ============================================================================
# Custom triplet loss
# ============================================================================

def triplet_loss(margin=0.2):
    """Triplet loss using cosine distance.

    Cosine distance: d(x, y) = 1 - cosine_similarity(x, y)
    Loss           : max(d(anchor, positive) - d(anchor, negative) + margin, 0)

    The loss function expects ``y_pred`` to be the **concatenation** of the
    three L2-normalized embeddings produced by the triplet model:
    ``[emb_anchor || emb_positive || emb_negative]`` with shape
    ``(batch_size, 3 * embedding_size)``.
    ``y_true`` is **ignored**; pass a zeros array of shape ``(N, 1)``::

        y_dummy = np.zeros((len(X_anchor), 1))
        model.fit([X_anchor, X_positive, X_negative], y_dummy, ...)

    Args:
        margin: Minimum required gap between the positive and negative
                cosine distances (default 0.2, configurable).

    Returns:
        A Keras-compatible loss callable.
    """
    def loss(y_true, y_pred):
        # y_pred shape: (batch_size, 3 * embedding_size)
        embedding_size = tf.shape(y_pred)[1] // 3
        anchor   = y_pred[:, :embedding_size]
        positive = y_pred[:, embedding_size : 2 * embedding_size]
        negative = y_pred[:, 2 * embedding_size :]

        # Cosine similarity for unit vectors equals the dot product
        sim_ap = tf.reduce_sum(anchor * positive, axis=1)  # (batch,)
        sim_an = tf.reduce_sum(anchor * negative, axis=1)  # (batch,)

        # Cosine distance: d = 1 - cosine_similarity
        d_ap = 1.0 - sim_ap
        d_an = 1.0 - sim_an

        # Hinge-style triplet loss
        loss_val = tf.maximum(d_ap - d_an + margin, 0.0)
        return tf.reduce_mean(loss_val)

    loss.__name__ = f'triplet_loss_margin_{margin}'
    return loss


# ============================================================================
# Custom callback: save only the encoder weights when val_loss improves
# ============================================================================

class _SaveEncoderWeights(tf.keras.callbacks.Callback):
    """Saves encoder weights (not the full triplet-model weights) when the
    monitored metric improves.  This keeps the saved file compatible with
    ``encoder.load_weights(path)`` at inference time."""

    def __init__(self, encoder, path, monitor='val_loss'):
        super().__init__()
        self._encoder = encoder
        self._path = path
        self._monitor = monitor
        self._best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self._monitor, float('inf'))
        if current < self._best:
            self._best = current
            self._encoder.save_weights(self._path)
            print(f'\nEpoch {epoch + 1}: {self._monitor} improved to '
                  f'{current:.4f} – encoder saved to {self._path}')


# ============================================================================
# Main class
# ============================================================================

class SiameseEncoderV2:
    """Handwritten signature verification using metric learning.

    The shared CNN encoder maps signature images (150 × 150 × 3) to
    L2-normalized embeddings (default 128-dimensional).  Two operational
    modes are provided:

    **Training mode** (triplet loss with cosine distance)::

        anchor   → encoder → emb_a ─┐
        positive → encoder → emb_p ─┤→ triplet_loss(margin)
        negative → encoder → emb_n ─┘

    **Inference mode** (cosine similarity)::

        sig_a → encoder → emb_a ─┐
        sig_b → encoder → emb_b ─┤→ dot(emb_a, emb_b) = cosine_similarity
                                  ┘

    Because embeddings are L2-normalized inside the model graph, the dot
    product of two embeddings equals their cosine similarity exactly.
    """

    def __init__(self, embedding_size=128,
                 model_path='models/siamese_encoder_v2.weights.h5'):
        self.embedding_size = embedding_size
        self.model_path = model_path
        self.input_shape = (150, 150, 3)

        os.makedirs('models', exist_ok=True)

        # Build the shared encoder (reused by all model variants)
        self.encoder = self._build_encoder()

        if os.path.exists(model_path):
            try:
                self.encoder.load_weights(model_path)
                print("✅ Encodeur V2 chargé")
                self.is_trained = True
            except Exception as e:
                print(f"🧠 Nouvel encodeur V2 créé (chargement échoué : {e})")
                self.is_trained = False
        else:
            print("🧠 Nouvel encodeur V2 créé")
            self.is_trained = False

    # ------------------------------------------------------------------
    # Encoder architecture
    # ------------------------------------------------------------------

    def _build_encoder(self):
        """CNN encoder: image → L2-normalized embedding.

        The final ``l2_normalize`` layer ensures all output vectors are unit
        vectors, which makes the dot product equal to cosine similarity.
        No activation is used on the embedding layer so the network can freely
        learn any direction in embedding space.
        """
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

            # Raw embedding (no activation)
            layers.Dense(self.embedding_size, activation=None),

            # L2 normalization: output is a unit vector
            # → dot product of two outputs = cosine similarity
            layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                          name='l2_normalize'),
        ], name='encoder')

        return model

    # ------------------------------------------------------------------
    # Model builders (training vs inference)
    # ------------------------------------------------------------------

    def build_siamese_model(self):
        """Build a pairwise cosine-similarity model for inference.

        Inputs : ``[input_a, input_b]`` – two signature images (150 × 150 × 3).
        Output : cosine similarity score ∈ [−1, 1].
                 (1.0 = identical embeddings; −1.0 = opposite).

        Because embeddings are L2-normalized, the ``Dot`` layer with
        ``normalize=False`` computes the exact cosine similarity.

        Returns:
            A Keras ``Model`` (not compiled; inference only).
        """
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')

        emb_a = self.encoder(input_a)
        emb_b = self.encoder(input_b)

        # Dot product of unit vectors = cosine similarity (no sigmoid head)
        similarity = layers.Dot(axes=1, normalize=False,
                                name='cosine_similarity')([emb_a, emb_b])

        return models.Model(inputs=[input_a, input_b], outputs=similarity,
                            name='siamese_cosine_model')

    def build_triplet_model(self, margin=0.2):
        """Build the triplet training model.

        The **same** encoder is applied to all three inputs (shared weights).

        Inputs : ``[anchor, positive, negative]`` – three signature images.
        Output : concatenated embeddings ``[emb_a || emb_p || emb_n]``.
                 The custom :func:`triplet_loss` unpacks them automatically.

        Training example::

            triplet_model = encoder.build_triplet_model(margin=0.2)
            # X_anchor, X_positive, X_negative: arrays (N, 150, 150, 3)
            y_dummy = np.zeros((len(X_anchor), 1))  # ignored by the loss
            triplet_model.fit([X_anchor, X_positive, X_negative], y_dummy, ...)

        Args:
            margin: Triplet loss margin (default 0.2, configurable).

        Returns:
            A compiled Keras ``Model`` ready for training.
        """
        input_anchor   = layers.Input(shape=self.input_shape, name='anchor')
        input_positive = layers.Input(shape=self.input_shape, name='positive')
        input_negative = layers.Input(shape=self.input_shape, name='negative')

        emb_anchor   = self.encoder(input_anchor)
        emb_positive = self.encoder(input_positive)
        emb_negative = self.encoder(input_negative)

        # Concatenate: the custom loss will split into three equal parts
        output = layers.Concatenate(name='triplet_embeddings')(
            [emb_anchor, emb_positive, emb_negative]
        )

        triplet_model = models.Model(
            inputs=[input_anchor, input_positive, input_negative],
            outputs=output,
            name='triplet_model',
        )

        # Optimizer: Adam with lr=0.0005 as specified
        triplet_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=triplet_loss(margin=margin),
        )

        return triplet_model

    # ------------------------------------------------------------------
    # Preprocessing and inference helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image_path):
        """Load and normalize a signature image to (150, 150, 3) float32 ∈ [0, 1]."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.resize(img, (150, 150))
            img = img.astype('float32') / 255.0
            return img
        except (cv2.error, OSError, ValueError) as e:
            print(f"⚠️  Erreur de chargement ({image_path}): {e}")
            return None

    def encode(self, image_path):
        """Encode a single signature image into an L2-normalized embedding.

        Args:
            image_path: Path to the signature image file.

        Returns:
            ``numpy.ndarray`` of shape ``(embedding_size,)``, or ``None`` on
            failure.
        """
        img = self._preprocess(image_path)
        if img is None:
            return None
        img_batch = np.expand_dims(img, axis=0)
        embedding = self.encoder.predict(img_batch, verbose=0)[0]
        return embedding

    def encode_batch(self, image_paths):
        """Encode a list of signature images in a single batch call.

        Args:
            image_paths: List of file paths to signature images.

        Returns:
            Tuple ``(embeddings, valid_paths)`` where ``embeddings`` has shape
            ``(N, embedding_size)`` and ``valid_paths`` contains the
            successfully loaded file paths.
        """
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

    @staticmethod
    def cosine_similarity(emb_a, emb_b):
        """Compute cosine similarity between two embeddings.

        Accepts both normalized embeddings (as returned by :meth:`encode`) and
        raw unnormalized vectors.  When the encoder is used normally its output
        is already L2-normalized, so the normalization step below is a no-op;
        the explicit division guards against callers that pass embeddings from
        other sources that may not be unit vectors.

        A value of ``1.0`` means identical directions; ``−1.0`` means opposite.

        Args:
            emb_a: ``numpy.ndarray`` of any shape matching ``emb_b``.
            emb_b: ``numpy.ndarray`` of any shape matching ``emb_a``.

        Returns:
            ``float`` in [−1, 1].
        """
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(emb_a / norm_a, emb_b / norm_b))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dossier_real, dossier_fake, epochs=40, batch_size=64,
              margin=0.2):
        """Train the encoder with triplet loss and cosine distance.

        Generates two types of triplets from the dataset:

        * **Type A** – anchor + positive from the same client, negative from
          a *different* client (inter-client discrimination).
        * **Type B** – anchor + positive from the same client, negative is a
          *forgery* of that client (genuine vs. forgery discrimination).

        After training the encoder weights are saved to :attr:`model_path`.

        Args:
            dossier_real: Directory containing genuine signatures.
                          Filenames must follow ``original_<id>_<n>.<ext>``.
            dossier_fake: Directory containing forged signatures.
                          Filenames must follow ``forgeries_<id>_<n>.<ext>``.
            epochs:       Number of training epochs (default 40).
            batch_size:   Mini-batch size (default 64).
            margin:       Triplet loss margin (default 0.2).

        Returns:
            Keras ``History`` object.
        """
        print("=" * 70)
        print("🔥 ENTRAÎNEMENT SIAMESE V2 (TRIPLET LOSS + COSINE SIMILARITY)")
        print("=" * 70)

        def extract_client_id(filepath):
            basename = os.path.basename(filepath)
            if basename.startswith('original_'):
                return basename.split('_')[1]
            elif basename.startswith('forgeries_'):
                return basename.split('_')[1]
            return None

        # ── 1. Load signatures ────────────────────────────────────────────────
        print("\n📂 Chargement des VRAIES signatures...")

        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
        files_real = [os.path.join(dossier_real, f)
                      for f in os.listdir(dossier_real)
                      if f.lower().endswith(extensions)]

        clients_real = defaultdict(list)
        for f in files_real:
            cid = extract_client_id(f)
            if cid:
                clients_real[cid].append(f)

        print(f"   ✅ {len(files_real)} signatures vraies")
        print(f"   ✅ {len(clients_real)} clients")

        print("\n📂 Chargement des FAUSSES signatures...")

        files_fake = [os.path.join(dossier_fake, f)
                      for f in os.listdir(dossier_fake)
                      if f.lower().endswith(extensions)]

        clients_fake = defaultdict(list)
        for f in files_fake:
            cid = extract_client_id(f)
            if cid and cid in clients_real:
                clients_fake[cid].append(f)

        print(f"   ✅ {len(files_fake)} signatures fausses")
        print(f"   ✅ {len(clients_fake)} clients avec fausses signatures")

        # ── 2. Generate triplets ──────────────────────────────────────────────
        print("\n🔗 Génération des triplets (ANCHOR / POSITIVE / NEGATIVE)...")

        triplets = []   # list of (anchor_path, positive_path, negative_path)
        client_list = list(clients_real.keys())

        # Type A: negative from a different client
        print("\n   Type A : négatif inter-clients")
        nb_a = 0
        for cid, sigs in clients_real.items():
            if len(sigs) < 2:
                continue
            for _ in range(min(40, len(sigs) * 2)):
                anchor, positive = random.sample(sigs, 2)
                other_clients = [c for c in client_list if c != cid]
                neg_client = random.choice(other_clients)
                negative = random.choice(clients_real[neg_client])
                triplets.append((anchor, positive, negative))
                nb_a += 1
        print(f"      ✅ {nb_a} triplets")

        # Type B: negative is a forgery of the same client
        print("\n   Type B : négatif = fraude du même client 🔥")
        nb_b = 0
        for cid in clients_real:
            if cid not in clients_fake:
                continue
            sigs_vraies  = clients_real[cid]
            sigs_fausses = clients_fake[cid]
            if len(sigs_vraies) < 2:
                continue
            for _ in range(min(20, len(sigs_vraies) * len(sigs_fausses))):
                anchor, positive = random.sample(sigs_vraies, 2)
                negative = random.choice(sigs_fausses)
                triplets.append((anchor, positive, negative))
                nb_b += 1
        print(f"      ✅ {nb_b} triplets")

        print(f"\n   📊 TOTAL : {len(triplets)} triplets")
        print(f"      • Type A (inter-clients) : {nb_a}")
        print(f"      • Type B (vraie/fraude)  : {nb_b} 🔥")

        random.shuffle(triplets)

        # ── 3. Preprocess images ──────────────────────────────────────────────
        print("\n🖼️  Prétraitement des images...")

        X_anchor, X_positive, X_negative = [], [], []
        total = len(triplets)
        for idx, (a_path, p_path, n_path) in enumerate(triplets):
            if idx % 500 == 0:
                print(f"   📊 {idx}/{total} ({idx / total * 100:.1f}%)")

            img_a = self._preprocess(a_path)
            img_p = self._preprocess(p_path)
            img_n = self._preprocess(n_path)

            if img_a is not None and img_p is not None and img_n is not None:
                X_anchor.append(img_a)
                X_positive.append(img_p)
                X_negative.append(img_n)

        X_anchor   = np.array(X_anchor)
        X_positive = np.array(X_positive)
        X_negative = np.array(X_negative)

        # Dummy labels – the triplet loss uses only y_pred (the embeddings)
        y_dummy = np.zeros((len(X_anchor), 1))

        print(f"\n   ✅ {len(X_anchor)} triplets prêts")

        # ── 4. Build and compile the triplet model ────────────────────────────
        print("\n🏗️  Construction du modèle triplet (cosine similarity)...")
        triplet_model = self.build_triplet_model(margin=margin)
        print("   ✅ Modèle créé")

        # ── 5. Train ──────────────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("⚡ DÉMARRAGE DE L'ENTRAÎNEMENT (TRIPLET LOSS)")
        print(f"{'=' * 70}")
        print(f"   • Epochs      : {epochs}")
        print(f"   • Batch size  : {batch_size}")
        print(f"   • Triplets    : {len(X_anchor)}")
        print(f"   • Margin      : {margin}")
        print(f"   • Avec fraudes: OUI ✅")
        print(f"   ⏱️  Temps estimé : {epochs * 1.2:.0f}–{epochs * 1.5:.0f} minutes\n")

        history = triplet_model.fit(
            [X_anchor, X_positive, X_negative],
            y_dummy,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                # Save only the encoder weights when val_loss improves
                # (compatible with encoder.load_weights() at inference time)
                _SaveEncoderWeights(self.encoder, self.model_path,
                                    monitor='val_loss'),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-5,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=7,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ],
        )

        # Persist the final encoder weights for future inference
        self.encoder.save_weights(self.model_path)
        self.is_trained = True

        print("\n" + "=" * 70)
        print("✅ ENTRAÎNEMENT TERMINÉ !")
        print("=" * 70)

        final_loss     = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"\n📊 Performances finales :")
        print(f"   • Triplet loss entraînement : {final_loss:.4f}")
        print(f"   • Triplet loss validation   : {final_val_loss:.4f}")

        if final_val_loss < 0.05:
            print(f"\n🏆 EXCELLENT ! Le modèle est très performant.")
        elif final_val_loss < 0.10:
            print(f"\n✅ BON ! Le modèle fonctionne bien.")

        print(f"\n🎯 Le modèle a appris à distinguer :")
        print(f"   ✅ Signatures du même client (cosine similarity élevée)")
        print(f"   ✅ Signatures de clients différents (cosine similarity faible)")
        print(f"   ✅ Vraies signatures vs FRAUDES 🔥")

        print(f"\n💡 Prochaine étape : python base_empreintes.py")

        return history


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT SIAMESE V2 (TRIPLET LOSS + COSINE SIMILARITY)")
    print("=" * 70)

    encoder = SiameseEncoderV2()

    if not encoder.is_trained:
        print("\n⚡ Configuration :")
        print("   • 40 triplets de type A (inter-clients) par client")
        print("   • 20 triplets de type B (vraie/fraude) par client")
        print("   • Triplet loss avec cosine distance (margin=0.2)")
        print("   • 40 epochs")
        print("   • Utilisation des fausses signatures ✅")
        print("")

        history = encoder.train(
            dossier_real="data/real",
            dossier_fake="data/fake",
            epochs=40,
            batch_size=64,
            margin=0.2,
        )
    else:
        print("\n✅ Encodeur V2 déjà entraîné")