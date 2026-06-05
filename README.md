# ARCA: Advanced Recurrent Cognitive Architecture

![Version](https://img.shields.io/badge/version-v2.6.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

ARCA est une architecture hybride de modèle de langage haute performance (wgpu / Rust) qui combine :
- **Un Réservoir Dynamique (LSM)** de 4096 unités pour la dynamique temporelle locale.
- **Une Mémoire Hebbienne Plastique (Bio-Inspired)** avec des couches gérées par contrôle homéostatique.
- **Un mécanisme d'Attention Hybride avec KV-Cache** pour la récupération à long terme (*In-Context Learning*).
- **Un Pipeline GPU "Zero-Sync" Extrême** conçu pour une exécution 100% VRAM-first (WGSL), avec compression FP16 et exécution par lots (Batching).

---

## 🚀 Fonctionnalités Clés (v2.6.0)

ARCA est spécifiquement optimisé pour éliminer le goulot d'étranglement des architectures hybrides : les transferts sur le bus PCIe et la bande passante mémoire.

### ⚡ Inférence "GPU-First" & "Batched"
Lors de l'inférence, **zéro synchronisation CPU-GPU n'a lieu** pour le calcul du graphe de l'encodeur jusqu'à l'échantillonnage.
1. **Pipeline VRAM Intégral** : `Encodeur -> Attention KV-Cache -> Reservoir -> Projections -> Agrégation -> Logits -> Top-K Sampling`.
2. **Exécution Matrice-Matrice (Batching GEMM)** : Le GPU gère `B` séquences d'inférence en parallèle massif.
3. **Zéro Readback** : Les états internes (`s_t`, KV-Cache, mémoire hebbienne) ne quittent jamais la VRAM.
4. **Précision Mixte FP16** : Toutes les matrices statiques (Reservoir, Embeddings, Attention) sont stockées compressées en `f16` et dézippées (`unpack2x16float`) à la volée par les shaders WGSL, doublant virtuellement la Memory Bandwidth.

### 🎲 Échantillonnage Top-P VRAM (Nucleus Sampling)
La sélection probabiliste est assistée par le GPU :
Un shader de réduction multi-passes trouve les `K` meilleurs tokens ou applique un filtrage **Top-P (Nucleus)** et masque les logits. Le CPU ne lit que les tokens finaux (une poignée d'octets) pour appliquer la **Température** et un **Tirage Pondéré (Softmax)** afin de garantir un texte fluide et cohérent.

### 🏋️‍♂️ Entraînement Full-GPU (AdamW & WGSL)
L'entraînement de la tête de prédiction d'ARCA (plus de 55M de paramètres) est calculé à 100% sur la carte vidéo sans blocage CPU.
- **Backpropagation in-place** : L'optimiseur AdamW est implémenté en pur WGSL.
- **Zero-Transfert** : Le calcul des gradients (`grad_logits`, `grad_y_hidden`) et les mises à jour de mémoire se font intégralement sur le GPU.

### ⚡ Tokenizer Multi-threadé (Rayon)
L'apprentissage du vocabulaire via BPE a été optimisé par du calcul distribué (Map-Reduce) sur tous les cœurs du CPU, réduisant drastiquement le temps de scan des gros corpus de texte.

---

## 🛠️ Architecture Hybride

### 1. `ArcaSystem` (Orchestrateur Principal)
L'objet central qui gère le graphe global, le dictionnaire BPE, et les flux de tokens.
```rust
// Exemple d'inférence batch extrême
let token_ids = system.forward_step_extreme_inference(&bytes_batch, &t_batch, &bpe_batch, 0.8);
```

### 2. `GpuInferenceContext` (Vulkan/Metal/DX12)
Moteur wgpu contenant tous les compute pipelines compilés et les buffers persistants FP16 :
- **`encoder.wgsl`** : Hachage des n-grams, lookup BPE et convolution causale.
- **`attention.wgsl`** : Mécanisme d'attention hybride avec gestion dynamique du KV-Cache.
- **`reservoir_update.wgsl`** : Calcule `tanh(R * s_{t-1} + W_in * x)` en matrice-matrice (batch).
- **`projections.wgsl`** : Projections locales pour la mise à jour plastique.
- **`hebbian_plasticity.wgsl`** : Règle de Hebb (outer-product) avec clamp homéostatique.
- **`aggregate.wgsl`** : Combinaison holographique des mémoires.
- **`logit_compute.wgsl`** : Produit scalaire massif sur la table d'embedding.
- **`top_k_sampling.wgsl`** : Réduction K-passes pour l'échantillonnage inférence.

### 3. Contrôleur Métabolique (CPU)
Gère le "climat" (macro-variables) qui module la plasticité selon l'erreur de prédiction : $\beta$ (tension), $\lambda$ (oubli), $\sigma$ (homéostasie).

### 4. API Python via PyO3 / Maturin
ARCA est intégralement compilable et utilisable depuis Python.
```python
import arca
model = arca.ArcaModel("modele.sovereign", "tokenizer.json")
model.train(corpus, "modele.sovereign", 64)
print(model.generate("Texte", max_tokens=50, temperature=0.7, top_p=0.9))
```

---

## 📦 Installation & Compilation

ARCA nécessite **Rust (Edition 2021)** et un GPU compatible avec les backends **wgpu**.

```bash
# Cloner le projet
git clone https://github.com/corbac10099/ARCA.git
cd ARCA

# Compiler avec support GPU natif (Recommandé)
cargo build --release --features gpu

# Lancer les tests unitaires
cargo test --release --features gpu
```

---

*ARCA est conçu pour la recherche en apprentissage hebbien et le calcul hybride (Liquid State Machine + Deep Learning) aux limites du matériel moderne.*
