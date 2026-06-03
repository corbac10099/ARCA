# ARCA: Advanced Recurrent Cognitive Architecture

![Version](https://img.shields.io/badge/version-v2.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

ARCA est une architecture hybride de modèle de langage haute performance (wgpu / Rust) qui combine :
- **Un Réservoir Dynamique (LSM)** de 4096 unités pour la dynamique temporelle.
- **Une Mémoire Hebbienne Plastique (Bio-Inspired)** avec des couches de matrices 32x32 gérées par contrôle homéostatique.
- **Un Pipeline GPU "Zero-Sync" Extrême** conçu pour minimiser le surcoût CPU (overhead) avec une exécution 100% VRAM-first (WGSL).

---

## 🚀 Fonctionnalités Clés (v2.1.0)

ARCA est spécifiquement optimisé pour éliminer le goulot d'étranglement historique des architectures hybrides : les transferts sur le bus PCIe entre le CPU et le GPU.

### ⚡ Inférence "GPU-First" (Zero-Sync)
Lors de l'inférence (génération), **zéro synchronisation CPU-GPU n'a lieu** pour le calcul du graphe.
1. **Pipeline VRAM Intégral** : `Reservoir -> Projections -> Agrégation -> Logits -> Argmax Sampling`.
2. **Échantillonnage (Sampling) sur GPU** : Un shader de réduction parallèle détermine le "Top-1" token directement sur le GPU. Le seul transfert PCIe est un entier de 4 octets (`u32`) par token généré.
3. **Zéro Readback** : Les états internes du réservoir (`s_t`) et de la mémoire (`M`) ne quittent jamais la VRAM.

### 🏋️‍♂️ Entraînement Haute Performance (Zero-Sync Orchestré)
La boucle d'apprentissage a été entièrement refondue pour offrir une bande passante massive tout en conservant le contrôle côté hôte (Rust) :
- **Orchestration Explicite** : Le CPU orchestre le graphe via des `dispatches` WGSL précis (`dispatch_reservoir`, `dispatch_projections`, `dispatch_hebbian`, `dispatch_aggregate`, `dispatch_logits`) évitant "l'effet boîte noire".
- **Clean Swap Buffers** : L'état `s_t` et les matrices de mémoire Hebbiennes `M` utilisent des "ping-pong buffers" gérés de façon *zero-copy* sur la VRAM.
- **Readback Minimal Stable Point** : Les états `s_t` et les `logits` (nécessaires pour le calcul de perte et le LSH) sont rappatriés en **un seul appel synchronisé**, éliminant les synchronisations par couche.

---

## 🛠️ Architecture

### 1. `ArcaSystem` (Orchestrateur Principal)
L'objet central qui gère le graphe global, le dictionnaire BPE, et les flux de tokens.
```rust
// Exemple d'inférence extrême
let token_id = system.forward_step_extreme_inference(text, t, &bpe_ids);
```

### 2. `GpuContext` / `GpuInferenceContext` (Vulkan/Metal/DX12)
Moteur wgpu contenant tous les compute pipelines compilés et les buffers persistants :
- **`reservoir_update.wgsl`** : Calcule `tanh(R * s_{t-1} + W_in * x_t)` avec *loop unrolling* AMD/RDNA.
- **`projections.wgsl`** : Projections locales (`local_e` et `local_s`) pour la mise à jour plastique.
- **`hebbian_plasticity.wgsl`** : Règle de Hebb (outer-product) avec clamp homéostatique et fatigue.
- **`aggregate.wgsl`** : Combinaison holographique des mémoires en `y_hidden`.
- **`logit_compute.wgsl`** : Produit scalaire massif sur l'embedding table (50 000 x 512).
- **`argmax_reduce.wgsl`** : Réduction pour le sampling inférence.

### 3. Contrôleur Métabolique (CPU)
Gère le "climat" (macro-variables) qui module la plasticité selon l'erreur de prédiction :
- $\beta$ (tension / surprise)
- $\lambda$ (taux d'oubli)
- $\sigma$ (réinitialisation homéostatique)

---

## 📦 Installation & Compilation

ARCA nécessite **Rust (Edition 2021)** et un GPU compatible avec les backends **wgpu** (Vulkan par défaut sous Windows/Linux, Metal sous macOS).

```bash
# Cloner le projet
git clone https://github.com/corbac10099/ARCA.git
cd ARCA

# Compiler avec support GPU natif (Recommandé)
cargo build --release --features gpu

# Lancer les tests unitaires
cargo test --release --features gpu
```

*(Note: Si la feature `gpu` n'est pas activée, ARCA compilera une version "Pure CPU" (fallback) qui utilise `ndarray` de manière séquentielle pour la recherche et le débogage de l'algorithme)*.

---

## 📚 Roadmap / Optimisations Futures

1. **Portage GPU de l'Encoder CPU** : Actuellement, le hachage N-Gram byte-level et les convolutions causales (`encoder.rs`) tournent sur le CPU avant l'envoi de `x_t`. Ce sera le prochain candidat au portage WGSL.
2. **Top-K / Nucleus Sampling en WGSL** : Ajout d'un algorithme de tri (ex: bitonic sort partiel) ou Radix pour étendre le sampling au délà du simple Argmax.
3. **KV-Cache hybride** : Si des capacités de longue mémoire contextuelle (Attention) sont hybridées dans le modèle.

---

*ARCA est conçu pour la recherche en apprentissage hebbien et le calcul hybride (Liquid State Machine + Deep Learning) aux limites du matériel moderne.*
