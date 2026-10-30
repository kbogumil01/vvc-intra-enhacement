## 1. Neuronowe przetwarzanie końcowe dla VVC — kod źródłowy

Kod źródłowy do pracy magisterskiej dotyczącej redukcji artefaktów kompresji
VVC (H.266) za pomocą splotowych sieci neuronowych typu *post-processing*.
Bazowa architektura: **VVC-PPFF** (progresywna fuzja cech, ~2,78 mln parametrów),
trenowana w trybie **All-Intra** z wykorzystaniem map partycji i kontekstu
temporalnego, a następnie dostrajana adwersarialnie (RaGAN).

Wszystkie eksperymenty przeprowadzono na superkomputerze **Athena** (ACK Cyfronet
AGH): NVIDIA A100 40 GB, SLURM, Python 3.13, CUDA 12.4.

---

## 2. Struktura katalogu

```
zip/
├── README.md                 ← ten plik (przewodnik reprodukcji)
├── requirements.txt          ← zależności Pythona (z wersjami)
├── src/                      ← kod źródłowy (wszystkie moduły — patrz §3)
├── slurm/                    ← skrypty zadań SLURM, pogrupowane:
│   ├── encode/               ← kodowanie danych koderem VTM
│   ├── preprocess/           ← mapy partycji + bufory memmap
│   ├── train/                ← treningi (bazowy, partycje, temporal, GAN)
│   └── eval/                 ← ewaluacja BD-Rate / LPIPS / figury
└── results/                  ← wyniki referencyjne (BD-Rate CSV) + podział zbioru
```

> **Ważne:** moduły w `src/` korzystają z płaskich importów (`import model_v2`,
> `from dataset import ...`, `importlib.import_module("vvc-ppff")`). Wszystkie
> pliki `.py` muszą pozostać w jednym katalogu. Skrypty uruchamiaj z wnętrza `src/`
> (albo dodaj `src/` do `PYTHONPATH`).

---


## 3. Katalog plików źródłowych (`src/`)

### 3.1. Modele i rejestr

| Plik | Opis |
|------|------|
| `registry.py`      | Rejestr modeli (wzorzec słownikowy). `create_model(name, **kw)` tworzy sieć po nazwie. |
| `vvc-ppff.py`      | **Architektura VVC-PPFF** (`ppff_base`) — generator i `PPFFLightningModule`. |
| `vvc_ppff_gan.py`  | Moduł GAN: generator PPFF + dyskryminator (spectral norm), strata RaGAN, strata percepcyjna VGG, regularyzacja R1. |
| `model_v2.py`      | Architektura DenseNet (`dense_*`) + funkcje metryk `compute_psnr`, `compute_y_psnr` (używane też przez PPFF). |
| `model_wcdann.py`  | Architektura WCDANN zaadaptowana do YUV (`wcdann_yuv`) — porównanie literaturowe. |

### 3.2. Dane i parsery

| Plik | Opis |
|------|------|
| `dataset.py`                   | Zbiory danych (`VVCFrameDataset`, `VVCMmapDataset`), budowa rekordów, podział train/val po sekwencjach. |
| `dataset_temporal.py`          | Zbiór danych z kontekstem temporalnym (`VVCTemporalMmapDataset`). |
| `encode_nofilter.py`           | Kodowanie VTM **bez filtrów pętlowych** (równoległe, `multiprocessing`). |
| `encode_dbonly.py`             | Kodowanie VTM **tylko z filtrem deblokującym**. |
| `csv_to_partition_maps.py`     | **Parser tracingu VTM** (CSV `BlockStat`) → 4 mapy rastrowe (głębokość + granice, luma/chroma). Normalizacja per-klatka przez `max()`. |
| `preprocess_memmap.py`         | Konwersja klatek YUV → bufor `memmap` (szybki trening bez I/O). |
| `preprocess_partition_memmap.py` | Bufor memmap z dołączonymi mapami partycji. |
| `preprocess_temporal_memmap.py`  | Bufor memmap z klatkami sąsiednimi (temporal). |
| `prepare_temporal_local.py`    | Przygotowanie lokalnych danych temporalnych (klatki sąsiednie). |
| `verify_partition_data.py`     | Walidacja poprawności wygenerowanych map partycji. |

### 3.3. Trening

| Plik | Opis |
|------|------|
| `train.py`              | Główny skrypt treningowy (bazowy + partycje). Parametryzowany `--model`. |
| `train_temporal.py`     | Trening modelu temporalnego (9/12 kanałów, `--temporal-mode`). |
| `train_gan.py`          | Dostrajanie adwersarialne (RaGAN/LSGAN) pretrenowanego generatora. |
| `train_discriminator.py`| Opcjonalny pretrening samego dyskryminatora. |

### 3.4. Ewaluacja

| Plik | Opis |
|------|------|
| `evaluate_bd.py`             | BD-Rate / BD-PSNR (PSNR i SSIM) per kanał (`bd_rate`, `bd_psnr`). Kafelkowanie klatek. |
| `evaluate_bd_temporal_ctc.py`| BD-Rate modelu temporalnego na CTC; `--anchor-decoded-root` dla porównania krzyżowego (anchor = VTM z filtrami). |
| `evaluate_bd_crossref.py`    | Porównanie względem wspólnego punktu odniesienia (VTM bez filtrów). |
| `evaluate_bd_gan.py`         | Ewaluacja modelu GAN. |
| `eval_lpips.py`              | Metryka percepcyjna LPIPS (baseline vs MSE vs GAN). |
| `compare_results.py`        | Zbiorcze porównanie plików wynikowych. |

### 3.5. Wizualizacje (figury do pracy)

`generate_comparison.py`, `generate_thesis_figures.py`, `visualize_artifacts.py`,
`visualize_partition*.py`, `visualize_patch_extraction.py`, `visualize_temporal.py`.

---

## 4. Potok reprodukcji (od zera)

Zmienną `DATA` ustaw na katalog z danymi źródłowymi (sekwencje YUV 4:2:0 10-bit).

### Krok 1 — Kodowanie danych koderem VTM

Wymaga skompilowanego **VTM 22.2** (GCC 13.3). Do map partycji potrzebny jest build
z flagą `ENABLE_TRACING` (patrz `slurm/preprocess/build_vtm_trace_slurm.sh`).

```bash
sbatch slurm/encode/encode_nofilter_ctc_slurm.sh   # dane testowe CTC, bez filtrów
sbatch slurm/encode/encode_nofilter_bvi_slurm.sh   # dane treningowe BVI
sbatch slurm/encode/encode_dbonly_ctc_slurm.sh     # wariant tylko-DB
```

Konfiguracja kodera: `encoder_intra_vtm.cfg` z wyłączonymi filtrami
(`LoopFilterDisable=1`, `SAO=0`, `ALF=0`). Strumień bitowy jest identyczny jak
z filtrami (w All-Intra filtry nie wpływają na przepływność binarną).

### Krok 2 — Mapy partycji i bufory memmap

```bash
sbatch slurm/preprocess/csv_to_maps_slurm.sh          # tracing VTM → mapy partycji
sbatch slurm/preprocess/preprocess_nofilter_slurm.sh  # YUV → memmap (bazowy)
sbatch slurm/preprocess/preprocess_partition_slurm.sh # + mapy partycji
sbatch slurm/preprocess/run_temporal_memmap.sh        # + klatki sąsiednie (temporal)
```

### Krok 3 — Trening

```bash
# Model bazowy (4 kan.) — główny punkt odniesienia:
sbatch slurm/train/train_ppff_nofilter_slurm.sh
# Warianty filtrów:
sbatch slurm/train/train_ppff_slurm.sh            # pełne filtry
sbatch slurm/train/train_ppff_dbonly_slurm.sh     # tylko DB
# Mapy partycji (8 kan.) + ablacja pojedynczych kanałów:
sbatch slurm/train/train_ppff_partition_nofilter_slurm.sh
sbatch slurm/train/train_ablation_nofilter_slurm.sh
# Model temporalny (12 kan.) — najlepszy wynik:
sbatch slurm/train/run_temporal_train.sh
# Ablacja kierunku (9 kan.): uruchom z --temporal-mode prev_only / next_only
# Dostrajanie adwersarialne (RaGAN):
sbatch slurm/train/train_gan_ragan_slurm.sh
```

Przykładowe wywołania (poglądowo — pełne w skryptach SLURM):

```bash
# bazowy nofilter
python train.py --model ppff_base \
  --decoded-root  "$DATA/decoded_BVI_frames_YUV420_10bit_nofilter" \
  --original-root "$DATA/splitted_DVI_BVC" \
  --memmap-dir    "$DATA/memmap_cache_nofilter" \
  --patch-size 256 --batch-size 12 --accumulate-grad-batches 3 \
  --lr 1e-4 --max-epochs 200 --early-stop-patience 20 --precision 16-mixed

# temporalny 12 kanałów
python train_temporal.py --model ppff_base --in-channels 12 \
  --memmap-dir "$DATA/temporal_memmap_nofilter" \
  --patch-size 256 --batch-size 12 --accumulate-grad-batches 3 \
  --lr 1e-4 --max-epochs 200 --early-stop-patience 20 --precision 16-mixed

# GAN (RaGAN) — dostrojenie pretrenowanego generatora
python train_gan.py --generator-model ppff_base \
  --generator-ckpt ../checkpoints/ppff_temporal_12ch_both/model.ckpt \
  --temporal-memmap-dir "$DATA/temporal_memmap_nofilter" \
  --gan-loss-type ragan --lambda-pixel 1.0 --lambda-perceptual 0.01 \
  --lambda-adv 0.005 --lambda-r1 10.0 --lr-g 1e-5 --lr-d 1e-4
```

### Krok 4 — Ewaluacja

```bash
sbatch slurm/eval/eval_all_ctc_slurm.sh           # BD-Rate modeli bazowych na CTC
sbatch slurm/eval/eval_temporal_ctc_slurm.sh      # BD-Rate modelu temporalnego
sbatch slurm/eval/eval_crossref_temporal_slurm.sh # porównanie z VTM+filtry (anchor)
sbatch slurm/eval/eval_ragan_slurm.sh             # BD-Rate modelu GAN
sbatch slurm/eval/eval_lpips_slurm.sh             # LPIPS (baseline/MSE/GAN)
```

---

## 5. Hiperparametry (wspólne)

| Parametr | Wartość |
|----------|---------|
| Optymalizator | Adam (β₁=0,9, β₂=0,999) |
| Tempo uczenia | 1e-4 (GAN: G 1e-5, D 1e-4) |
| Rozmiar wsadu | 12 × akumulacja 3 = 36 wycinków |
| Rozmiar wycinka | 256 × 256 |
| Precyzja | 16-mixed (`float16`) |
| Maks. epok | 200 (early-stop, patience 20) |
| Strata | MSE (GAN: MSE + 0,005·RaGAN + 0,01·VGG) |
| QP | {22, 27, 32, 37} |

Podział train/val (seed 42, 15 % walidacji, po sekwencjach) zapisany w
`results/dataset_split_info.json` — kluczowy dla wiernej reprodukcji.

---

## 6. Mapa: wyniki ↔ modele ↔ tabele w pracy

| Model (checkpoint) | Wynik (CSV) | Tabela w pracy |
|--------------------|-------------|----------------|
| `ppff_filter_4ch`             | `bd_results_ctc_v92_filter.csv`          | tab. filtrów / porównanie architektur |
| `ppff_nofilter_4ch`           | `bd_results_ctc_v93_nofilter.csv`        | tab. filtrów (bazowy) |
| `ppff_dbonly_4ch`             | `bd_results_ctc_v98_dbonly.csv`          | tab. filtrów |
| `ppff_nofilter_partition_8ch` | `bd_results_ctc_v99_nofilter_part.csv`   | tab. map partycji |
| `ppff_filter_partition_8ch`   | `bd_results_ctc_v100_filter_part.csv`    | tab. map partycji |
| `ppff_dbonly_partition_8ch`   | `bd_results_ctc_v101_dbonly_part.csv`    | tab. map partycji |
| `ppff_temporal_12ch_both`     | `bd_results_temporal_ctc.csv`            | tab. kontekstu temporalnego (główny wynik) |
| `ppff_temporal_9ch_prev_only` | `bd_results_temporal_ctc_prev_only.csv`  | ablacja kierunku |
| `ppff_temporal_9ch_next_only` | `bd_results_temporal_ctc_next_only.csv`  | ablacja kierunku |
| `ppff_gan_ragan_temporal_12ch`| `bd_results_ctc_ragan_v2784784.csv`      | tab. GAN (PSNR/SSIM) |
| (temporal vs VTM+filtry)      | `bd_results_temporal_crossref_filter_anchor.csv` | porównanie z literaturą |
| (LPIPS)                       | `lpips_comparison.csv`                   | tab. LPIPS |


---

## 7. Uwagi końcowe

- Skrypty SLURM zawierają ścieżki klastra Athena
  (`/net/tscratch/people/plgkarolbogumil/...`) — przed uruchomieniem dostosuj
  zmienną `DATA` oraz `--account`/`--partition` do własnego środowiska.
- Punkty kontrolne zapisywane są przez PyTorch Lightning z prefiksem `model.` /
  `generator.`; skrypty ewaluacyjne usuwają go automatycznie.
