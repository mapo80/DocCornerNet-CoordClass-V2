# DocCornerNet V2 - Proposta Revisionata per l'Augmentation

## 0. Sintesi della review

La proposta originale individua il problema giusto, ma mescola correzioni necessarie, porting sensati da `v1` e idee troppo complesse per una prima iterazione stabile.

Le conclusioni corrette sono queste:

- nel training `v2` reale l'augmentation oggi e di fatto assente: `make_tf_dataset(..., augment=True)` accetta il flag ma non applica alcuna trasformazione
- `augment_sample()` in `v2/dataset.py` non e solo "parzialmente sbagliata": oggi applica rotazione e scale alle coordinate ma non all'immagine
- `v1` contiene gia la direzione tecnica giusta: batch augmentation TensorFlow, weak augmentation finale, supporto opzionale per outlier statici
- la parte "automatic per-epoch outlier detection" va ridimensionata drasticamente: non e la MVP giusta
- prima va corretta la coerenza della pipeline, poi si valuta il resto

Conclusione: la `v2` ha bisogno prima di tutto di una augmentation reale, coerente e integrata nel training loop. Il mining automatico degli outlier dalla validation va tenuto fuori dalla prima implementazione.

---

## 1. Stato reale del repository

### 1.1 Cosa fa oggi `v2`

- `v2/train_ultra.py` carica immagini e label in numpy, poi `make_tf_dataset()` normalizza, batcha e fa prefetch
- il parametro `augment` di `make_tf_dataset()` oggi non ha alcun effetto
- `v2/train_ultra.py` non usa `create_dataset()` di `v2/dataset.py`, quindi `augment_sample()` non entra nel training path reale
- `v2/dataset.py::augment_sample()` applica augmentation fotometriche via PIL, ma la parte geometrica e incoerente:
  - rotazione: modifica solo le coordinate
  - scale: modifica solo le coordinate
  - horizontal flip: assente
- `DEFAULT_AUG_CONFIG` contiene `translate` e `perspective`, ma in `v2/dataset.py` non esiste codice che li applichi davvero

### 1.2 Cosa esiste gia in `v1` e vale la pena portare

`v1` ha gia una base piu solida:

- `tf_augment_batch()`
- `_tf_rotate_batch()`
- `_tf_scale_batch()`
- `tf_augment_color_only()`
- `load_outlier_list()`
- weak augmentation negli ultimi epoch
- supporto a outlier statici e weighted sampling

Il punto non e inventare una nuova pipeline. Il punto e portare in `v2` la parte utile, senza importare complessita inutile.

---

## 2. Correzioni alla proposta originale

| Claim originale | Valutazione | Correzione |
|---|---|---|
| "`v2` train gira con zero augmentation" | Sostanzialmente corretto | E vero per il training path reale di `v2/train_ultra.py`. Non e corretto dire che tutta `v2` non abbia augmentation: `augment_sample()` esiste, ma oggi e scollegata dal trainer. |
| "Bug: rotation applies to coordinates only, not the image" | Corretto ma incompleto | Il problema vale anche per `scale`. Oggi tutta la geometria di `augment_sample()` non e affidabile. |
| "Portiamo tutto il sistema `v1`" | Troppo largo | Da portare subito: batch augmentation TF e weak augmentation finale. Da valutare dopo: outlier statici. Da rimandare: auto-mining outlier per epoch. |
| "Automatic per-epoch outlier detection" | Non approvato come MVP | Introduce coupling train/validation, assunzioni fragili sui filename e troppa complessita per il primo pass. |
| "`validate_quadrilateral()` come parte centrale" | Priorita bassa | Utile al massimo in test o debug. Non e il blocco principale per rimettere in sesto la pipeline. |

---

## 3. Decisione raccomandata

### 3.1 MVP da implementare subito

1. Correggere il fatto che il training `v2` non applichi augmentation reali.
2. Portare in `v2` una batch augmentation TensorFlow sul modello di `v1`.
3. Correggere o ridurre `augment_sample()` in modo che non possa piu produrre geometria incoerente.
4. Tenere `validation` e `test` senza augmentation.
5. Validare prima su `dataset/DocCornerDataset_small`, poi su `dataset/DocCornerDataset`.

### 3.2 Cosa NON fare nel primo pass

- niente outlier detection automatica per epoch
- niente mapping da validation filenames a train filenames
- niente dataset rebuild ad ogni epoch guidato dai risultati di validation
- niente perspective augmentation
- niente controlli runtime invasivi tipo `validate_quadrilateral()` nel path caldo
- niente aumento del numero di CLI arguments oltre lo stretto necessario

---

## 4. Proposta tecnica raccomandata

### 4.1 Strategia generale

La soluzione giusta non e aggiungere una nuova pipeline parallela. La soluzione giusta e:

- mantenere il caricamento attuale di `v2/train_ultra.py`
- applicare augmentation TensorFlow batch-wise nel loop di training
- lasciare `make_tf_dataset()` come builder del dataset, non come finto punto di augmentation
- usare `augment_sample()` solo per debug, offline augmentation o visualization, mai come base del training serio se la parte geometrica non e coerente

### 4.2 `tf_augment_batch()` per `v2`

Implementare in `v2/dataset.py` una funzione sul modello di `v1`:

```python
def tf_augment_batch(
    images,
    coords,
    has_doc,
    img_size=224,
    image_norm="imagenet",
    rotation_range=5.0,
    scale_range=0.0,
):
    ...
```

Scelte raccomandate:

- input: immagini gia normalizzate `float32`
- photometric augmentations su tutti i sample:
  - brightness
  - contrast
  - saturation
- geometric augmentations solo sui sample positivi:
  - horizontal flip
  - rotation
  - scale opzionale
- clip dei pixel in base a `image_norm`, come in `v1`
- coordinate sempre clippate in `[0, 1]`

Motivazione:

- riusa una strategia gia difendibile nel repo
- evita `tf.py_function` e PIL nel training hot path
- genera augmentation fresca ad ogni step
- non richiede una riscrittura completa del loader

### 4.3 Geometriche: cosa includere davvero

#### Sempre dentro la MVP

- horizontal flip 50%
- rotation fino a `+-5 deg`

Queste due servono subito. Sono gia allineate con `v1` e attaccano il caso d'uso giusto senza introdurre troppi failure mode.

#### Da supportare ma con default prudente

- scale tramite `crop_and_resize`

Raccomandazione critica:

- implementarla pure, perche esiste gia il precedente `v1`
- ma partire con `scale_range=0.0` come default in `v2`
- abilitare `0.1` o `0.15` solo dopo smoke test su `DocCornerDataset_small`

Motivo:

- la scale e utile, ma e anche la trasformazione geometrica piu facile da rendere distruttiva se il documento e gia vicino ai bordi
- il fatto che la proposta originale la presenti come default implicito e troppo assertivo

#### Fuori dalla MVP

- perspective
- translate esplicito
- gaussian blur nel training TF path

`perspective` e `translate` oggi compaiono nel config ma non esistono davvero in `v2`. Non vanno raccontati come se fossero quasi pronti. Se restano nel config, vanno marcati come inutilizzati o rimossi.

---

## 5. Correzione di `augment_sample()`

La proposta originale sottostima il problema: non basta ruotare anche l'immagine. Bisogna allineare tutta la parte geometrica.

### 5.1 Stato attuale non accettabile

Oggi `augment_sample()`:

- ruota le coordinate ma non l'immagine
- scala le coordinate ma non l'immagine
- non supporta flip orizzontale
- quindi puo generare sample con immagine e label in disaccordo

### 5.2 Correzione minima accettabile

Ci sono solo due opzioni sensate:

#### Opzione A - Fix completo

- ruotare immagine e coordinate insieme
- scalare immagine e coordinate insieme
- aggiungere horizontal flip coerente
- mantenere `augment_sample()` come utility offline

#### Opzione B - Riduzione di scope

- lasciare in `augment_sample()` solo photometric augmentation
- rimuovere temporaneamente la geometria dal path PIL
- spostare tutta la geometria nel solo `tf_augment_batch()`

Giudizio critico:

- per ridurre il rischio, l'Opzione B e la piu robusta
- l'Opzione A ha senso solo se vuoi davvero usare `create_dataset(..., augment=True)` fuori dal trainer principale

---

## 6. Weak augmentation finale

La weak augmentation finale di `v1` e una buona idea, ma non va messa davanti alla correzione base della pipeline.

Raccomandazione:

- implementare `tf_augment_color_only()` in `v2`
- aggiungere `--aug_weak_epochs`
- usarla solo dopo che la full augmentation base e stabile

Razionale:

- la logica e semplice
- l'idea e gia presente in `v1`
- puo aiutare il refinement finale della precisione
- ma non sostituisce il lavoro principale: far funzionare davvero l'augmentation durante il training

---

## 7. Outlier strategy: cosa tenere e cosa no

### 7.1 Cosa ha senso

Se il progetto vuole recuperare la robustezza di `v1`, la prima cosa sensata da portare e la strategia semplice:

- outlier list statica
- stronger photometric augmentation opzionale
- weighted sampling opzionale

Questa e una strategia gia presente nel repo e relativamente facile da spiegare, misurare e disattivare.

### 7.2 Cosa NON approvo

La proposta di:

- attivare outlier augmentation quando la validation IoU supera una soglia
- identificare outlier ogni epoch sulla validation
- mappare quei filename sul train
- ricostruire il dataset ogni epoch con oversampling

non e una buona MVP per questi motivi:

1. usa la validation come segnale operativo per cambiare il training data distribution
2. indebolisce il ruolo della validation come benchmark pulito di model selection
3. l'assunzione "training samples are a superset" non e dimostrata
4. il mapping per filename tra validation e train puo essere semplicemente falso
5. la complessita cresce molto piu del probabile guadagno

Conclusione netta:

- static outlier support: approvabile come fase 2
- auto-mining outlier da validation: backlog, non MVP

---

## 8. Integrazione nel training loop

### 8.1 Scelta raccomandata

Integrare augmentation nel loop di training di `v2/train_ultra.py`, non in `make_tf_dataset()`.

Schema:

```python
for images, targets in train_ds:
    coords = targets["coords"]
    has_doc = targets["has_doc"]

    if args.augment:
        if use_weak_aug:
            images = tf_augment_color_only(images, image_norm=args.input_norm)
        else:
            images, coords = tf_augment_batch(
                images,
                coords,
                has_doc,
                img_size=args.img_size,
                image_norm=args.input_norm,
                rotation_range=args.rotation_range,
                scale_range=args.scale_range,
            )
        targets = {"coords": coords, "has_doc": has_doc}

    trainer.train_step((images, targets))
```

### 8.2 Nota importante su `make_tf_dataset()`

Oggi il parametro `augment` e fuorviante.

Scelte corrette:

- o viene rimosso
- oppure viene tenuto ma documentato come flag di alto livello usato fuori dalla funzione

La situazione attuale, in cui il flag esiste ma non fa nulla, va corretta esplicitamente.

---

## 9. CLI minima raccomandata

Per il primo pass bastano pochi argomenti:

```python
parser.add_argument("--augment", action="store_true")
parser.add_argument("--rotation_range", type=float, default=5.0)
parser.add_argument("--scale_range", type=float, default=0.0)
parser.add_argument("--aug_weak_epochs", type=int, default=0)
```

Eventuale fase 2:

```python
parser.add_argument("--outlier_list", type=str, default=None)
parser.add_argument("--outlier_weight", type=float, default=1.0)
```

Non servono 11 nuovi flag per rimettere in piedi la pipeline base.

---

## 10. File da modificare

| File | Modifica raccomandata |
|---|---|
| `v2/dataset.py` | Portare `tf_augment_batch()`, `_tf_rotate_batch()`, opzionalmente `_tf_scale_batch()`, `tf_augment_color_only()`. Correggere o ridurre `augment_sample()`. Rimuovere o marcare come non usati `translate` e `perspective`. |
| `v2/train_ultra.py` | Aggiungere i pochi CLI args necessari. Applicare augmentation nel training loop reale. Correggere il significato di `augment` in `make_tf_dataset()`. |
| `v2/tests/test_dataset.py` | Aggiornare i test di `augment_sample()` per impedire mismatch immagine/coordinate. |
| `v2/tests/test_train_eval_export.py` | Aggiungere smoke/integration tests per il training loop con augmentation attiva. |
| `v2/tests/test_augmentation.py` | Nuovo file consigliato per testare batch augmentation TF in isolamento. |

---

## 11. Test plan minimo serio

Target: `>= 90%` coverage sulla logica nuova o modificata.

Test che servono davvero:

1. `tf_augment_batch` preserva shape e dtype
2. le coordinate restano in `[0, 1]`
3. i sample negativi non subiscono trasformazioni geometriche sulle coordinate
4. horizontal flip rimappa correttamente `TL, TR, BR, BL`
5. `rotation_range=0` e identity
6. `scale_range=0` e identity
7. `tf_augment_color_only()` non modifica le coordinate
8. `augment_sample()` non puo piu produrre geometria incoerente
9. il training loop usa davvero augmentation quando `args.augment` e attivo
10. validation e test restano senza augmentation

Non serve partire con una matrice di 29 test prima ancora di aver fissato il design. Serve prima coprire i contratti corretti.

---

## 12. Piano di rollout

### Step 1

- batch TF augmentation base
- fix o riduzione di `augment_sample()`
- flip + rotation
- scale supportato ma disattivato di default

### Step 2

- weak augmentation finale
- ablation su `scale_range` (`0.0`, `0.1`, `0.15`) prima su `DocCornerDataset_small`, poi su `DocCornerDataset`

### Step 3

- solo se emerge un bisogno reale: outlier list statica + weighted sampling

### Backlog esplicito

- auto-mining outlier per epoch
- validation-driven dataset rebuild
- perspective augmentation
- runtime quadrilateral validator

---

## 13. Decisione finale

La direzione giusta e piu piccola della proposta originale.

Approvato:

- port della batch augmentation TF di `v1`
- fix della geometria incoerente in `augment_sample()`
- weak augmentation finale come estensione a basso rischio
- eventuale supporto a outlier statici in una fase successiva

Non approvato come MVP:

- sistema automatico di outlier detection e oversampling guidato dalla validation
- logica extra non ancora supportata dal codice (`perspective`, `translate`)
- espansione del piano di test oltre il necessario prima di stabilizzare il design
