# DocCornerNet V2 - Proposta `aug_factor` senza filesystem

## 0. Obiettivo

Voglio un training set **virtualmente piu grande** senza:

- generare file augmentati su disco
- duplicare davvero il dataset in RAM
- cambiare le augmentation attuali

Le augmentation devono restare esattamente quelle gia implementate in `v2`:

- `tf_augment_batch()`
- `tf_augment_color_only()`

Quindi il problema non e "aggiungere nuove augmentation". Il problema e:

- far vedere ogni sample piu volte per epoch
- lasciando che ogni visita riceva una nuova trasformazione random on-the-fly

---

## 1. Decisione raccomandata

La soluzione che approvo e questa:

- aggiungere un parametro CLI `--aug_factor N`
- usare un `train_ds.repeat()` infinito
- limitare il numero di batch per epoch con `take(effective_steps_per_epoch)`
- mantenere invariata la logica di augmentation nel training loop

In altre parole:

- non si moltiplica il dataset sul filesystem
- non si materializzano copie
- non si cambiano le funzioni di augmentation
- si moltiplica solo il numero di "visite stochastiche" per epoch

---

## 2. Perche questa soluzione e migliore di `repeat(aug_factor)` puro

La proposta piu ingenua sarebbe:

- `repeat(aug_factor)`
- poi `shuffle(len * aug_factor)`

Non la consiglio come soluzione principale per questi motivi:

1. aumenta inutilmente la semantica del dataset invece di controllare direttamente gli step
2. spinge a gonfiare troppo lo shuffle buffer
3. rende piu opaco il concetto di "epoch"
4. non aggiunge benefici reali rispetto a `repeat() + take()`

La soluzione giusta e piu semplice:

- dataset infinito
- numero di step per epoch controllato esplicitamente

Questo rende il comportamento molto piu chiaro:

- `aug_factor=1` -> comportamento attuale
- `aug_factor=2` -> il modello vede circa 2x batch train per epoch
- `aug_factor=4` -> il modello vede circa 4x batch train per epoch

---

## 3. Semantica corretta di `aug_factor`

`aug_factor` non significa:

- "salvo N copie augmentate"
- "creo N varianti permanenti di ogni immagine"

Significa:

- ogni sample puo essere rivisto piu volte nello stesso epoch
- ogni visita passa di nuovo dentro `tf_augment_batch()` o `tf_augment_color_only()`
- ogni visita riceve una nuova randomizzazione

Quindi il risultato pratico e:

- dataset base identico
- training set effettivo diverso a ogni epoch
- piu diversita stocastica senza overhead disco

---

## 4. Vincolo importante

Per la prima implementazione, `aug_factor > 1` deve essere supportato **solo quando l'augmentation e attiva fin dal primo epoch**.

Quindi:

- `--augment` obbligatorio
- `--aug_start_epoch` deve essere `1`
- `--aug_min_iou` deve essere `0.0`

Motivo:

Se `aug_factor > 1` ma l'augmentation e ancora disattiva, il modello vedrebbe semplicemente piu copie identiche degli stessi sample. Questo:

- non e il comportamento desiderato
- complica inutilmente `steps_per_epoch`
- rende meno pulito il learning-rate schedule

Conclusione:

- `aug_factor > 1` e compatibile solo con augmentation attiva da subito
- eventuale supporto a delayed activation si valuta dopo, non nella prima patch

---

## 5. Implementazione proposta

## 5.1 Nuovo argomento CLI

In `v2/train_ultra.py`:

```python
parser.add_argument(
    "--aug_factor",
    type=int,
    default=1,
    help="Virtual train multiplier via repeated stochastic views per epoch (1 = default)",
)
```

Validazione raccomandata:

- `aug_factor >= 1`
- se `aug_factor > 1`:
  - `args.augment` deve essere `True`
  - `args.aug_start_epoch == 1`
  - `args.aug_min_iou == 0.0`

Se i vincoli non sono rispettati:

- `raise ValueError(...)`

---

## 5.2 `make_tf_dataset()`

La pipeline deve restare leggera.

Versione raccomandata:

```python
def make_tf_dataset(
    images,
    coords,
    has_doc,
    batch_size,
    shuffle,
    image_norm="imagenet",
    drop_remainder=False,
    repeat_forever=False,
):
    ds = tf.data.Dataset.from_tensor_slices((
        images,
        {"coords": coords, "has_doc": has_doc},
    ))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(images), 10000),
            reshuffle_each_iteration=True,
        )

    if repeat_forever:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(
        lambda img, tgt: (_normalize_image(img, image_norm), tgt),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
```

Scelta importante:

- **non** introdurre `repeat(aug_factor)` dentro il dataset builder
- introdurre solo `repeat_forever=True/False`

Motivo:

- il fattore reale si controlla negli step per epoch
- il dataset builder resta semplice

---

## 5.3 Train dataset

Per il train:

```python
repeat_train = args.augment and args.aug_factor > 1

train_ds = make_tf_dataset(
    train_images,
    train_coords,
    train_has_doc,
    args.batch_size,
    shuffle=True,
    image_norm=args.input_norm,
    drop_remainder=True,
    repeat_forever=repeat_train,
)
```

Per validation:

```python
val_ds = make_tf_dataset(
    val_images,
    val_coords,
    val_has_doc,
    args.batch_size,
    shuffle=False,
    image_norm=args.input_norm,
    drop_remainder=False,
    repeat_forever=False,
)
```

Validation e test **non devono mai essere moltiplicati**.

---

## 5.4 Steps per epoch

Qui sta il cuore della feature.

Calcolo corretto:

```python
base_train_steps = len(train_images) // args.batch_size
effective_aug_factor = args.aug_factor if args.augment else 1
train_steps = base_train_steps * effective_aug_factor
val_steps = math.ceil(len(val_images) / args.batch_size)
```

Poi anche il learning rate schedule deve usare `train_steps`:

```python
steps_per_epoch = train_steps
total_steps = steps_per_epoch * args.epochs
warmup_steps = steps_per_epoch * args.warmup_epochs
```

Questo e corretto, perche:

- l'epoch reale ora contiene piu step
- il cosine schedule deve riflettere il nuovo numero di update

---

## 5.5 Training loop

Il training loop non deve cambiare logica di augmentation.

Bisogna solo iterare su un numero fisso di batch per epoch.

Versione raccomandata:

```python
for epoch in range(1, args.epochs + 1):
    epoch_ds = train_ds.take(train_steps) if repeat_train else train_ds

    for images, targets in tqdm(epoch_ds, total=train_steps):
        if aug_active:
            coords = targets["coords"]
            has_doc = targets["has_doc"]
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

Punto chiave:

- le augmentation restano quelle attuali
- cambia solo quante volte il modello vede sample randomizzati per epoch

---

## 6. Logging raccomandato

Nel summary iniziale:

```text
Train images:           32968
Augmentation:           ON
Aug factor:             4x
Base train steps:       128
Effective train steps:  512
Validation steps:       34
```

Se `aug_factor == 1`:

```text
Aug factor:             1x (default)
```

Questo serve per evitare ambiguita nella lettura del training log.

---

## 7. Config e riproducibilita

`config.json` deve includere:

- `augment`
- `aug_factor`
- `rotation_range`
- `scale_range`
- `aug_weak_epochs`
- `aug_start_epoch`
- `aug_min_iou`

Anche se `aug_factor > 1` nella prima versione richiede attivazione immediata, il config va salvato completo.

---

## 8. README

In `v2/README.md` va documentato che:

- `aug_factor` non crea copie fisiche del dataset
- moltiplica solo il numero di viste stochastiche per epoch
- ha senso solo con `--augment`
- nella prima implementazione non va combinato con delayed augmentation

---

## 9. Test che approvo

### Unit test

In `v2/tests/`:

1. `aug_factor=1` -> stessi step del comportamento attuale
2. `repeat_forever=True` + `take(train_steps)` -> numero di batch corretto
3. validation dataset non viene ripetuto
4. `aug_factor>1` senza `--augment` -> errore
5. `aug_factor>1` con `aug_start_epoch>1` -> errore
6. `aug_factor>1` con `aug_min_iou>0` -> errore
7. learning-rate schedule usa gli step effettivi

### Smoke test

Esempio:

```bash
python -m v2.train_ultra \
  --data_root dataset/DocCornerDataset_small \
  --output_dir runs/v2_aug_factor_smoke \
  --epochs 3 \
  --batch_size 8 \
  --augment \
  --aug_factor 2 \
  --rotation_range 5.0 \
  --scale_range 0.0 \
  --aug_start_epoch 1 \
  --aug_min_iou 0.0 \
  --backbone_weights none
```

Verifiche:

- log con `Aug factor: 2x`
- `train_steps` raddoppiati
- validation invariata
- training loop completo senza errori

---

## 10. Cosa NON fare

Per questa feature non approvo:

- scrittura di immagini augmentate su disco
- duplicazione esplicita di array numpy
- `shuffle(len(images) * aug_factor)` aggressivo
- modifica di `tf_augment_batch()` o `tf_augment_color_only()`
- supporto immediato al delayed activation con `aug_factor > 1`

Questa feature deve restare una estensione semplice del trainer, non una nuova pipeline dati.

---

## 11. Comando raccomandato

Esempio conservativo:

```bash
python -m v2.train_ultra \
  --data_root dataset/DocCornerDataset \
  --output_dir runs/v2_full_aug_factor2 \
  --epochs 100 \
  --batch_size 256 \
  --img_size 224 \
  --num_bins 224 \
  --learning_rate 2e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --sigma_px 2.0 \
  --loss_tau 0.5 \
  --w_simcc 1.0 \
  --w_coord 0.2 \
  --w_score 1.0 \
  --label_smoothing 0.0 \
  --backbone_weights none \
  --num_workers 256 \
  --augment \
  --aug_factor 2 \
  --rotation_range 5.0 \
  --scale_range 0.0 \
  --aug_weak_epochs 10 \
  --aug_start_epoch 1 \
  --aug_min_iou 0.0 \
  --init_weights runs/v2_small_final/best_model.weights.h5
```

Per la prima prova, io partirei da:

- `aug_factor=2`

non da `4` o `8`.

---

## 12. Decisione finale

Approvato:

- `aug_factor` come moltiplicatore di viste stochastiche per epoch
- zero filesystem
- augmentation attuali immutate
- implementazione centrata su `train_ultra.py`

Non approvato:

- materializzazione del dataset augmentato
- moltiplicazione pesante del buffer di shuffle
- supporto immediato a tutti i casi edge con activation ritardata

In una riga:

la soluzione giusta e **piu step per epoch su dataset ripetuto all'infinito, con augmentation attuale on-the-fly**, non copie del dataset e non precomputazione su disco.
