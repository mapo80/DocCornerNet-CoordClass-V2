# Augmentation selettiva per outlier dinamica - proposta e review

## 0. Scopo del documento

Questo file raccoglie:

- la proposta di augmentation geometrica applicata solo agli outlier, identificati dinamicamente durante il training
- le mie osservazioni critiche
- una versione rivista della proposta, piu prudente e piu difendibile

L'obiettivo non e bocciare l'idea in assoluto. L'obiettivo e capire se questa strategia sia una buona prima implementazione per `v2`, oppure solo una ablation avanzata da tenere dietro feature flag.

---

## 1. Proposta ricevuta

### 1.1 Contesto

Il modello presenta outlier, cioe campioni con IoU basso, che degradano le metriche. L'idea e controllare se l'augmentation geometrica debba essere applicata:

- a tutti i campioni
- oppure solo agli outlier

Gli outlier devono essere identificati dinamicamente durante il training, senza file esterni.

### 1.2 Idea centrale

Dopo ogni epoch di validazione:

1. si esegue un forward pass rapido sul training set
2. si identificano come outlier i campioni con `IoU < soglia`
3. la geometria (`flip`, `rotation`, `scale`) viene applicata solo a quei campioni nel ciclo successivo
4. la fotometria (`brightness`, `contrast`, `saturation`) resta attiva per tutti

Primo epoch:

- nessun outlier ancora noto
- fallback: augmentation geometrica su tutti

### 1.3 Modifiche proposte

#### `v2/metrics.py`

Aggiungere `return_per_sample=True` a `ValidationMetrics.compute()` per restituire:

- `per_sample_ious`
- `per_sample_mask`

#### `v2/dataset.py`

Estendere `tf_augment_batch()` con `aug_mask` opzionale:

- `1.0` = applica geometria
- `0.0` = salta la geometria

Mascherare:

- horizontal flip
- rotation
- scale

La fotometria resta sempre attiva per tutti.

#### `v2/train_ultra.py`

Aggiungere:

- `--aug_target` con valori `all | outliers`
- `--outlier_iou_threshold`

Poi:

- aggiungere `is_outlier` ai target del dataset
- introdurre una helper `identify_training_outliers(...)`
- dopo la validazione, fare un pass sul train set per ricostruire la mask
- ricostruire `train_ds` con la nuova informazione di outlier

### 1.4 Test proposti

- `aug_mask=zeros` -> nessuna geometria
- `aug_mask=ones` -> comportamento identico a prima
- `aug_mask` misto
- `return_per_sample=True` in `metrics.py`
- smoke test di `identify_training_outliers()`

---

## 2. Valutazione critica

## 2.1 Cosa c'e di buono

- E molto meglio della vecchia idea "validation-driven" con mapping da validation a train tramite filename.
- L'idea di separare fotometria e geometria ha un razionale tecnico.
- `aug_mask` dentro `tf_augment_batch()` e una modifica localizzata e relativamente pulita.
- L'estensione di `ValidationMetrics.compute()` con ritorno per-sample e tecnicamente fattibile, perche la classe accumula gia predizioni e ground truth.

## 2.2 Problemi principali

### Problema 1 - Overhead troppo alto per essere la prima scelta

Il costo non e solo il forward pass extra sul train set.

Nel codice `v2` attuale, ricostruire `train_ds` non e gratis:

- `make_tf_dataset()` riconverte e rinormalizza tutte le immagini
- la ricostruzione ad ogni epoch aggiunge churn CPU, RAM e tempo

Quindi il piano costa:

- un'inferenza extra su tutto il train set
- una ricostruzione del dataset a ogni epoch

Come prima implementazione, e troppo pesante.

### Problema 2 - Feedback loop potenzialmente instabile

La regola proposta e:

- i sample sbagliati di piu ricevono piu augmentation geometrica

Questo non e automaticamente un bene. Se un campione e:

- rumoroso
- ambiguo
- annotato male
- intrinsecamente difficile

renderlo ancora piu difficile puo peggiorare il training invece di aiutarlo.

In altre parole:

- gli outlier veri e utili potrebbero beneficiare della strategia
- i falsi outlier potrebbero essere rinforzati

### Problema 3 - Soglia fissa `IoU < 0.90` troppo rigida

Una soglia fissa come default non mi convince:

- a inizio training rischi di flaggare quasi tutto
- piu avanti rischi oscillazioni forti da un epoch all'altro

Il sistema rischia di collassare alternativamente in:

- "augmenta tutti"
- "augmenta una mask instabile"

### Problema 4 - Cambio di distribuzione troppo brusco

Il fallback:

- epoch 1: geometria su tutti
- epoch 2+: geometria solo sugli outlier

e un cambio troppo brusco.

Se si vuole davvero provare questa strategia, serve una transizione piu morbida:

- warmup iniziale
- aggiornamento non ad ogni epoch
- probabilita ridotta per i non-outlier, non zero assoluto

### Problema 5 - Sta diventando una politica di curriculum, non solo augmentation

Questa proposta non e piu solo "augmentation selettiva".
Di fatto introduce una policy di curriculum dinamico guidata dalle predizioni del modello.

Questa e una idea legittima, ma:

- e piu complessa di quanto sembri
- non dovrebbe essere presentata come semplice estensione della pipeline base

---

## 3. Giudizio netto

Non la approverei come direzione principale della `v2`.

La approverei solo come:

- ablation avanzata
- esperimento opzionale
- feature flag esplicita

Se oggi il progetto deve scegliere dove investire prima, l'ordine corretto resta:

1. augmentation base corretta per tutti
2. baseline stabile su `DocCornerDataset_small`
3. baseline stabile su `DocCornerDataset`
4. solo dopo, test della variante `outliers-only geometric augmentation`

Quindi:

- idea interessante: si
- priorita immediata: no
- default di training: no

---

## 4. Versione rivista consigliata

Se si vuole mantenere viva questa idea, la versione che consiglierei e molto piu prudente.

### 4.1 Principio

Trattarla come esperimento controllato, non come nuova policy standard.

### 4.2 Regole raccomandate

#### Warmup iniziale

Per i primi `5-10` epoch:

- augmentation geometrica su tutti i campioni positivi
- nessuna selezione dinamica per outlier

Razionale:

- evitare che la mask venga costruita quando il modello e ancora troppo instabile

#### Aggiornamento non ad ogni epoch

Identificare gli outlier ogni `3-5` epoch, non ad ogni epoch.

Razionale:

- meno overhead
- meno oscillazioni
- meno churn nella ricostruzione del dataset

#### Niente soglia rigida come unico criterio

Meglio una di queste due strategie:

- top `q%` dei sample peggiori
- media mobile / EMA dello score di difficolta

Se proprio si tiene una soglia:

- non usarla come unico default
- accompagnarla con cap minimo/massimo sulla percentuale di outlier

#### Non azzerare la geometria per i non-outlier

Meglio:

- outlier: probabilita alta di geometria
- non-outlier: probabilita ridotta ma non zero

Per esempio:

- outlier -> `p_geom = 1.0`
- non-outlier -> `p_geom = 0.25` o `0.50`

Questo evita un salto troppo brusco nella distribuzione dei batch.

#### Tenere la feature dietro flag sperimentale

La feature va presentata come:

- `experimental`
- non default
- da confrontare contro `aug_target=all`

---

## 5. Modifiche consigliate al piano originale

### 5.1 `v2/metrics.py`

La modifica proposta e accettabile:

- `compute(return_per_sample=True)` va bene

Ma la userei solo per esperimenti specifici.

Se la feature resta sperimentale, e corretto che anche il supporto per-sample resti minimale e ben delimitato.

### 5.2 `v2/dataset.py`

`aug_mask` va bene, ma lo farei con semantica chiara:

- controlla solo la geometria
- non tocca la fotometria

Questo e coerente con l'idea originale.

Inoltre:

- flip deve essere mascherato sia per immagine sia per coordinate
- rotation e scale devono diventare identita per i sample non selezionati

### 5.3 `v2/train_ultra.py`

Qui renderei il piano piu conservativo:

- aggiungere il flag sperimentale
- warmup iniziale
- aggiornamento mask ogni `K` epoch
- no ricostruzione aggressiva senza misurare l'overhead reale

Se possibile, la ricostruzione del dataset andrebbe resa piu economica prima di introdurre questa feature. Nel codice `v2` attuale e troppo facile pagare il costo di normalizzazione e rebuild ad ogni giro.

---

## 6. Piano operativo che approverei

### Fase A - Baseline obbligatoria

- augmentation base per tutti
- fotometria per tutti
- geometria per tutti i positivi
- nessuna selezione dinamica

### Fase B - Ablation sperimentale

Attivata solo con flag esplicito.

Configurazione minima:

- warmup `5-10` epoch
- refresh outlier ogni `3-5` epoch
- top `q%` oppure threshold con cap
- geometria ridotta per i non-outlier, non zero

### Fase C - Go / no-go

La feature si tiene solo se migliora davvero:

- `mean_iou`
- `recall_90`
- `recall_95`
- tail metrics

senza un overhead sproporzionato.

Se il vantaggio e marginale, va rimossa.

---

## 7. Raccomandazione finale

La proposta e interessante, ma non la venderei come "prossimo passo naturale".

La formulazione giusta e:

- buona idea da testare
- pessima idea da rendere default subito

La mia raccomandazione finale e:

- tenere `aug_target=all` come baseline principale
- implementare la variante dinamica solo come esperimento
- non usare update per-epoch
- non usare soglia fissa `0.90` come policy universale
- non spegnere del tutto la geometria per i non-outlier

In una riga:

questa proposta ha valore come ablation di curriculum dinamico, non come sostituto della pipeline base di augmentation.
