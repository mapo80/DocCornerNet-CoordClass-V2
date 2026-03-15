# DocCornerNet V2 — Proposta Revisionata

## 0. Sintesi della review

La proposta originale contiene alcune idee valide, ma mescola fatti verificati, ipotesi architetturali e stime numeriche non difendibili. Le correzioni principali sono:

- `0.9929` non e il baseline ufficiale di `mobilenetv2_256_best` su `val_cleaned`: nel repo il valore verificato e `0.9902`. Il `0.9929` esiste, ma su `val_clean_iter4_mix` per `mobilenetv2_256_clean_iter3`.
- `loss_tau=0.1` non e una novita di v2: nel `train_ultra.py` corrente esiste gia come parametro separato da `tau`, che resta il tau di decode.
- `num_bins > img_size` non raddoppia automaticamente la precisione geometrica: il decode attuale usa gia soft-argmax sub-pixel. Bins piu alti sono un'ablation utile, non una garanzia.
- La tabella parametri della proposta originale non torna: `v1` oggi ha `495353` parametri, quindi una `v2` con piu blocchi non puo scendere a `~381K` senza ridurre esplicitamente i canali.
- La delega XNNPACK completa non si puo dichiarare "a tavolino": va verificata dopo export TFLite sul grafo reale.
- Il vero punto debole del progetto non e solo il clean set, ma la robustezza su `rev-new` e sulla coda lunga/outlier. La `v2` va valutata anche li.

Conclusione: la direzione giusta per una `v2` esiste, ma va resa piu minimale, piu verificabile e meno speculativa.

---

## 1. Stato reale del progetto

### Dataset operativi per v2

Per la `v2`, i dataset da usare come source of truth sono:

- dataset di prova rapida: `dataset/DocCornerDataset_small`
- dataset finale: `dataset/DocCornerDataset`

Quindi il flusso corretto e:

1. smoke test e ablation veloci su `DocCornerDataset_small`
2. training e validazione reali su `DocCornerDataset`
3. una sola conferma finale su `test` di `DocCornerDataset`, dopo aver congelato architettura e iperparametri

Gli split come `val_cleaned`, `val_clean_iter4_mix` e `rev-new` restano utili solo come **benchmark storici legacy** del lavoro `v1`. Non devono essere il criterio principale di approvazione della `v2`.

### Dati verificati nel repository

| Voce | Valore verificato |
|------|-------------------|
| Dataset principale | `32968` train / `8645` validation / `6652` test |
| Modello mobile corrente | `MobileNetV2 alpha=0.35`, `fpn_ch=32`, `simcc_ch=96` |
| Parametri `v1` reali | `495353` |
| `mobilenetv2_224_best` su `val_cleaned` | `mIoU=0.9894`, `err_mean=0.57px` |
| `mobilenetv2_256_best` su `val_cleaned` | `mIoU=0.9902`, `err_mean=0.60px` |
| Miglior clean score presente nel repo | `mIoU=0.9929` su `val_clean_iter4_mix` (`mobilenetv2_256_clean_iter3`) |
| Miglior worst-case cross-dataset nel repo | `mIoU=0.9047` su `rev-new` (`mobilenetv2_224_from256_clean_iter3`) |
| Export TFLite gia verificati | top-4 modelli `100%` delegati a XNNPACK |
| TFLite INT8 top-4 | `0.82-0.84 MB`, `2.39-2.92 ms` invoke-only |

### Osservazioni importanti

- La baseline "ufficiale" per la nuova `v2` va ridefinita su `dataset/DocCornerDataset`, non sugli split legacy.
- `val_cleaned` e `val_clean_iter4_mix` non vanno mescolati: sono benchmark storici diversi.
- La famiglia `clean_iter3` e piu robusta cross-dataset della famiglia `*_best` salvata nei checkpoint principali.
- `v1` contiene gia diversi mattoni riutilizzabili: `ECA`, `Resize1D`, `NearestUpsample2x`, `Conv1DAsConv2D`, `Broadcast1D`, `SimCCDecode`, `loss_tau` separato da `tau`.

---

## 2. Obiettivi corretti di v2

La `v2` va giudicata prima di tutto sul dataset finale `dataset/DocCornerDataset`. Gli artifact legacy servono per orientarsi, ma non per decidere il go/no-go finale.

### 2.1 Criterio principale

| Dataset / split | Ruolo |
|-----------------|-------|
| `dataset/DocCornerDataset_small` / `validation` | smoke test e ablation veloci |
| `dataset/DocCornerDataset` / `validation` | benchmark primario di sviluppo |
| `dataset/DocCornerDataset` / `test` | verifica finale una tantum, non tuning loop |

### 2.2 Target pratici

| Metrica | Ambito | Target primario | Stretch |
|--------|--------|-----------------|---------|
| `mIoU` | `DocCornerDataset` / `validation` | migliorare la baseline `v1` da misurare sullo stesso split | `>= 0.995` se realisticamente raggiungibile |
| `err_mean` | `DocCornerDataset` / `validation` | migliorare la baseline `v1` sullo stesso split | `< 0.35px` se sostenibile |
| `R@99` | `DocCornerDataset` / `validation` | migliorare la baseline `v1` sullo stesso split | `> 95%` se sostenibile |
| TFLite INT8 size | export finale | `< 1MB` | `< 0.9MB` |
| XNNPACK delegation | export finale | nessuna regressione | `100%` delega confermata |
| Input resolution | train/inferenza | `224x224` | nessuna variazione |

Nota: finche non congeliamo una baseline `v1` misurata su `dataset/DocCornerDataset/validation`, tutti i numeri assoluti del paragrafo storico restano orientativi, non contrattuali.

Regola di lettura dei benchmark:

- benchmark primario di go/no-go: `dataset/DocCornerDataset/validation`
- benchmark finale di conferma: `dataset/DocCornerDataset/test` una sola volta
- benchmark rapido per iterazione: `dataset/DocCornerDataset_small/validation`
- benchmark secondari legacy: `val_cleaned`, `val_clean_iter4_mix`, `rev-new`

---

## 3. Cosa e supportato dai dati, cosa e solo ipotesi

### Fatti supportati dal repo

- La pipeline attuale e gia molto efficiente per deployment.
- Il collo di bottiglia non e il classificatore `has_doc`: la score head e gia quasi saturata.
- La generalizzazione su clean e buona; la coda lunga e il domain shift restano il problema aperto.
- `MobileNetV2 alpha=0.35` e oggi il miglior default mobile del progetto per tradeoff accuracy/size/latency.

### Ipotesi plausibili ma non ancora dimostrate

- Le feature condivise prima del SimCC creano ambiguita tra i 4 corner.
- Una leggera separazione per-corner prima del pooling puo aiutare piu di ulteriori aumenti di capacita globale.
- Un neck con leggero bottom-up path puo migliorare i casi difficili senza impatto rilevante sul deploy.
- Bins piu alti e target piu stretti possono aiutare la precisione fine, ma solo se non peggiorano memoria, stabilita o robustezza.

### Ipotesi da tenere fuori dalla MVP architetturale

- `Adaptive Wing Loss` come sostituto diretto della loss coordinata.
- `DFL` come combinazione obbligatoria con SimCC.
- Cross-corner context dedicato come prima modifica architetturale.

Queste idee non sono da scartare, ma non vanno messe nella MVP della `v2`.

`EMA` non appartiene a questa lista: e gia implementato e BN-corretto nel codice corrente (`train_ultra.py:739-779`, traccia `model.variables`). Va trattato come ablation training-only in Step A, non come modifica architetturale speculativa. Questo pero non significa "gratis": ha overhead di memoria e di `assign` a ogni step, quindi va valutato come costo runtime, non come costo di integrazione.

---

## 4. Proposta v2 raccomandata

### 4.1 Principio guida

Fare una `v2` piccola e leggibile, non una riscrittura totale. La modifica architetturale deve essere una sola ipotesi forte: **separare prima possibile l'informazione dei 4 corner**, mantenendo invariati backbone, export path e quasi tutta la testa SimCC.

### 4.2 MVP architetturale

```
Input [B, 224, 224, 3]
        |
        v
+---------------------------+
| MobileNetV2 alpha=0.35    |
| C2 56x56  C3 28x28        |
| C4 14x14  C5  7x7         |
+---------------------------+
        |
        v
+---------------------------+
| Mini-FPN v1 (top-down)    |
| P2 + upsample(P3) + fuse  |
+---------------------------+
        |
        v
+---------------------------+
| p_fused [B,56,56,32]      |
+---------------------------+
        |
        v
+---------------------------+
| Shared SepConv2D          |
+---------------------------+
        |
   +----+----+----+----+
   |    |    |    |    |
   v    v    v    v
 att_TL att_TR att_BR att_BL
   |    |    |    |
   *    *    *    *      elementwise multiply with p_fused
   |    |    |    |
 feat  feat feat feat
  TL    TR   BR   BL
   |    |    |    |
   +----+----+----+------------------------------+
                                                |
                           per-corner X/Y AxisMean + Resize1D
                                                |
                           shared SimCC Conv1D head per branch
                                                |
                                   4 x (x logits, y logits)
                                                |
                                                v
                                         coords [B, 8]

Parallel branch:
  C5 -> score head invariata -> score_logit
```

### 4.3 Modifica architetturale consigliata: Corner Spatial Attention

Il punto non e che `AxisMean` "distrugge" completamente l'informazione spaziale. Il punto e che **oggi i 4 corner leggono dallo stesso tensore condiviso** e vengono separati solo molto tardi. La proposta piu sensata e introdurre 4 mappe di attenzione leggere prima del pooling:

```python
# p_fused: [B, 56, 56, 32]

shared = SepConv2D(32, 3, padding="same")(p_fused)

att_tl = sigmoid(Conv2D(1, 1)(shared))
att_tr = sigmoid(Conv2D(1, 1)(shared))
att_br = sigmoid(Conv2D(1, 1)(shared))
att_bl = sigmoid(Conv2D(1, 1)(shared))

feat_tl = p_fused * att_tl
feat_tr = p_fused * att_tr
feat_br = p_fused * att_br
feat_bl = p_fused * att_bl
```

Da qui in poi, la testa SimCC **condivide gli stessi layer Conv1D** di `v1` (pooling per asse, `Resize1D`, due `Conv1D`, output logit per asse), ma il flusso dati cambia: 4 Multiply + 8 AxisMean + 8 Resize1D (vs 2+2 in v1), piu la pipeline Conv1D applicata 4 volte (shared weights) o ristrutturata con batch-concat. I parametri crescono di ~1.5K, ma il **compute aumenta di ~4x per le operazioni spaziali**. Serve profiling latency su WASM subito dopo implementazione per verificare che il budget realtime sia rispettato.

### 4.4 Cosa NON fare subito

- Non introdurre insieme `corner attention`, `PAN-Lite`, `cross-corner context`, `AWL`, `DFL` e nuova schedule multi-fase. Sarebbe impossibile attribuire i guadagni.
- Non cambiare backbone nella MVP.
- Non toccare la score head.

### 4.5 Tabella comparativa `v1` vs `v2`

| Aspetto | `v1` | `v2` proposta |
|---------|------|---------------|
| Stato | implementata e verificata nel repo | proposta, da implementare e misurare |
| Input | `224x224` o `256x256` nei checkpoint storici | `224x224` fisso come target |
| Backbone | `MobileNetV2 alpha=0.35` | invariato |
| Neck | Mini-FPN top-down | Mini-FPN top-down invariata nella MVP |
| Fusione feature | un solo `p_fused` condiviso | un solo `p_fused`, ma seguito da separazione per-corner |
| Separazione dei 4 corner | tardiva, soprattutto nella testa SimCC finale | anticipata tramite `corner-specific spatial attention` |
| Pooling spaziale | `AxisMean` globale su feature condivise | `AxisMean` eseguito per ciascun corner dopo attention |
| Resize 1D | una coppia `X/Y` su feature condivise | quattro coppie `X/Y`, una per corner |
| SimCC head | una testa condivisa che emette i 4 corner | stessa logica di testa, ma applicata a rami per-corner |
| Score head | su `C5`, gia sufficiente | invariata |
| Parametri | `495353` verificati | stimati `~497K` nella MVP |
| Compute runtime | baseline attuale | piu alto della `v1`, soprattutto nelle operazioni per-corner |
| Rischio deploy | basso, gia validato | medio: export e latenza da riconfermare |
| Filosofia del modello | massima condivisione fino a valle | separazione anticipata minima, senza riscrivere il modello |
| Obiettivo tecnico | baseline mobile efficiente | ridurre l'ambiguita tra i 4 corner mantenendo i vincoli mobile |
| Workflow di validazione | storico, con vari benchmark legacy | `DocCornerDataset_small` per prove rapide, `DocCornerDataset` per decisione finale |

In sintesi:

- `v1` e un modello mobile molto compatto che condivide quasi tutto fino alla fine
- `v2` non cambia famiglia di modello: aggiunge una sola ipotesi forte, cioe separare prima i 4 corner
- la `v2` va letta come evoluzione minima della `v1`, non come architettura completamente nuova

---

## 5. Modifiche opzionali, in ordine corretto

### Step A — Training-only ablations sulla `v1`

Prima di toccare l'architettura, lo Step A va diviso in due stadi:

1. **pruning rapido su `dataset/DocCornerDataset_small`**
   serve a scartare configurazioni chiaramente cattive, instabili o troppo lente
2. **conferma su `dataset/DocCornerDataset`**
   serve a verificare solo i candidati sopravvissuti al pruning rapido

Gli esperimenti candidati sono:

- `num_bins`: provare `224`, `336`, `448`
- `sigma_px`: scalarlo proporzionalmente a `num_bins` per mantenere la stessa larghezza pixel. Formula: `sigma_new = sigma_old × (num_bins_new / num_bins_old)`. Esempio: `sigma_px=3.0` a `224` bins → `sigma_px=6.0` a `448` bins (stessa copertura di ±3 pixel immagine)
- `loss_tau`: confrontare **prima `0.5`, poi `0.1`**. Sullo small dataset gli artifact esistenti suggeriscono che `0.5` sia la partenza piu conservativa: `E_phase_a` (`loss_tau=0.1`, `sigma=1.5`, `num_bins=224`) chiude a `mean_iou=0.03999` dopo 15 validation epochs, mentre `E_phase_a_tau05` (`loss_tau=0.5`) arriva a `0.53889` alla epoch 22 e a `0.39345` alla epoch 15. Questa non e una prova che `0.5` sia ottimale sul full dataset; e solo un'indicazione che `0.1` oggi e piu rischioso come primo tentativo
- `w_coord`: confrontare `0.5` vs `1.0`
- `EMA`: provare `--ema_decay 0.999 --ema_warmup_steps 2000`. L'implementazione in `train_ultra.py` e gia completa e BN-corretta (traccia `model.variables`, non solo `trainable_variables`). Costo di integrazione: quasi nullo. Costo runtime/memoria: non nullo, quindi va misurato
- outlier weighting: preservare la strategia esistente
- weak augmentation finale: tenere la logica gia presente

**Nota sugli ablation precedenti**: esperimenti su `DocCornerDataset_small` esistono gia nel repo (`A_maximum`, `E_phase_a`, `E_phase_a_tau05`) ma sono inconcludenti — dataset troppo piccolo, early termination, convergenza parziale. Questo non rende inutile `DocCornerDataset_small`: lo rende inadatto a decisioni finali, ma ancora utile per pruning rapido e smoke test.

Questo step serve a capire quanta strada si puo fare senza nuova architettura.

### Step B — Corner attention

Se gli ablation training-only si fermano presto, introdurre la sola corner attention.

### Step C — PAN-Lite opzionale

Solo dopo aver misurato lo Step B. Un piccolo bottom-up path e ragionevole, ma non va messo nella MVP iniziale.

Proposta corretta:

- usare `AvgPool2D(2,2)` come scelta leggera e parameter-free
- non motivarlo con "strided conv non e XNNPACK-safe", perche non e vero

### Step D — Cross-corner context

Da backlog, non da MVP. E una modifica piu difficile da isolare e da giustificare.

---

## 6. Loss e training strategy corrette

### 6.1 Cosa tenere

- SimCC Gaussian CE come loss primaria
- loss coordinata semplice (`L1` o al massimo `SmoothL1/Charbonnier`)
- outlier-aware sampling e augmentation
- training a `224x224`

### 6.2 Cosa correggere nella proposta originale

- `tau` di decode non va abbassato automaticamente: il parametro che interessa al training e `loss_tau`, gia presente in `v1`.
- `sigma_px=3.0` con `num_bins=448` non e "piu stretto" in senso assoluto: e l'equivalente di `1.5px` a `224`, quindi va sempre ragionato nello spazio dei bins. Regola di scaling: `sigma_new = sigma_old × (num_bins_new / num_bins_old)` per mantenere la stessa larghezza pixel immagine.
- `EMA` va considerato opzionale come prerequisito architetturale, ma e un ablation training-only gia implementato e BN-corretto. Non e a costo zero in runtime/memoria, ma ha costo di integrazione quasi nullo. Va testato in Step A.
- `Adaptive Wing Loss` e `DFL` non sono correzioni necessarie: sono backlog.

### 6.3 Piano di training raccomandato

Due fasi bastano:

| Fase | Obiettivo | Configurazione |
|------|-----------|----------------|
| Phase 1 | convergenza stabile | cosine, warmup, full aug, outlier strategy attuale |
| Phase 2 | rifinitura | lr bassa, weak aug finale, opzionale aumento `w_coord` |

Tre fasi con molte schedule diverse sono possibili, ma non sono la priorita.

---

## 7. Budget parametri corretto

La proposta originale va corretta: la `v2` non puo avere meno parametri di `v1` senza una riduzione esplicita di canali.

### Stima realistica

| Variante | Delta stimato | Totale stimato |
|----------|---------------|----------------|
| `v1` attuale | — | `495353` |
| `+ corner attention` | `~1.5K` | `~496.8K` |
| `+ PAN-Lite` | `~2.5K-3K` | `~499K-500K` |
| `+ cross-corner context` | `~2K-3K` | `~501K-503K` |

Questi numeri sono stime di progetto, non conti finali. Il criterio corretto e:

1. implementare
2. fare `model.count_params()`
3. esportare TFLite INT8
4. misurare davvero size e latenza

### Implicazione pratica

Anche una `v2` da `~500K` parametri resta ampiamente compatibile con il vincolo `< 1MB` in INT8. Il vero rischio non e il peso dei parametri, ma l'effetto su export, delega e attivazioni runtime.

---

## 8. Compatibilita XNNPACK: versione corretta

### Cosa possiamo dire con buona confidenza

- I blocchi gia presenti in `v1` (`NearestUpsample2x`, `Resize1D`, `Broadcast1D`, `Conv1DAsConv2D`, `SimCCDecode`) sono gia stati usati in modelli con delega completa.
- `Conv2D`, `SeparableConv2D`, `Add`, `Multiply`, `AvgPool2D`, `Dense`, `Softmax`, `MatMul` non sono di per se un problema per il deploy mobile del progetto.

### Cosa NON possiamo dichiarare in anticipo

- Che una futura `v2` sara `100%` delegata senza test.
- Che ogni combinazione di reshape/permute/resize introdotta resti equivalente al grafo exportato oggi.

### Regola corretta

Una modifica architetturale e "XNNPACK-safe" solo dopo:

1. export TFLite
2. delegate report
3. verifica assenza di nodi non delegati
4. misura di size e latency

---

## 9. Piano di verifica raccomandato

### 9.1 Smoke test

- modello compila
- shape output corrette
- `model.count_params()` coerente con il budget
- train breve `5-10` epoche su `dataset/DocCornerDataset_small` senza crash

### 9.2 Accuratezza

Misurare sempre in questo ordine:

1. `dataset/DocCornerDataset_small` / `validation` per capire se il modello si comporta bene nelle prove rapide
2. `dataset/DocCornerDataset` / `validation` come benchmark vero di sviluppo
3. `dataset/DocCornerDataset` / `test` come conferma finale una sola volta, dopo aver congelato la scelta

I benchmark legacy:

- `val_cleaned`
- `val_clean_iter4_mix`
- `rev-new`

sono opzionali e servono solo se vogliamo confrontare la `v2` con la storia del progetto `v1`.

### 9.3 Deploy

- export TFLite INT8
- size `< 1MB`
- delegate report XNNPACK
- latency su artifact reale, non solo stima teorica

### 9.4 Criterio di promozione

Una modifica passa allo step successivo solo se:

- migliora `DocCornerDataset_small/validation` o almeno non mostra regressioni evidenti nelle prove rapide
- poi migliora `DocCornerDataset/validation` in modo misurabile
- non rompe export / delega / size budget

Il controllo su `DocCornerDataset/test` non va usato a ogni iterazione: va eseguito solo sul candidato finale.

---

## 10. Ordine di lavoro raccomandato

1. Riprodurre una baseline `v1` @224 su `dataset/DocCornerDataset_small`, solo per smoke test e sanity check.
2. Riprodurre una baseline `v1` @224 su `dataset/DocCornerDataset/validation` e congelare queste metriche come riferimento principale.
3. Fare pruning rapido su `dataset/DocCornerDataset_small` per gli ablation training-only (`num_bins`, `sigma_px`, `loss_tau`, `w_coord`, opzionalmente `EMA`).
4. Promuovere solo i candidati sensati su `dataset/DocCornerDataset/validation`.
5. Implementare la sola `corner spatial attention`.
6. Verificare prima su `DocCornerDataset_small`, poi su `DocCornerDataset/validation`.
7. Valutare se serve davvero `PAN-Lite`.
8. Usare `dataset/DocCornerDataset/test` una sola volta sul candidato finale.
9. Tenere `cross-corner context`, `AWL`, `DFL` in backlog finche i passi 1-8 non sono saturi.

---

## 11. Decisione finale

La proposta originale va **semplificata**.

La `v2` sensata per questo repo e:

- stessa base mobile di `v1`
- stessa filosofia di deploy
- miglioramento mirato alla separazione dei 4 corner
- valutazione centrata su `DocCornerDataset_small` per prove rapide e su `DocCornerDataset` per la decisione finale

La modifica architetturale che ha piu senso implementare per prima e **corner-specific spatial attention**. Tutto il resto deve essere subordinato a misure reali, non a stime teoriche.
