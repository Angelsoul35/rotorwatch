# 5. Ejemplo resuelto y el control de barajado

## Los números reales

Salida de `python -m rotorwatch.real_benchmark` (reproducible, `seed=0`):

```
engines            : 100 (80 train / 20 held-out), 20631 cycles, 34 features, RUL cap 125

                           RMSE      MAE   NASA score
held-out engines         20.050   14.072     117318.5
grouped CV (mean)        18.685   13.059      62265.2
train engines            11.903    8.390            -
naive-mean baseline      41.980   36.817    1349753.7
shuffle control          41.800   36.470    1577197.9

train/held-out RMSE gap : +8.147   held-out vs CV agreement: 7.3%   CV RMSE std: 2.064
verdict: well-fit
```

## Lectura, línea por línea

* **held-out 20,05 de RMSE / 14,07 de MAE.** Sobre 20 motores que el modelo **jamás** vio,
  estima la vida restante con un error típico de unos 20 ciclos (y ~14 en promedio
  absoluto). Está en el rango competitivo que reporta la literatura para FD001 con RUL
  recortada.
* **baseline de la media 41,98.** Predecir siempre la RUL media de entrenamiento es el
  **suelo**: el error que hay que superar para valer algo. El modelo lo deja en **menos de la
  mitad**, y su score de NASA es más de **10× mejor** (117 318 vs 1 349 753).
* **validación cruzada 18,69 (std 2,06).** El held-out (20,05) coincide con la validación
  cruzada agrupada a un **7,3%**. Esto es lo que convierte el número en creíble: no fue
  suerte de una partición, es una propiedad del método. La desviación de 2,06 entre pliegues
  confirma estabilidad.
* **train 11,90 — brecha +8,15.** El entrenamiento tiene menos error que el held-out. **Es
  esperado**: un ensamble de árboles ajusta el entrenamiento muy bien. Con evaluación
  disjunta por motor, lo que importa es que held-out ≈ validación cruzada (y lo es), no que
  la brecha sea cero. Por eso el veredicto es **well-fit**, no overfit.

## El control de barajado: ¿la señal es real?

Un buen número fuera de muestra todavía podría ser un artefacto del pipeline. El control
definitivo: **barajar las etiquetas**. Se reentrena el modelo con la columna RUL
**permutada** al azar (se destruye la relación sensor → desgaste, dejando intacto todo lo
demás) y se mide sobre los mismos motores held-out.

Resultado: el error **colapsa de vuelta al baseline** (RMSE 41,80 vs el baseline 41,98).
Con las etiquetas revueltas, lo mejor que puede hacer el modelo es predecir la media — que
es exactamente el baseline.

> **Qué prueba esto.** Que la ventaja del modelo (20,05 vs 41,98) **no** viene del pipeline
> ni de una fuga: viene de señal real de desgaste que aprendió de los sensores. Un pipeline
> que "acertara" con etiquetas barajadas estaría midiendo fuga, no aprendizaje. Una prueba
> del proyecto exige que el control de barajado quede por encima de 0,9× el baseline y muy
> por encima del held-out real.

## El demo sintético

`python -m rotorwatch.cli demo` corre la misma maquinaria sobre una flota **sintética** cuya
señal de desgaste inyectamos a propósito. Ahí el held-out da RMSE ~1,4 contra un baseline de
~42 y un control barajado de ~43: como la señal es limpia y la construimos nosotros, el
modelo la recupera casi perfecto. Es la demostración rápida (sin red) de que el método
recupera una señal conocida — y de que, si la barajas, el modelo colapsa.

## La lección

1. Reporta el error **fuera de muestra**, con división **por motor**, nunca por fila.
2. Compáralo siempre contra un **baseline** (la media): un número aislado no dice nada.
3. Recorta el objetivo RUL: pídele al modelo lo que **sí** es predecible.
4. Añade un **control de barajado**: es la prueba más barata y más contundente de que
   mediste aprendizaje y no fuga.
5. Diagnostica el ajuste por el **acuerdo held-out ↔ validación cruzada**, no por la brecha
   con el entrenamiento.
