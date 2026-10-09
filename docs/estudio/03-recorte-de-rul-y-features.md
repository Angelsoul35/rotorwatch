# 3. El recorte de RUL y las features

## Por qué se recorta la RUL

Un motor nuevo y sano no da pistas de cuándo fallará. Sus sensores se ven igual 300 ciclos
antes de la falla que 200 ciclos antes: planos, normales. La degradación — y con ella la
señal que un modelo puede leer — **aparece tarde**, cuando el desgaste empieza a mover los
sensores.

Si entrenamos al modelo para que prediga la RUL exacta desde el primer ciclo, le estamos
pidiendo lo imposible (adivinar 300 vs 250 sin información) y solo le metemos **ruido de
etiqueta**. La práctica estándar en la literatura de C-MAPSS resuelve esto con un objetivo
**lineal por tramos (piecewise-linear)**: se recorta la RUL en un tope constante *C*.

### La fórmula

$$
\text{RUL}_{\text{objetivo}}(t) = \min\big(\text{RUL}_{\text{real}}(t),\; C\big), \qquad C = 125
$$

Leída como tramos:

$$
\text{RUL}_{\text{objetivo}}(t) =
\begin{cases}
C & \text{si } \text{RUL}_{\text{real}}(t) \ge C \quad (\text{vida lejana: objetivo plano})\\[4pt]
\text{RUL}_{\text{real}}(t) & \text{si } \text{RUL}_{\text{real}}(t) < C \quad (\text{desgaste visible: baja lineal a 0})
\end{cases}
$$

Es decir: mientras falten más de 125 ciclos, el objetivo se mantiene **plano** en 125;
cuando la RUL real baja de 125, el objetivo **decrece linealmente** hasta 0 en la falla.

> **Esto NO es reducir el alcance.** Es el método documentado. Le pedimos al modelo la
> pregunta que **sí** tiene respuesta — *¿está cerca la falla, y qué tan cerca?* — en vez de
> una que no (adivinar una vida lejana). Las predicciones también se recortan a `[0, C]`:
> una máquina no puede tener vida restante negativa, y por encima del tope el número no
> significa nada.

En el código, `clip_rul` es exactamente `min(rul, 125)`, y una prueba comprueba que
`[10, 124, 125, 200, 361]` se convierte en `[10, 124, 125, 125, 125]`.

## Las features (sin fuga)

El pipeline (`FeaturePipeline`) convierte las lecturas crudas en una matriz lista para el
modelo, en tres pasos. **Todo se ajusta solo con los motores de entrenamiento** y se aplica
igual a los de prueba — si el escalador o la selección "vieran" los motores de prueba, sería
otra forma de fuga.

### 1. Quitar canales muertos

Varios sensores de FD001 reportan un **valor constante** en toda la flota: no contienen
información sobre el desgaste. `select_informative` deja fuera toda columna cuya desviación
estándar sea prácticamente cero. En FD001 esto descarta 7 canales planos (entre ellos
`sensor_1`, `sensor_5`, `op_setting_3`, …). La selección se mide **sobre el entrenamiento**,
no sobre toda la base.

> **Concepto — selección por varianza, no por nombre.** No borramos "sensor_1" porque sepamos
> que es constante; borramos cualquier columna que no varíe. Una prueba arma un DataFrame
> pequeño con columnas constantes y variables y verifica que se quedan solo las que varían.

### 2. Suavizar por motor (media móvil *trailing*)

Las lecturas son ruidosas ciclo a ciclo. Para denoise, se añade una **media móvil** corta
(ventana 5) de cada canal, **agrupada por motor** y *trailing* (solo usa el ciclo actual y
los anteriores del **mismo** motor, con `min_periods=1`). Dos propiedades la hacen libre de
fuga:

* **Agrupada por motor:** nunca promedia lecturas de dos motores distintos.
* **Trailing:** el valor en el ciclo *t* usa solo ciclos ≤ *t*, nunca el futuro.

Una prueba lo fija con números a mano: el primer ciclo de cada motor conserva su valor
crudo, y el primer ciclo del segundo motor **no** se mezcla con el último del primero.

### 3. Escalar

Los sensores viven en escalas muy distintas (uno ronda 9000, otro 0,03). Un `StandardScaler`
(ajustado en entrenamiento) deja cada feature con media 0 y desviación 1, para que todas
sean comparables.

El resultado son **34 features**: 17 canales informativos más su media móvil por motor.
Quitar el suavizado baja un poco la precisión (RMSE held-out 20,4 vs 20,0), lo que justifica
incluirlo: aporta señal real, sin fuga.
