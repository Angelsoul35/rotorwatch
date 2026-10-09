# 4. El modelo y las métricas

## El modelo: gradient boosting

El regresor es un `HistGradientBoostingRegressor` de scikit-learn: un ensamble de árboles de
decisión entrenados en secuencia, donde cada árbol corrige el error del conjunto anterior
(gradient boosting), con una variante basada en histogramas que lo hace rápido en decenas de
miles de filas.

**¿Por qué gradient boosting y no una regresión lineal o una red?** La señal de RUL son
umbrales no lineales que **interactúan** entre canales ruidosos: una temperatura que sube *y*
una presión que cae *a la vez* significan desgaste. Los ensambles de árboles capturan esas
interacciones directamente, sin suponer la forma del efecto de cada sensor, y son muy
fuertes en datos tabulares como estos.

Se eligieron hiperparámetros **regularizados a priori** (no ajustados contra los motores de
prueba — eso sería otra fuga): una tasa de aprendizaje baja, un tope de hojas, una
penalización L2 y un mínimo de muestras por hoja. Esa regularización evita que el modelo
memorice motores individuales, que es justo lo que arruinaría una evaluación por motor.

> **Concepto — gradient boosting.** Muchos árboles pequeños en cadena; cada uno aprende el
> residual (lo que el conjunto aún no explica). Es como ir corrigiendo poco a poco una
> estimación, en lugar de ajustar un solo modelo grande de una vez.

## Las métricas

Para cada predicción, el error de una instancia es la diferencia entre la RUL predicha y la
real (ambas recortadas al tope).

### RMSE — error cuadrático medio (en ciclos)

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Penaliza más los errores grandes (por el cuadrado). Es la métrica principal de C-MAPSS.

### MAE — error absoluto medio (en ciclos)

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

El "cuántos ciclos nos equivocamos, en promedio". Fácil de comunicar.

### Score de NASA — la función asimétrica de C-MAPSS (PHM08)

Sea $d = \hat{y} - y$ (predicho menos real) el error de una instancia:

$$
s = \sum_{i=1}^{n}
\begin{cases}
e^{-d_i/13} - 1 & \text{si } d_i < 0 \quad (\text{predicción temprana: conservadora})\\[4pt]
e^{\,d_i/10} - 1 & \text{si } d_i \ge 0 \quad (\text{predicción tardía: sobreestima la vida})
\end{cases}
$$

Menor es mejor; 0 es perfecto. La clave es la **asimetría**: como el denominador de las
tardías (10) es menor que el de las tempranas (13), un error tardío cuesta **más** que uno
temprano del mismo tamaño.

> **¿Por qué asimétrico?** Una predicción **tardía** dice que queda más vida de la que hay:
> corres una máquina que está por fallar — peligroso y caro. Una **temprana** la sacas de
> servicio antes de tiempo: desperdicio, pero sin catástrofe. La métrica codifica ese
> compromiso de mantenimiento, por eso es la que reporta el área. Una prueba verifica que un
> error tardío de 10 ciclos puntúa más que uno temprano de 10 ($e^{1}-1 > e^{10/13}-1$).

El score se suma sobre todas las instancias held-out; como el modelo, el baseline y el
control comparten exactamente las mismas instancias, sus scores son comparables entre sí.

## Over/underfitting: cómo se diagnostica

Tres números cuentan la historia del ajuste:

* **Error de entrenamiento** (sobre los motores de entrenamiento).
* **Error held-out** (sobre los 20 motores nunca vistos).
* **Error de validación cruzada agrupada** (promedio sobre pliegues disjuntos por motor).

Y el **veredicto** (`verdict`) los combina:

* **underfit** — el error held-out no le gana al baseline de la media. El modelo no aprendió
  nada útil.
* **overfit** — le gana al baseline, pero el error held-out se **aleja** del de validación
  cruzada (más de 25%). Señal de estar ajustándose a motores específicos, no al desgaste.
* **well-fit** — el held-out le gana al baseline **y** coincide con la validación cruzada.

> **Cuidado con la brecha train/held-out.** Un ensamble de árboles ajusta el entrenamiento
> **muy** bien por diseño: es normal que el error de entrenamiento sea bastante menor que el
> de held-out. Con una evaluación **disjunta por motor**, la prueba correcta de
> generalización **no** es que esa brecha sea cero, sino que el held-out **coincida con la
> validación cruzada**. Por eso el modelo **no** se mutila para cerrar la brecha: eso
> sacrificaría precisión real para maquillar un número.
