# 2. La trampa de fuga y la división por motor

Este es el concepto más importante del proyecto. Si solo te llevas una idea de esta guía,
que sea esta.

## La trampa

Los ciclos consecutivos de un mismo motor son **casi idénticos**: el motor 7 en el ciclo
100 y en el ciclo 101 tiene lecturas de sensores prácticamente iguales y RULs que difieren
en 1. Si repartes las **filas** al azar entre entrenamiento y prueba, casi todos los motores
terminan con filas en **ambos** lados.

¿Qué pasa entonces? El modelo ve, durante el entrenamiento, el ciclo 100 del motor 7 con su
RUL. En la prueba le preguntan por el ciclo 101 del motor 7 — que es casi el mismo punto.
El modelo no necesita *aprender degradación*: le basta con **reconocer el motor** y copiar
la respuesta del vecino que ya vio. El error de prueba sale espectacular... y es mentira.
En producción, el motor es nuevo, nunca se vio, y el modelo falla.

A esto se le llama **fuga de datos (data leakage)**: información del conjunto de prueba se
cuela al entrenamiento. Aquí la fuga entra por el *grupo* (el motor), no por una feature.

> **Concepto — fuga por grupo.** Cuando las filas vienen en grupos muy correlacionados
> (ciclos de un motor, latidos de un paciente, transacciones de un cliente), dividir por
> fila deja que el modelo memorice el grupo. La evaluación mide memoria, no generalización.

## La solución: dividir por motor

RotorWatch **nunca** divide por fila. Divide por **motor**:

* **Holdout por motor** (`split_units`): se barajan los ids de motor con una semilla fija y
  se apartan 20 motores **completos** para prueba. Todos los ciclos de un motor van al mismo
  lado. Así los 20 motores de prueba son motores que el modelo **jamás** vio.
* **Validación cruzada agrupada** (`GroupKFold` por motor): para estimar el error de forma
  estable (no de una sola partición con suerte), se hacen varios pliegues; en cada uno, los
  motores de validación son **disjuntos** de los de entrenamiento.

El benchmark incluye una aseveración explícita: antes de reportar cualquier número,
comprueba que los conjuntos de motores de entrenamiento y de prueba **no se solapan**. Si se
solaparan, el benchmark se detiene. Una prueba del proyecto rompe a propósito esta división
(la cambia a un split por fila) y verifica que las aseveraciones fallan: así sabemos que la
protección realmente protege.

```python
# La idea, en esencia:
units = unique_units(df)               # ids de motor, no filas
test_units = shuffled[-n_test:]        # 20 motores completos a prueba
is_test = df["unit_number"].isin(test_units)
train_df, test_df = df[~is_test], df[is_test]
assert train_units.isdisjoint(test_units)   # sin solape de motores
```

## Por qué importa en la práctica (petróleo y gas)

Es el mismo error que entrenar un modelo con la bomba P-101 y "validarlo" con más lecturas
de la **misma** P-101. El número saldría optimista y no diría nada sobre la P-102, que es la
que te interesa. La pregunta real siempre es: *¿funciona en una máquina que no estaba en los
datos de entrenamiento?* La única forma de responderla es apartar máquinas completas.

## La lección

- Reporta el error **fuera de muestra**, nunca el de entrenamiento.
- Cuando los datos vienen en grupos correlacionados, la división **debe** ser por grupo.
- Que el error de holdout y el de la validación cruzada agrupada **coincidan** es la señal
  de que el número es una propiedad del método, no suerte de una partición.
