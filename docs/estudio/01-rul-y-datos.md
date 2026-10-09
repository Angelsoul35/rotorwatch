# 1. RUL y datos run-to-failure

## ¿Qué es la RUL?

**RUL** (Remaining Useful Life, vida útil restante) es cuántos ciclos de operación le
quedan a una máquina antes de fallar. Si una turbina falla en el ciclo 192 y hoy va en el
ciclo 50, su RUL es 142. Estimar la RUL es el corazón del **mantenimiento predictivo**.

Hay tres formas de mantener equipo:

* **Correctivo:** reparar cuando falla. Barato en papel, carísimo en la práctica (paro no
  planeado, daño secundario, y en una plataforma, un problema de seguridad).
* **Preventivo (por calendario):** cambiar piezas cada *N* horas. O se cambia equipo sano
  demasiado pronto, o se nos escapa el que ya se está degradando.
* **Predictivo:** mirar los sensores, estimar cuánta vida queda, y programar la
  intervención para la máquina correcta en el momento correcto.

El mantenimiento predictivo reemplaza el calendario por un **tiempo de anticipación basado
en datos**. Ese tiempo de anticipación es la RUL.

> **Concepto — ¿por qué es un problema de regresión?** La RUL es un número continuo (ciclos
> restantes). Cada fila es un ciclo de un motor con sus lecturas de sensores; la etiqueta es
> la RUL de ese ciclo. Predecir un número a partir de features es **regresión**.

## Datos run-to-failure

Los datos *run-to-failure* ("correr hasta la falla") son el insumo natural de la prognosis:
cada unidad se opera desde un estado sano hasta que falla, registrando sus sensores y
settings de operación en cada ciclo. Como la unidad llega hasta la falla, conocemos la RUL
exacta de cada ciclo: en el último ciclo la RUL es 0, y hacia atrás sube de uno en uno.

Esta es una propiedad que una prueba del proyecto verifica directamente: **dentro de un
motor, la RUL nunca sube** (solo baja o se mantiene) y **llega a 0** en la falla. Si subiera,
los datos estarían mal ordenados o mal etiquetados.

## C-MAPSS: el banco de pruebas estándar

**C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) es un simulador de la
NASA que generó el dataset público de referencia para RUL: flotas de motores turbofan
corridos hasta la falla, con 21 sensores y 3 settings de operación por ciclo. Tiene cuatro
sub-datasets (FD001–FD004) de dificultad creciente.

Este proyecto usa **FD001**: 100 motores, **una sola** condición de operación y **un solo**
modo de falla. Es el caso canónico y el más limpio para demostrar el método. El espejo
público de Hugging Face concatena los cuatro sub-datasets y **reinicia `unit_number` en 1**
para cada uno; por eso vendemos solo el bloque inicial de FD001 (hasta que el número de
unidad se reinicia), que son exactamente 20 631 filas para los motores 1–100. Mezclar
sub-datasets juntaría motores de regímenes de operación distintos bajo números de unidad
repetidos — un error que evitamos quedándonos en un solo sub-dataset.

### Las columnas

| Columna            | Significado                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `unit_number`      | id del motor (1–100 en FD001)                                      |
| `time_in_cycles`   | ciclo de operación, empieza en 1                                   |
| `op_setting_1..3`  | tres settings de operación (el 3 es constante en FD001)            |
| `sensor_1..21`     | 21 canales de sensor (7 son planos en FD001 y no aportan señal)    |
| `RUL`              | vida útil restante: ciclos hasta la falla, 0 en la falla           |

En petróleo y gas el equivalente sería el historian de una bomba o compresor: una fila por
intervalo, columnas de vibración, temperaturas de cojinete, presiones y flujos, y — cuando
se tiene historial de fallas — los ciclos u horas que faltaban para la falla.
