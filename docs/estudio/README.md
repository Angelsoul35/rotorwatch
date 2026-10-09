# Guía de estudio — RotorWatch

Esta guía está en español y explica el proyecto para estudiarlo y sustentarlo en una
entrevista. El código y la documentación técnica están en inglés; aquí va el *porqué*: las
ideas, las fórmulas y las decisiones de diseño, con los números reales del proyecto.

El concepto central: **estimar la vida útil restante (RUL) de equipo rotativo a partir de
sus sensores**, y hacerlo de forma que el número resista la realidad — sin fuga de datos
entre motores, con el objetivo recortado estándar, y con un control que prueba que el
modelo aprende señal real de desgaste.

## Orden de lectura

1. [RUL y datos run-to-failure](01-rul-y-datos.md) — qué es la vida útil restante, qué son
   los datos de "correr hasta la falla", y por qué C-MAPSS.
2. [La trampa de fuga y la división por motor](02-fuga-y-division-por-motor.md) — el error
   más grave en prognosis y cómo se evita (holdout de motores completos, GroupKFold).
3. [El recorte de RUL y las features](03-recorte-de-rul-y-features.md) — el objetivo
   piecewise-linear (con su fórmula), la selección de sensores y el suavizado sin fuga.
4. [El modelo y las métricas](04-modelo-y-metricas.md) — el gradient boosting, RMSE, MAE y
   el score asimétrico de NASA (con fórmulas), y cómo se diagnostica over/underfitting.
5. [Ejemplo resuelto y el control de barajado](05-ejemplo-y-control.md) — los resultados
   reales leídos uno por uno, y la prueba de que la señal es real.

## Cómo correrlo mientras estudias

```bash
pip install -e ".[dev]"
python -m rotorwatch.cli demo        # fleet sintética (rápido, sin red)
python -m rotorwatch.real_benchmark  # benchmark completo con datos reales de C-MAPSS
pytest                               # 39 pruebas
```

## Relevancia en petróleo y gas

Un "motor" aquí es cualquier máquina rotativa monitoreada en una plataforma: el turbo-gas
que mueve un compresor, una bomba centrífuga de exportación de crudo, una bomba de
inyección de agua. El mismo método estima las horas (o ciclos) hasta la falla a partir de
las señales de vibración, temperatura, presión y flujo de esa máquina, y convierte el
historial de fallas en un tiempo de anticipación para programar el mantenimiento.
