# `feature_selection.py` — Explicación detallada

## Contexto y motivación

Cuando un dataset tiene decenas de variables, limpiarlas todas a mano es ineficiente. Muchas de ellas pueden ser ruido puro: no tienen ninguna relación estadística con lo que queremos predecir. Esta función aplica cuatro algoritmos de selección de variables de forma automática y devuelve un informe que permite descartar las irrelevantes antes de entrar en el proceso de limpieza y feature engineering.

El caso de uso concreto es un problema de **clasificación binaria** con una **variable target categórica** (dos clases en formato string, por ejemplo `"yes"/"no"` o `"positive"/"negative"`).

---

## Estructura general del código

El módulo se organiza en tres bloques:

1. **Helpers internos** (`_encode_features`, `_safe_impute`): preparan los datos para que puedan alimentar a los modelos.
2. **Funciones de scoring** (`_mutual_info`, `_permutation_importance_score`, `_lasso_coef`, `_boruta_score`): cada una aplica un algoritmo distinto y devuelve una `pd.Series` con el score de cada feature.
3. **Función principal** (`select_features`): orquesta los pasos anteriores, ensambla los resultados en un DataFrame y aplica el criterio de descarte.

---

## Preparación de los datos

### `_encode_features`

Los algoritmos de sklearn esperan matrices numéricas. Las columnas categóricas del dataframe se convierten con `LabelEncoder`, que asigna un entero a cada categoría. Los `NaN` se preservan: antes de codificar se guarda una máscara booleana de sus posiciones y se restauran después, para no perder información sobre qué valores faltan.

Las columnas numéricas no se tocan en este paso.

### `_safe_impute`

`LabelEncoder` ya maneja los `NaN` de las categóricas, pero los modelos de sklearn no aceptan `NaN` en ninguna posición de la matriz. Esta función reemplaza todos los valores faltantes con la **mediana de cada columna**. Es una imputación temporal y mínima: solo sirve para poder ejecutar los modelos de scoring; el dataframe original nunca se modifica.

---

## Los cuatro algoritmos de scoring

### 1. Información mutua — `_mutual_info`

La **información mutua** (MI) mide cuánta información comparte una variable con el target. A diferencia de la correlación de Pearson, no asume ninguna forma funcional: detecta relaciones lineales, no lineales, o de cualquier otro tipo.

Un score de `0` indica independencia estadística total. Cuanto mayor sea el valor, más dependencia existe entre la variable y el target.

Se usa `mutual_info_classif` de sklearn, que estima la MI mediante k-vecinos más cercanos.

**Cuándo puede fallar**: con muchos valores faltantes o variables con muy poca varianza, la estimación puede ser ruidosa.

### 2. Permutation Importance — `_permutation_importance_score`

Este método entrena un **Random Forest** con todas las variables y luego, para cada variable, baraja aleatoriamente sus valores `n_repeats` veces y mide cuánto cae la accuracy del modelo en cada barajado.

La lógica es: si una variable es importante para las predicciones del modelo, romper su relación con el target (barajándola) debería empeorar el modelo notablemente. Si al barajarla el modelo no cambia, la variable no aportaba nada.

A diferencia de la importancia basada en impureza (`feature_importances_` del Random Forest), este método no está sesgado hacia variables con alta cardinalidad.

**Cuándo puede fallar**: si dos variables están altamente correlacionadas entre sí, al barajar una el modelo puede compensar con la otra, y ambas pueden aparecer como poco importantes aunque juntas sí aporten.

### 3. Coeficientes L1 (Lasso) — `_lasso_coef`

Se entrena una **Regresión Logística con penalización L1**, que es el equivalente del Lasso para problemas de clasificación. La penalización L1 añade al coste de entrenamiento un término proporcional a la suma de los valores absolutos de los coeficientes, lo que fuerza a que los coeficientes de las variables menos relevantes converjan exactamente a **cero**.

Las variables se estandarizan con `StandardScaler` antes de entrenar, para que los coeficientes sean comparables entre sí independientemente de la escala de cada variable.

Se devuelve el valor absoluto de cada coeficiente. Un coeficiente de `0` (o muy cercano a `0`) indica que el modelo no necesitó esa variable para aprender.

**Cuándo puede fallar**: al ser un modelo lineal, no detecta relaciones no lineales. Una variable que solo interactúa con el target de forma no lineal puede recibir un coeficiente de cero aunque sea informativa.

### 4. Boruta — `_boruta_score`

Boruta es el método más riguroso de los cuatro. Su idea central es crear una prueba estadística formal para decidir si una variable es relevante o no.

El proceso es el siguiente:

1. Para cada columna del dataset, crea una **shadow feature**: una copia con los valores barajados aleatoriamente. Estas sombras no tienen ninguna relación real con el target.
2. Entrena un Random Forest sobre el dataset original más todas las shadow features.
3. Compara la importancia de cada feature original con la importancia de la **mejor shadow feature**. Si la original supera sistemáticamente a la mejor sombra a lo largo de múltiples iteraciones (test binomial), se considera **confirmada**. Si consistentemente queda por debajo, se **rechaza**.

El resultado para cada variable puede ser:
- `1` → confirmada como relevante
- `0` → rechazada
- `-1` → tentativa (no hay suficiente evidencia en ninguna dirección; merece revisión manual)

**Cuándo puede fallar**: es computacionalmente caro (puede tardar varios minutos) y puede necesitar muchas iteraciones para resolver las variables tentativas.

---

## Criterio de descarte

Una variable se marca con `discard = 1` únicamente si cae por debajo del umbral en los **tres métodos principales simultáneamente** (MI, permutation importance y Lasso). Se usa una regla AND conservadora a propósito: si al menos uno de los tres métodos encuentra algo relevante en la variable, se conserva.

Boruta no vota en el descarte automático, pero su columna (`boruta_support`) sirve para revisar manualmente los casos dudosos, especialmente los tentativas (`-1`).

Los umbrales por defecto son deliberadamente bajos para ser conservadores:
- `mi_threshold = 0.01`
- `perm_threshold = 0.0`
- `lasso_threshold = 1e-4`

Se pueden ajustar al llamar a la función si se quiere ser más o menos agresivo.

---

## Output

La función devuelve un `pd.DataFrame` ordenado: primero las variables conservadas (ordenadas por información mutua descendente), luego las descartadas.

| columna | descripción |
|---|---|
| `feature` | nombre de la variable |
| `discard` | `1` = descartar, `0` = conservar |
| `mutual_info` | score de información mutua |
| `perm_importance` | importancia media por permutación |
| `lasso_coef` | coeficiente L1 en valor absoluto |
| `boruta_support` | `1` confirmada / `0` rechazada / `-1` tentativa / `NaN` si se omitió |

---

## Uso básico

```python
from feature_selection import select_features

df_features = df.drop(columns=["target"])
target      = df["target"]

report = select_features(df_features, target)

# Ver variables descartadas
report[report["discard"] == 1]

# Reducir el dataframe a las variables útiles
features_to_keep = report.loc[report["discard"] == 0, "feature"].tolist()
df_reduced = df[features_to_keep + ["target"]]
```

Para omitir Boruta y obtener resultados más rápido:

```python
report = select_features(df_features, target, run_boruta=False)
```

---

## Limitaciones y consideraciones

- La función está diseñada para ser ejecutada **antes del data cleaning**. Los datos sucios (valores faltantes, outliers, encoding incorrecto) no impiden que funcione, pero pueden añadir ruido a los scores. Una variable con muchos `NaN` puede recibir un score bajo simplemente porque hay poca información disponible, no porque sea intrínsecamente irrelevante.
- El criterio AND conservador minimiza los falsos negativos (variables útiles descartadas por error), pero puede dejar pasar algunas variables irrelevantes. El objetivo en esta fase es reducir el trabajo de limpieza, no hacer una selección definitiva.
- La selección final de features debería hacerse después del cleaning y el feature engineering, no antes.
