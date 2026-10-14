# Desafío 4: traductor inglés → español con Seq2Seq

Trabajo práctico de **Procesamiento de Lenguaje Natural I** (CEIA, FIUBA).

**Autor:** Leandro A. Britez

## Descripción

Traductor *encoder-decoder* con LSTM, basado en el modelo desarrollado en clase. El encoder resume la oración en inglés en sus estados finales $(h, c)$ y el decoder genera la traducción en español palabra por palabra, con *teacher forcing* durante el entrenamiento y decodificación *greedy* en la inferencia.

El trabajo replica el modelo de clase y lo extiende con más datos, más unidades recurrentes y más capas.

## Experimentos

| Exp. | Datos | LSTM | Propósito |
|---|---|---|---|
| E0 | 10.000 oraciones, vocabulario de 8.000 palabras | 256 unidades × 1 capa | Réplica adaptada del modelo de clase |
| E1 | 60.000 oraciones, vocabulario de 12.000 / 15.000 palabras, sin truncar | 256 × 1 | Escenario extendido |
| E2 | Igual que E1 | 512 × 1 | Más unidades |
| E3 | Igual que E1 | 256 × 2 | Más capas |

Los cuatro modelos se evalúan sobre un mismo conjunto de test de 6.000 oraciones que no se usa para entrenar. Se informan la loss y la accuracy sin contar el padding, y BLEU sobre las traducciones generadas.

## Resultados

| Exp. | Parámetros entrenables | Accuracy test | BLEU |
|---|---|---|---|
| E0 | 4,5 M | 0,336 | 3,35 |
| E1 | 8,5 M | 0,494 | 11,22 |
| E2 | 14,3 M | 0,516 | 13,67 |
| E3 | 9,6 M | 0,507 | 13,45 |

El paso al escenario extendido es el cambio que más mejora los resultados. E2 y E3 superan a E1 y quedan muy cerca entre sí, con E3 usando bastantes menos parámetros. Cada configuración se entrenó una sola vez, por lo que las comparaciones son exploratorias. El análisis completo, los ejemplos de traducción y las conclusiones están en el notebook.

## Contenido

- [`Desafio_4_Traductor_V2.ipynb`](Desafio_4_Traductor_V2.ipynb): notebook con el código, los resultados y las conclusiones, ejecutado en Google Colab.

## Ejecución

1. Abrir el notebook en [Google Colab](https://colab.research.google.com/).
2. Seleccionar un entorno con GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.
3. Ejecutar todas las celdas. El notebook descarga automáticamente el dataset `spa-eng` y los embeddings GloVe.

El entrenamiento de los cuatro modelos lleva alrededor de 1 hora y 50 minutos en una T4. Se utilizó TensorFlow 2.21.

## Datos

- Pares de oraciones inglés-español (`spa-eng.zip`), el mismo dataset del notebook de clase.
- Embeddings GloVe de 50 dimensiones para el encoder, provistos por la cátedra.
