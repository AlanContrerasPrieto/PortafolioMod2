# PortafolioMod2
## Instrucciones:
Entregable: Implementación de un modelo de deep learning.

* Crea un repositorio de github para este proyecto.
* Selecciona un problema y consigue un dataset para resolver dicho problema.
* El dataset deberá ser diferente a los empleados en clase.
* El dataset no puede estar clasificado como un "toy dataset" (por ejemplo MNIST)
* Implementa una arquitectura de deep learning para solucionar el problema utilizando keras + Tensorflow. Lo que se busca es que apliques correcta y efectivamente las técnicas vistas en el módulo.
* La arquitectura no puede estar conformada únicamente por capas densas, se deben emplear unidades especializadas (Conv, LSMT, etc.)
* Analiza los resultados de tu modelo usando un set de pruebas y validación.
* Mejora tu modelo usando técnicas de regularización, ajustando hiper parámetros, modificando la arquitectura de tu modelo o buscando otro modelo. 
* Documenta y explica cuáles son los cambios que funcionaron y por qué funcionaron. 
* Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
* Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

## Formato de entrega:  
* Sube el link del repositorio del proyecto de github a la actividad en canvas. Dentro del repositorio deberán de estar contenidos:
* Conjunto de datos utilizados, en caso de ser necesario (se puede omitir si notebook incluye código para descargar datos o liga a su origen).
* Modelo entrenado, arquitectura + pesos en formato .keras
* Reporte en formato ipynb. El notebook deberá de estar organizado en las siguientes secciones:
* Introducción: planteamiento del problema que se busca resolver, así como su relevancia.
* Datos: descripción (incluyendo fuente), análisis, separación en entrenamiento y prueba y preprocesamiento (de ser necesario) del dataset * empleado.
* Desarrollo del modelo: descripción de la arquitectura empleada, entrenamiento y evaluación de resultados.
* Ajuste del modelo: ajuste de hiperparámetros/cambios en arquitectura para mejorar resultados previos (al menos una iteración).
* Resultados: evaluación de modelo final con datos de prueba.
* Conclusiones: análisis de los resultados obtenidos, identificación de posibles áreas de mejora.
* Aplicación: función para probar modelo con datos nuevos (por ejemplo, argumento=ruta a imagen, salida=resultado de clasificación).
* Todas las celdas de código del notebook deberán de mostrar el resultado de ejecutarlas (donde aplique).