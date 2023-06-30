# **CareerPilot**

![Google Colab](https://img.shields.io/badge/Google%20Colab-%23ffffff.svg?style=flat&logo=google-colab&logoColor=%23000)
![Gradio](https://img.shields.io/badge/Gradio-%23ffffff.svg?style=flat&logo=gradio&logoColor=%23000)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23ffffff.svg?style=flat&logo=scikit-learn&logoColor=%23000)
![XGBoost](https://img.shields.io/badge/XGBoost-%23ffffff.svg?style=flat&logo=xgboost&logoColor=%23000)
![MIT License](https://img.shields.io/badge/license-MIT-%23000.svg?style=flat)

<p align="center">
  <img src="CareerPilot_Logo.png" alt="CareerPilot" width="150" height="150">
</p>

*CareerPilot* es una herramienta de aprendizaje automático diseñada para revolucionar tus perspectivas laborales. Podrás:
-  Obtener información sobre tu empleabilidad para el próximo mes.
-  Obtener información sobre los sectores más probables que se ajusten a tus habilidades e intereses.

## **Tabla de Contenidos**

- [**Introducción**](#introducción)
- [**Uso**](#uso)
- [**Implementación propia**](#implementación-propia)
- [**Licencia**](#licencia)

----

## **Introducción**

CareerPilot es una aplicación desarrollada para la 4ª Edición de Saturdays.AI en Alicante por María Alfaro-Contreras y Jorge L. Casanova. Utilizando técnicas de aprendizaje automático, CareerPilot tiene la capacidad de predecir la probabilidad de empleo de los usuarios. Además, ofrece información detallada sobre los sectores con más oportunidades, recomendaciones de ocupaciones y formaciones clave, así como orientación sobre dónde buscar oportunidades laborales. 

Para elaborar este proyecto, usamos datos de la Muestra Continua de Vidas Laborales ([**MCVL**](https://www.seg-social.es/wps/portal/wss/internet/EstadisticasPresupuestosEstudios/Estadisticas/EST211)) que contiene la vida laboral del 5% de los trabajadores afiliados en España entre los años 2000 y 2019. Esta muestra supone casi un millón y medio de trabajadores, de los cuales consideramos aquellos con:
- Edades comprendidas entre los 16 y los 60 años.
- Al menos un episodio de empleo[^1] entre 2000 y 2019.
- Una fecha de fallecimiento posterior al 2019.

[^1]: Un episodio de empleo es un contrato laboral que contiene, al menos, una fecha de alta en la Seguridad Social. En el caso de la fecha de baja, esta puede ser determinada o indefinida (en cuyo caso consideramos como fecha de fin el día previo al episodio de empleo posterior o, de no existir, el 31/12/2019).

### *Preparación de los datos*

Los datos contienen una información extensa sobre los episodios de empleo así como características personales. Para nuestro análisis, consideramos la siguiente información:
- Nacionalidad española.
- Minuslavía: mayor o menor al 33%.
- Número de hijos.
- Género.
- Días trabajados acumulados, en 4 tramos: 0, menos de 1 año, entre 1 y 2 años, más de 2 años.
- Días trabajados acumulados a jornada completa, en 4 tramos: 0, menos de 1 año, entre 1 y 2 años, más de 2 años.
- Días trabajados acumulados con contrato indefinido, en 4 tramos: 0, menos de 1 año, entre 1 y 2 años, más de 2 años.
- Edad en 4 tramos: 16-25, 25-29, 30-45, +45.
- Tiempo en paro desde el último episodio de empleo, en 5 tramos: entre 0 y 3 meses, entre 4 y 6 meses, entre 7 y 12 meses, entre 1 y 2 años y más de 2 años.

El formato inicial de los datos es de una observación por trabajador y episodio de empleo. Para poder usarlos, transformamos los datos de la siguiente manera:
- Tomamos como referencia un día concreto para el momento del tiempo. En nuestro caso, el segundo martes de cada mes.
- Transformamos los datos en panel, donde cada individuo tiene una fila por cada momento del tiempo.
- Si el trabajador posee un empleo para ese día en concreto, se rellenan sus datos y se indica que está empleado. Si no, se encuentra desempleado.
- Solo tomamos en cuenta episodios de empleo por cuenta ajena, no propia.
- Si un trabajador se encuentra pluriempleado, tomamos como característica aquella más favorable para el trabajador (indefinido, tiempo completo).

Una vez transformados los datos, excluimos de la muestra todas aquellas observaciones donde el trabajador ya estaba empleado o se encontraba en situación de paro por más de cuatro años. El objetivo es captar transiciones del paro al empleo, por lo que solo nos interesa entrenar el modelo con episodios de búsqueda de empleo e identificar cuando se produce esta transición.

El cuaderno [**`eda.ipynb`**](eda.ipynb) contiene un análisis exploratorio de datos.

### **IMPORTANTE** 

> Los datos utilizados para el desarrollo de la aplicación no son públicos. Para obtener acceso a los datos, es necesario solicitarlos a [**mcvl.dgoss-sscc@seg-social.es**](mailto:mcvl.dgoss-sscc@seg-social.es). Puedes seguir los pasos detallados en [**este enlace**](https://www.seg-social.es/wps/portal/wss/internet/EstadisticasPresupuestosEstudios/Estadisticas/EST211) para solicitar el acceso. Una vez obtenidos, deberán ubicarse en la carpeta **`gridSearch/data`**. **NO ES NECESARIO TENER LOS DATOS PARA USAR LA APLICACIÓN.** Sin embargo, si deseas ampliar el análisis exploratorio o reentrenar los modelos, deberás seguir el proceso mencionado anteriormente. Se tiene en mente subir el script empleado para la transformación de datos en un futuro. Este script ha sido realizado en Stata y facilitará el proceso de preprocesamiento y transformación de los datos en el formato esperado por la aplicación.

## **Uso**

La aplicación se ha desarrollado utilizando la interfaz de Gradio. Para ejecutar la aplicación localmente, ya sea en tu navegador web o integrada en una celda de código de un cuaderno de Jupyter, es necesario instalar los requisitos del proyecto listados en el archivo [**`requirements.txt`**](requirements.txt). A continuación, se muestran las instrucciones para ejecutar la aplicación:

```shell
pip install -r requirements.txt
python CareerPilot.py
```

o

```shell
pip install -r requirements.txt
# Ejecutar las celdas correspondientes del cuarderno CareerPilot.ipynb
```

El funcionamiento de la aplicación se basa en varias funciones, las cuales se encuentran dentro de la carpeta **`utils`**. A continuación, se detalla la estructura de dicha carpeta:

- [**`utils/interface.py`**](utils/interface.py): Este archivo contiene las funciones necesarias para implementar la aplicación utilizando la interfaz de Gradio. Aquí se definen las interfaces de usuario y las acciones asociadas.
- [**`utils/model.py`**](utils/model.py): En este archivo se encuentra la función encargada de cargar los modelos utilizados por la aplicación. Aquí se establece la lógica de carga y configuración de los modelos.
- [**`utils/data.py`**](utils/data.py): En este archivo se encuentran las funciones necesarias para transformar los datos introducidos por el usuario al formato esperado por los modelos. Aquí se definen las transformaciones y preprocesamientos de los datos de entrada.

Esta estructura organizada en la carpeta [**`utils`**](utils) permite un enfoque modular y facilita la mantenibilidad y escalabilidad de la aplicación.

## **Implementación propia**

Se ha llevado a cabo una comparativa de varios modelos utilizando un enfoque de búsqueda aleatoria con 10 iteraciones para determinar el modelo más adecuado para el problema en cuestión. Los archivos relacionados con esta comparativa se encuentran en la carpeta **`gridSearch`**:

```
.
├── gridSearch
│   ├── bin_models.py -> Se comparan cinco modelos para el problema de la estimación de la probabilidad de empleabilidad.
│   ├── class_models.py -> Se comparan dos modelos para el problema de la estimación del sector más probable en el que encontrar trabajo.
│   ├── run_grid.sh -> Script que llama a gridSearch/bin_models.py y gridSearch/class_models.py, respectivamente.
│   └── final_models.py -> Script que entrena los dos mejores modelos encontrados.
```

Para poder ejecutarlos, así como modificar los modelos existentes y la búsqueda de hiperparámetros o añadir nuevos modelos, es necesario contar con los [**datos**](#importante).

Puedes encontrar información más detallada sobre el entrenamiento de los modelos en nuestro artículo en [**Medium**](https://medium.com/@marialfacon/careerpilot-41630f8b27d0).

## **Licencia**

Este proyecto está bajo la [**Licencia MIT**](LICENSE).
