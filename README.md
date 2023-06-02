# **CareerPilot**

![Google Colab](https://img.shields.io/badge/Google%20Colab-%23ffffff.svg?style=flat&logo=google-colab&logoColor=%23000)
![Gradio](https://img.shields.io/badge/Gradio-%23ffffff.svg?style=flat&logo=gradio&logoColor=%23000)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23ffffff.svg?style=flat&logo=scikit-learn&logoColor=%23000)
![XGBoost](https://img.shields.io/badge/XGBoost-%23ffffff.svg?style=flat&logo=xgboost&logoColor=%23000)
![MIT License](https://img.shields.io/badge/license-MIT-%23000.svg?style=flat)


CareerPilot es una herramienta de aprendizaje automático diseñada para revolucionar tus perspectivas laborales. Podrás:
-  Obtener información sobre tu empleabilidad para el próximo mes.
-  Obtener información sobre los sectores más probables que se ajusten a tus habilidades e intereses.


## Tabla de Contenidos

- [Introducción](#introducción)
- [Uso](#uso)
- [Estructura de Archivos](#estructura-de-archivos)
- [Licencia](#licencia)

## Introducción

TODO: Proporciona una breve introducción al proyecto, explicando su propósito y objetivos.
Aquí habría que hablar de los datos y se puede mencionar el eda.ipynb

## Uso

Explica cómo utilizar el proyecto y cualquier detalle importante sobre su funcionalidad. Proporciona ejemplos de comandos o fragmentos de código.

TODO: En local.
```shell
python CareerPilot.py
```

Habría que ver si se puede crear un notebook que tenga unas liberías ya instaladas.
```shell
pip install -r requirements.txt
```

## Estructura de Archivos

Quitar los principales archivos que se mencionan antes y solo hablar del tema del  gridsearch.
El repositorio del proyecto tiene la siguiente estructura:

```
.
├── CareerPilot.py
├── CareerPilot.ipynb
├── eda.ipynb
├── requirements.txt
├── gridSearch
│   ├── run_grid.sh
│   ├── bin_models.py
│   └── data
│       ├── ipi_spain_ine.xlsx
│       ├── 04_mcvl_sample_model.csv
│       ├── load_data.py
│       └── 05_mcvl_full_sample.csv
│   └── bin_grid_results
│       ├── cv_splits.npy
│       └── models
│           ├── model_decision_tree_best_estimator.joblib
│           └── model_logistic_regression_best_estimator.joblib
├── utils
│   ├── interface.py
│   ├── model.py
│   └── data.py
├── models
│   ├── multiclass.pkl
│   └── binclass.pkl
├── README.md
└── .gitignore
```

### Descripción de Archivos

- `CareerPilot.py`: Descripción del archivo.
- `CareerPilot.ipynb`: Descripción del archivo.
- `eda.ipynb`: Descripción del archivo.
- `requirements.txt`: Lista de dependencias y sus versiones.
- `gridSearch/run_grid.sh`: Descripción del archivo.
- `gridSearch/bin_models.py`: Descripción del archivo.
- `gridSearch/data/ipi_spain_ine.xlsx`: Descripción del archivo.
- `gridSearch/data/04_mcvl_sample_model.csv`: Descripción del archivo.
- `gridSearch/data/load_data.py`: Descripción del archivo.
- `gridSearch/data/05_mcvl_full_sample.csv`: Descripción del archivo.
- `gridSearch/bin_grid_results/cv_splits.npy`: Descripción del archivo.
- `gridSearch/bin_grid_results/models/model_decision_tree_best_estimator.joblib`: Descripción del archivo.
- `gridSearch/bin_grid_results/models/model_logistic_regression_best_estimator.joblib`: Descripción del archivo.
- `utils/interface.py`: Descripción del archivo.
- `utils/model.py`: Descripción del archivo.
- `utils/data.py`: Descripción del archivo.
- `models/multiclass.pkl`: Descripción del archivo.
- `models/binclass.pkl`: Descripción del archivo.
- `README.md`: Este archivo proporciona una descripción general del proyecto.
- `.gitignore`: Descripción del archivo.

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).