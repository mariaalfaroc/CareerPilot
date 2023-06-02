import pandas as pd

WORK_CATEGORIES = [
    'Artisticas',
    'Teleco',
    'Finanzas',
    'Cientificas',
    'Primario',
    'Extractivas',
    'Publica',
    'Sanidad',
    'Transporte',
    'Educacion',
    'Administrativas',
    'Comercio',
    'Construccion',
    'Hosteleria',
    'Paro'
]

# Binary classification problem
def read_and_preprocess_bin_data(path):
    df = pd.read_csv(path, index_col='id', delimiter=';')
    # Convert 'date_column' to datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d%b%Y')
    # Generate the month column
    df['mes'] = df['fecha'].dt.month
    # Get input and output variables
    y = df[['empleado']]
    # Keep 'Administrativas', 'Comercio', 'Construccion', 'Hosteleria' and 'Paro'
    X = df.drop(['t', 'fecha', 'fnacim', 'empleado', 'empleado_perma', 'empleado_sector', 'empleado_tcompleto', 'paro_ultimo', 'edad', 'dias_trab', 'expe_Artisticas', 'expe_Teleco', 'expe_Finanzas', 'expe_Cientificas', 'expe_Primario', 'expe_Extractivas', 'expe_Publica', 'expe_Sanidad', 'expe_Transporte', 'expe_Educacion'], axis=1)
    # Generate dummies for categorical variables
    X_dum = pd.get_dummies(X,
                           columns = ['sexo', 'edad_cat', 'educa_cat', 'expe', 'expe_perma', 'expe_tcompleto', 'paro', 'mes'],
                           prefix = {'sexo':'sexo', 'edad_cat':'edad_cat', 'educa_cat':'educa_cat', 'expe':'expe', 'expe_perma':'expe_perma', 'expe_tcompleto':'expe_tcompleto', 'paro':'paro', 'mes':'mes'},
                           drop_first=True)
    return X_dum.to_numpy(), y.to_numpy().ravel()

# Multiclass classification problem
def read_and_preprocess_class_data(path):
    df = pd.read_csv(path, index_col='id', delimiter=';')
    # Convert 'date_column' to datetime
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d%b%Y')
    # Generate the month column
    df['mes'] = df['fecha'].dt.month
    # Get input and output variables
    y = df[['empleado_sector']]
    # Keep 'Administrativas', 'Comercio', 'Construccion', 'Hosteleria' and 'Paro'
    X = df.drop(['t', 'fecha', 'fnacim', 'empleado', 'empleado_perma', 'empleado_sector', 'empleado_tcompleto', 'paro_ultimo', 'edad', 'dias_trab', 'expe_Artisticas', 'expe_Teleco', 'expe_Finanzas', 'expe_Cientificas', 'expe_Primario', 'expe_Extractivas', 'expe_Publica', 'expe_Sanidad', 'expe_Transporte', 'expe_Educacion'], axis=1)
    # Generate dummies for categorical variables
    X_dum = pd.get_dummies(X,
                           columns = ['sexo', 'edad_cat', 'educa_cat', 'expe', 'expe_perma', 'expe_tcompleto', 'paro', 'mes'],
                           prefix = {'sexo':'sexo', 'edad_cat':'edad_cat', 'educa_cat':'educa_cat', 'expe':'expe', 'expe_perma':'expe_perma', 'expe_tcompleto':'expe_tcompleto', 'paro':'paro', 'mes':'mes'},
                           drop_first=True)          
    return X_dum.to_numpy(), y.to_numpy().ravel()