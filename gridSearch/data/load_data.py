import pandas as pd

# Binary classification problem
def read_and_preprocess_bin_data(path):
    df = pd.read_csv(path, index_col='id', delimiter=';')
    # Convert 'date_column' to datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Generate the month column
    df['mes'] = df['fecha'].dt.month
    # Get input and output variables
    y = df[['empleado']]
    X = df.drop(['t', 'fecha', 'fnacim', 'empleado', 'empleado_perma', 'empleado_sector', 'empleado_tcompleto', 'paro_ultimo', 'edad', 'dias_trab'], axis=1)
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
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Generate the month column
    df['mes'] = df['fecha'].dt.month
    # Get input and output variables
    y = df[['empleado_sector']]
    X = df.drop(['t', 'fecha', 'fnacim', 'empleado', 'empleado_perma', 'empleado_sector', 'empleado_tcompleto', 'paro_ultimo', 'edad', 'dias_trab'], axis=1)
    # Generate dummies for categorical variables
    X_dum = pd.get_dummies(X,
                           columns = ['sexo', 'edad_cat', 'educa_cat', 'expe', 'expe_perma', 'expe_tcompleto', 'paro', 'mes'],
                           prefix = {'sexo':'sexo', 'edad_cat':'edad_cat', 'educa_cat':'educa_cat', 'expe':'expe', 'expe_perma':'expe_perma', 'expe_tcompleto':'expe_tcompleto', 'paro':'paro', 'mes':'mes'},
                           drop_first=True)                
    return X_dum.to_numpy(), y.to_numpy().ravel()