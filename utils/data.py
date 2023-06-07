import pandas as pd

def transform_data(data):
    """Transforms the data to be used in the model.
    data = {'sexo': str,
            'edad': int,
            'nhijos': int,
            'espanol': str,
            'minusv': str,
            'educa_cat': str,       
            'expe': str,
            'expe_perma': str,
            'expe_tcompleto': str,
            'paro': str,
            'mes_actual': str,
            'expe_sector: list}"""
    # First, create a DataFrame
    df = pd.DataFrame([data])
  
    # Define the desired order of columns
    desired_order = ['minusv', 'espanol', 'nhijos', 'expe_Administrativas', 'expe_Comercio',
                     'expe_Construccion', 'expe_Hosteleria', 'sexo_Male',
                     'edad_cat_16-24', 'edad_cat_25-29', 'edad_cat_30-45',
                     'educa_cat_Primarios', 'educa_cat_Secundarios',
                     'educa_cat_Sin estudios', 'educa_cat_Universitarios', 'expe_1-2',
                     'expe_<1', 'expe_>2', 'expe_perma_1-2', 'expe_perma_<1',
                     'expe_perma_>2', 'expe_tcompleto_1-2', 'expe_tcompleto_<1',
                     'expe_tcompleto_>2', 'paro_1-2a', 'paro_4-6m', 'paro_7-12m', 'paro_>2a',
                     'mes_2', 'mes_3', 'mes_4', 'mes_5', 'mes_6', 'mes_7', 'mes_8', 'mes_9',
                     'mes_10', 'mes_11', 'mes_12']
  
    # Fill frame
    df['minusv'] = (df['minusv'] == 'Sí').astype(int)
    df['espanol'] = (df['espanol'] == 'Sí').astype(int)  
    df['sexo_Male'] = (df['sexo'] == 'Hombre').astype(int)
    df['edad_cat_16-24'] = ((df['edad'] >= 16) & (df['edad'] <= 24)).astype(int)
    df['edad_cat_25-29'] = ((df['edad'] >= 25) & (df['edad'] <= 29)).astype(int)
    df['edad_cat_30-45'] = ((df['edad'] >= 30) & (df['edad'] <= 45)).astype(int)
    df['educa_cat_Primarios'] = (df['educa_cat'] == 'Primarios').astype(int)
    df['educa_cat_Secundarios'] = (df['educa_cat'] == 'Secundarios').astype(int)
    df['educa_cat_Sin estudios'] = (df['educa_cat'] == 'Sin estudios').astype(int)
    df['educa_cat_Universitarios'] = (df['educa_cat'] == 'Universitarios').astype(int)
    df['expe_1-2'] = (df['expe'].isin(['Entre 1-2 años'])).astype(int)
    df['expe_<1'] = (df['expe'].isin(['Menos de 1 año'])).astype(int)
    df['expe_>2'] = (df['expe'].isin(['Más de 2 años'])).astype(int)
    df['expe_perma_1-2'] = (df['expe_perma'].isin(['Entre 1-2 años'])).astype(int)
    df['expe_perma_<1'] = (df['expe_perma'].isin(['Menos de 1 año'])).astype(int)
    df['expe_perma_>2'] = (df['expe_perma'].isin(['Más de 2 años'])).astype(int)
    df['expe_tcompleto_1-2'] = (df['expe_tcompleto'].isin(['Entre 1-2 años'])).astype(int)
    df['expe_tcompleto_<1'] = (df['expe_tcompleto'].isin(['Menos de 1 año'])).astype(int)
    df['expe_tcompleto_>2'] = (df['expe_tcompleto'].isin(['Más de 2 años'])).astype(int)
    df['paro_1-2a'] = df.paro.isin(['Entre 1-2 años'])
    df['paro_4-6m'] = df.paro.isin(['Entre 4-6 meses'])
    df['paro_7-12m']= df.paro.isin(['Entre 7-12 meses'])
    df['paro_>2a'] = df.paro.isin(['Más de 2 años'])
    df['mes_2'] = (df['mes_actual'] == 'Febrero').astype(int)
    df['mes_3'] = (df['mes_actual'] == 'Marzo').astype(int)
    df['mes_4'] = (df['mes_actual'] == 'Abril').astype(int)
    df['mes_5'] = (df['mes_actual'] == 'Mayo').astype(int)
    df['mes_6'] = (df['mes_actual'] == 'Junio').astype(int)
    df['mes_7'] = (df['mes_actual'] == 'Julio').astype(int)
    df['mes_8'] = (df['mes_actual'] == 'Agosto').astype(int)
    df['mes_9'] = (df['mes_actual'] == 'Septiembre').astype(int)
    df['mes_10'] = (df['mes_actual'] == 'Octubre').astype(int)
    df['mes_11'] = (df['mes_actual'] == 'Noviembre').astype(int)
    df['mes_12'] = (df['mes_actual'] == 'Diciembre').astype(int)  
    df['expe_Construccion'] = df['expe_sector'].apply(lambda x: 'Construcción' in x).astype(int)
    df['expe_Comercio'] = df['expe_sector'].apply(lambda x: 'Comercio' in x).astype(int)
    df['expe_Hosteleria'] = df['expe_sector'].apply(lambda x: 'Hostelería' in x).astype(int)
    df['expe_Administrativas'] = df['expe_sector'].apply(lambda x: 'Administrativas y servicios auxiliares' in x).astype(int) 
    df = df.drop(['mes_actual', 'sexo', 'edad', 'educa_cat', 'expe', 'expe_perma', 'expe_tcompleto', 'paro', 'expe_sector'], axis=1)
 
    # Reorder the columns
    df = df.reindex(columns=desired_order)

    return df

def clean_sector(input):
    """Clean the sector column of the input dataframe."""
    input.loc[input['classes'] == 'Primario', 'sector_clean'] = 'Primario'
    input.loc[input['classes'] == 'Extractivas', 'sector_clean'] = 'Industrias extractivas y manuf. y suministros'
    input.loc[input['classes'] == 'Construccion', 'sector_clean'] = 'Construcción'
    input.loc[input['classes'] == 'Comercio', 'sector_clean'] = 'Comercio'
    input.loc[input['classes'] == 'Transporte', 'sector_clean'] = 'Transporte y almacenamiento'
    input.loc[input['classes'] == 'Hosteleria', 'sector_clean'] = 'Hostelería'
    input.loc[input['classes'] == 'Teleco', 'sector_clean'] = 'Información y comunicaciones'
    input.loc[input['classes'] == 'Finanzas', 'sector_clean'] = 'Finanzas, seguros e inmobiliarias'
    input.loc[input['classes'] == 'Cientificas', 'sector_clean'] = 'Profesionales, científicas y técnicas'
    input.loc[input['classes'] == 'Administrativas', 'sector_clean'] = 'Administrativas y servicios auxiliares'
    input.loc[input['classes'] == 'Publica', 'sector_clean'] = 'Administración Pública'
    input.loc[input['classes'] == 'Educacion', 'sector_clean'] = 'Educación'
    input.loc[input['classes'] == 'Sanidad', 'sector_clean'] = 'Sanidad y servicios sociales'
    input.loc[input['classes'] == 'Artisticas', 'sector_clean'] = 'Hogar, actividades asociativas, y artísticas y recreativas'
    return input