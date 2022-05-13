#Clean Titanic Data Files:

def clean_titanic_data(df):
    '''
    Takes in a titanic dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Returns: clean_df - a dataframe with the cleaning operations performed on it
    '''
    #Drop Duplicates
    df.drop_duplicates(inplace = True)
    #Drop Columns
    columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)
    #Encoded Categorical Variables
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df.drop(columns =  ['sex', 'embark_town'])

def impute_age(train, validate, test):
    '''
    Imputes the mean age of train to all three datasets.
    '''
    imputer = SimpleImputer(strategy = 'mean', missing_values = np.nan)
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def prep_titanic_data(df):
    df = clean_titanic_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.survived, random_state = 1234)
    train, validate = train_test_split(train, train_size = .7, stratify = train.survived, random_state = 1234)
    train, validate, test = impute_age(train, validate, test)
    return train, validate, test

#--------------------------------------------------------------------------------------------------------------------------------------------

#Cleaning Iris Data:
def clean_iris_data(df):
    #Dropping unneeded columns
    columns_to_drop = ['species_id', 'measurement_id']
    df = df.drop(columns = columns_to_drop)
    #renaming species_name column
    df = df.rename(columns = {'species_name' : 'species'})
    #creating dummy variables of species
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    #concatenating dummy variables onto original dataframe
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def prep_iris(df):
    df = clean_iris_data(df)
    train, test = train_test_split(df, train_size = 0.2, stratify = df.species, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.3, stratify = train.species, random_state = 1234)
    return train, validate, test