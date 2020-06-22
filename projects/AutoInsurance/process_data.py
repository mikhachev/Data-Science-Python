import pandas as pd


def map_for_dict_Gender(Gender):
    dict_Gender = {'Male': 0, 'Female': 1}
    res = dict_Gender.get(Gender)
    return res

def map_for_dict_HasKmLimit(HasKmLimit):
    dict_HasKmLimit = {'no': 0, 'yes': 1}
    res = dict_HasKmLimit.get(HasKmLimit)
    return res

def map_for_dict_MariStat(MariStat):
    dict_MariStat = {'Other': 0, 'Alone': 1}
    res = dict_MariStat.get(MariStat)
    return res


def f_VehUsage_Professional(VehUsage):
    if VehUsage == 'Professional':
        VehUsage_Professional = 1
    else:
        VehUsage_Professional = 0
    return VehUsage_Professional


def f_VehUsage_Private_trip_to_office(VehUsage):
    if VehUsage == 'Private+trip to office':
        VehUsage_Private_trip_to_office = 1
    else:
        VehUsage_Private_trip_to_office = 0
    return VehUsage_Private_trip_to_office


def f_VehUsage_Private(VehUsage):
    if VehUsage == 'Private':
        VehUsage_Private = 1
    else:
        VehUsage_Private = 0
    return VehUsage_Private


def f_VehUsage_Professional_run(VehUsage):
    if VehUsage == 'Professional run':
        VehUsage_Professional_run = 1
    else:
        VehUsage_Professional_run = 0
    return VehUsage_Professional_run


def return_NewDataFrame():
    columns = [
        'LicAge',
        'Gender',
        'MariStat',
        'DrivAge',
        'HasKmLimit',
        'BonusMalus',
        'RiskArea',
        'VehUsg_Private',
        'VehUsg_Private+trip to office',
        'VehUsg_Professional',
        'VehUsg_Professional run',
        'SocioCateg_CSP1',
        'SocioCateg_CSP2',
        'SocioCateg_CSP3',
        'SocioCateg_CSP4',
        'SocioCateg_CSP5',
        'SocioCateg_CSP6',
        'SocioCateg_CSP7'
    ]

    return pd.DataFrame(columns=columns)


def process_input(json_input):
    LicAge = json_input["LicAge"]

    Gender = map_for_dict_Gender(json_input["Gender"])
    MariStat = map_for_dict_MariStat(json_input["MariStat"])
    DrivAge = json_input["DrivAge"]
    HasKmLimit = map_for_dict_HasKmLimit(json_input["HasKmLimit"])
    BonusMalus = json_input["BonusMalus"]
    RiskArea = json_input["RiskArea"]
    VehUsg_Private = f_VehUsage_Private(json_input["VehUsage"])
    VehUsg_Private_trip_to_office = f_VehUsage_Private_trip_to_office(json_input["VehUsage"])
    VehUsg_Professional = f_VehUsage_Professional(json_input["VehUsage"])
    VehUsg_Professional_run = f_VehUsage_Professional_run(json_input["VehUsage"])
    SocioCateg = json_input["SocioCateg"]

    df = return_NewDataFrame()

    df.loc[0, 'LicAge'] = LicAge
    df.loc[0, 'Gender'] = Gender
    df.loc[0, 'MariStat'] = MariStat
    df.loc[0, 'DrivAge'] = DrivAge
    df.loc[0, 'HasKmLimit'] = HasKmLimit
    df.loc[0, 'BonusMalus'] = BonusMalus
    df.loc[0, 'RiskArea'] = RiskArea
    df.loc[0, 'VehUsg_Private'] = VehUsg_Private
    df.loc[0, 'VehUsg_Private+trip to office'] = VehUsg_Private_trip_to_office
    df.loc[0, 'VehUsg_Professional'] = VehUsg_Professional
    df.loc[0, 'VehUsg_Professional run'] = VehUsg_Professional_run
    df.loc[0, 'SocioCateg_CSP1'] = 0
    df.loc[0, 'SocioCateg_CSP2'] = 0
    df.loc[0, 'SocioCateg_CSP3'] = 0
    df.loc[0, 'SocioCateg_CSP4'] = 0
    df.loc[0, 'SocioCateg_CSP5'] = 0
    df.loc[0, 'SocioCateg_CSP6'] = 0
    df.loc[0, 'SocioCateg_CSP7'] = 0
    df.loc[0, 'SocioCateg_' + str(SocioCateg)] = 1

    return df
