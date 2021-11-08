import numpy as np

def add_user_items_features(df, item_features, user_features):
    df = df.merge(item_features, on='item_id', how='left')
    df = df.merge(user_features, on='user_id', how='left')
    return df

def user_average_bill(df):
    """средний чек"""
    user_avr_bill=df.groupby(['user_id', 'basket_id'])['sales_value'].mean().reset_index()
    user_avr_bill=user_avr_bill.groupby(['user_id'])['sales_value'].mean().reset_index()
    user_avr_bill.rename(columns={'sales_value': 'user_average_bill'}, inplace=True)
    return user_avr_bill

def overall_average_bill(user_average_bill):
    overall_avr_bill=user_average_bill['user_average_bill'].mean()
    return overall_avr_bill

def user_item_quantity(df):
    """количество покупок товаров пользователем"""
    user_item_quantity=df.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    user_item_quantity.rename(columns={'quantity': 'user_item_quantity'}, inplace=True)
    return user_item_quantity

def overall_average_item_quantity(user_item_quantity):
    overall_average_item_quantity=user_item_quantity.groupby(['item_id'])['user_item_quantity'].mean().reset_index()
    overall_average_item_quantity.rename(columns={'user_item_quantity': 'overall_average_item_quantity'}, inplace=True)
    return overall_average_item_quantity

def user_total_quantity(df):
    """количество покупок товаров пользователем"""
    user_total_quantity=df.groupby(['user_id'])['quantity'].count().reset_index()
    user_total_quantity.rename(columns={'quantity': 'user_total_quantity'}, inplace=True)
    return user_total_quantity

def avr_total_quantity(user_total_quantity):
    avr_total_quantity=user_total_quantity['user_total_quantity'].mean()
    return avr_total_quantity

def user_item_frequency(df):
    '''overall_item_frequency'''
    user_baskets_count=df.groupby(['user_id'])['basket_id'].count().reset_index()
    user_item_freq=df.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    user_item_freq=user_item_freq.merge(user_baskets_count, on='user_id', how='left')
    user_item_freq["item_frequency"]=user_item_freq["quantity"]/user_item_freq["basket_id"]
    user_item_freq=user_item_freq.drop(["quantity","basket_id"], axis=1)
    return user_item_freq

def overall_item_frequency(user_item_frequency):
    overall_item_freq=user_item_frequency.groupby(['item_id'])['item_frequency'].mean().reset_index()
    overall_item_freq.rename(columns={'item_frequency': 'overall_item_frequency'}, inplace=True)
    return overall_item_freq

def user_comodity_popularity(df):
    comodity_popularity=df.groupby(['user_id', 'sub_commodity_desc'])['quantity'].sum().reset_index()
    comodity_popularity.rename(columns={'quantity': 'user_comodity_popularity'}, inplace=True)
    return comodity_popularity

def overall_comodity_popularity(user_comodity_popularity):
    overall_comodity_popularity=user_comodity_popularity.groupby(['sub_commodity_desc'])['user_comodity_popularity'].mean().reset_index()
    overall_comodity_popularity.rename(columns={'user_comodity_popularity': 'overall_comodity_popularity'}, inplace=True)
    return overall_comodity_popularity

def hh_comp_comodity_popularity(df):
    commodity_quantity=df.groupby(['commodity_desc'])['quantity'].sum().reset_index()
    hh_comp_comodity=df.groupby(['commodity_desc', 'hh_comp_desc'])['quantity'].sum().reset_index()
    hh_comp_comodity=hh_comp_comodity.merge(commodity_quantity, on='commodity_desc', how='left')
    hh_comp_comodity['hh_comp_comodity_popularity']=hh_comp_comodity['quantity_x']/hh_comp_comodity['quantity_y']
    hh_comp_comodity['hh_comp_comodity_popularity'].fillna(0, inplace=True)
    return hh_comp_comodity


def add_part1_features(df, users, item_features, user_features, user_average_bill, overall_average_bill, overall_average_item_quantity, user_item_quantity,
                      overall_item_frequency,user_item_frequency, user_total_quantity, avr_total_quantity):

    targets = df[['user_id', 'item_id']].copy()
    targets['target'] = 1  # тут только покупки 
    targets = users.merge(targets, on=['user_id', 'item_id'], how='left')
    targets['target'].fillna(0, inplace= True)
    targets.drop('flag', axis=1, inplace=True)
    targets = targets.merge(item_features, on='item_id', how='left')
    targets = targets.merge(user_features, on='user_id', how='left')
    targets = targets.merge(user_average_bill, on='user_id', how='left')
    targets['user_average_bill'].fillna(overall_average_bill, inplace= True)
    targets = targets.merge(overall_average_item_quantity, on=['item_id'], how='left')
    targets = targets.merge(user_item_quantity, on=['user_id', 'item_id'], how='left')
    targets['user_item_quantity'].fillna(targets['overall_average_item_quantity'], inplace= True)
    targets['user_item_quantity'].fillna(0, inplace= True)
    targets=targets.drop('overall_average_item_quantity', axis=1)
    targets = targets.merge(overall_item_frequency, on=['item_id'], how='left')
    targets = targets.merge(user_item_frequency, on=['user_id', 'item_id'], how='left')
    targets['item_frequency'].fillna(targets['overall_item_frequency'], inplace= True)
    targets['item_frequency'].fillna(0, inplace= True)
    targets['overall_item_frequency'].fillna(0, inplace= True)
    #targets=targets.drop('overall_item_frequency', axis=1)
    targets = targets.merge(user_total_quantity, on=['user_id'], how='left')
    targets['user_total_quantity'].fillna(avr_total_quantity, inplace= True)
    targets['user_log_quantity']=np.log(targets['user_item_quantity']+1)

 
    return targets

def add_part2_features(targets, overall_comodity_popularity, user_comodity_popularity):
    targets=targets.merge(overall_comodity_popularity, on=['sub_commodity_desc'], how='left')

    targets=targets.merge(user_comodity_popularity, on=['user_id', 'sub_commodity_desc'], how='left')
 
    targets['user_comodity_popularity'].fillna(targets['overall_comodity_popularity'], inplace= True)
 
    #targets=targets.drop('overall_comodity_popularity', axis=1)

    return targets

def targets_fillblanks(targets, hh_comp_comodity_popularity):
    targets['age_desc'].fillna('45-54', inplace= True)
    targets['marital_status_code'].fillna('U', inplace= True)
    targets['income_desc'].fillna('50-74K', inplace= True)
    targets['household_size_desc'].fillna('2', inplace= True)
    targets['kid_category_desc'].fillna('None/Unknown', inplace= True)
    targets['hh_comp_desc'].fillna('2 Adults No Kids', inplace= True)
    targets['homeowner_desc'].fillna('Homeowner', inplace= True)
    targets = targets.merge(hh_comp_comodity_popularity, on=['commodity_desc', 'hh_comp_desc'], how='left')
    targets['hh_comp_comodity_popularity'].fillna(0, inplace=True)
    targets=targets.drop(['quantity_x','quantity_y'] , axis=1)
    targets['item_norm_quantity']=targets['user_item_quantity']/targets['user_total_quantity']
 
    return targets