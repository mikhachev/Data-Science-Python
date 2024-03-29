import pandas as pd
import numpy as np

def prefilter_items(data_train: pd.DataFrame, item_features, take_n_popular):
    
    
    # Уберем самые популярные товары (их и так купят)
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
#     top_popular = popularity[popularity['share_unique_users'] > 0.8].item_id.tolist()
#     data_train = data_train[~data_train['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.1].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_notpopular)]
    
    
    
    # Уберем товары, которые не продавались за последние 12 месяцев
#     item_last_day=data_train.groupby('item_id')['day'].max().reset_index()
#     max_day=item_last_day['day'].max()
#     fresh_items=item_last_day[item_last_day['day']>(max_day-360)].item_id.tolist()
#     data_train = data_train[data_train['item_id'].isin(fresh_items)]
    
    
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                        groupby('department')['item_id'].nunique().\
                                        sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data_train = data_train[~data_train['item_id'].isin(items_in_rare_departments)]
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data_train['price'] = data_train['sales_value'] / (np.maximum(data_train['quantity'], 1))
    data_train = data_train[data_train['price'] > 2]
    
    # Уберем слишком дорогие товарыs
    data_train = data_train[data_train['price'] < 50]
    
    # Возбмем топ по популярности
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-N000, то он "купил" такой товар)
    data_train.loc[~data_train['item_id'].isin(top), 'item_id'] = 999999
    
    return data_train
    
def postfilter_items(user_id, recommednations):
    pass