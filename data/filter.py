import gzip
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import tqdm
import json

def load_python_dict_gz(file_path, head=None,key=None):
    data = []
    count = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if not line:
                continue
            record = ast.literal_eval(line)
            if key is not None and key not in record:
                continue
            import pdb; pdb.set_trace()
            data.append(record)
            count += 1
            if head is not None and count >= head:
                break

    return data


def save_python_dict_gz(data, file_path):
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        for record in data:
            f.write(str(record))  # <-- Python dict 字面量字符串
            f.write("\n")

bundle_path = './bundle_data.json.gz'
review_path = '../steam_reviews.json.gz'
item_path = './australian_users_items.json.gz'
game_path = './steam_games.json.gz'

# bundles = load_python_dict_gz(bundle_path)
# ['bundle_final_price', 'bundle_url', 'bundle_price', 'bundle_name', 'bundle_id', 'items', 'bundle_discount']

reviews = load_python_dict_gz(review_path)
# import pdb; pdb.set_trace()


#88310
users = load_python_dict_gz(item_path,key='user_id')

# ['user_id', 'items_count', 'steam_id', 'user_url', 'items']

#32135
games = load_python_dict_gz(game_path,key='id')

users_dict = {user['user_id']: user for user in users}
games_dict = {game['id']: game for game in games}

new_users = {}
new_games = {}
new_reviews = []
count = 0

# save_python_dict_gz(list(users_dict.values()), './filtered_users_items.json.gz')
# save_python_dict_gz(list(games_dict.values()), './filtered_steam_games.json.gz')

with gzip.open('./australian_user_reviews.json.gz', 'rt', encoding='utf-8') as f:
    for line in tqdm.tqdm(f):
        line = line.strip()
        if not line:
            continue
        record = ast.literal_eval(line)
        if len(record['reviews']) == 0:
            continue
        f_review = record['reviews'][0]
        user_id = record['user_id']
        item_id = f_review['item_id']

        if user_id in users_dict and item_id in games_dict:
            if user_id not in new_users:
                new_users[user_id] = users_dict[user_id]
            if item_id not in new_games:
                new_games[item_id] = games_dict[item_id]
            new_reviews.append(record)

print(len(new_reviews)) #23538
print(len(list(new_users.values()))) #23250
print(len(list(new_games.values()))) #2143
save_python_dict_gz(new_reviews, './filtered_user_reviews.json.gz')
save_python_dict_gz(list(new_users.values()), './filtered_users_items.json.gz')
save_python_dict_gz(list(new_games.values()), './filtered_steam_games.json.gz')