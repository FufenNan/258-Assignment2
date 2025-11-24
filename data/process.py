import gzip
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import tqdm
import pickle

def load_python_dict_gz(file_path, head=None):
    data = []
    count = 0
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            if not line:
                continue
            record = ast.literal_eval(line)

            data.append(record)
            count += 1
            if head is not None and count >= head:
                break

    return data

bundle_path = './/bundle_data.json.gz'
review_path = '../filtered_user_reviews.json.gz'
item_path = '../filtered_users_items.json.gz'
game_path = '../filtered_steam_games.json.gz'

# bundles = load_python_dict_gz(bundle_path)
# ['bundle_final_price', 'bundle_url', 'bundle_price', 'bundle_name', 'bundle_id', 'items', 'bundle_discount']

# ['user_id', 'user_url', 'reviews']
# print(review[0].keys())


# user
# user_id 76561197970982479
# items_count 277
# steam_id 76561197970982479
# user_url http://steamcommunity.com/profiles/76561197970982479
# items [{'item_id': '10', 'item_name': 'Counter-Strike', 'playtime_forever': 6, 'playtime_2weeks': 0}, {'item_id': '20', 'item_name': 'Team Fortress Classic', 'playtime_forever': 0, 'playtime_2weeks': 0}]


# review
# user_id 76561197970982479
# user_url http://steamcommunity.com/profiles/76561197970982479
# reviews [{'funny': '', 'posted': 'Posted November 5, 2011.', 'last_edited': '', 'item_id': '1250', 'helpful': 'No ratings yet', 'recommend': True, 'review': 'Simple yet with great replayability. In my opinion does "zombie" hordes and team work better than left 4 dead plus has a global leveling system. Alot of down to earth "zombie" splattering fun 
# for the whole family. Amazed this sort of FPS is so rare.'}, {'funny': '', 'posted': 'Posted July 15, 2011.', 'last_edited': '', 'item_id': '22200', 'helpful': 'No ratings yet', 'recommend': True, 'review': "It's unique and worth a playthrough."}, {'funny': '', 'posted': 'Posted April 21, 2011.', 'last_edited': '', 'item_id': '43110', 'helpful': 'No ratings yet', 'recommend': True, 'review': 'Great atmosphere. The gunplay can be a bit chunky at times but at the end of the day this game is definitely worth it and I hope 
# they do a sequel...so buy the game so I get a sequel!'}]

# game
# publisher Kotoshiro
# genres ['Action', 'Casual', 'Indie', 'Simulation', 'Strategy']
# app_name Lost Summoner Kitty
# title Lost Summoner Kitty
# url http://store.steampowered.com/app/761140/Lost_Summoner_Kitty/
# release_date 2018-01-04
# tags ['Strategy', 'Action', 'Indie', 'Casual', 'Simulation']
# discount_price 4.49
# reviews_url http://steamcommunity.com/app/761140/reviews/?browsefilter=mostrecent&p=1
# specs ['Single-player']
# price 4.99
# early_access False
# id 761140
# developer Kotoshiro

# print(bundles[0].keys())
#88310
users = load_python_dict_gz(item_path,head=100)
# ['user_id', 'items_count', 'steam_id', 'user_url', 'items']
#32135
games = load_python_dict_gz(game_path,head=100)
reviews = load_python_dict_gz(review_path)

print(users[0].keys())
print(reviews[0].keys())
print(games[0].keys())

all_genres = set()
all_specs = set()
all_tags = set()
game_dict = {}

for g in games:
    if 'id' not in g:
        continue
    game_dict[g['id']] = g
    all_genres.update(g.get('genres', []))
    all_specs.update(g.get('specs', []))
    all_tags.update(g.get('tags', []))

genres_to_idx = {g: i for i, g in enumerate(sorted(all_genres))}
specs_to_idx = {s: i for i, s in enumerate(sorted(all_specs))}
tags_to_idx = {t: i for i, t in enumerate(sorted(all_tags))}

review_per_user = defaultdict(int)
recommand_per_user = defaultdict(int)
review_per_item = defaultdict(int)
recommand_per_item = defaultdict(int)
recommand_cnt = 0
for r in reviews:
    first_r = r['reviews'][0] if r['reviews'] else None
    if first_r:
        review_per_user[r['user_id']] += 1
        review_per_item[first_r['item_id']] += 1
        if first_r['recommend'] == True:
            recommand_cnt += 1
            recommand_per_user[r['user_id']] += 1
            recommand_per_item[first_r['item_id']] += 1

average_recommand = recommand_cnt / len(reviews)
print(f'average_recommand: {average_recommand}')
for k,v in recommand_per_user.items():
    recommand_per_user[k] = v / review_per_user[k] if review_per_user[k]>0 else 0.0
for k,v in recommand_per_item.items():
    recommand_per_item[k] = v / review_per_item[k] if review_per_item[k]>0 else 0.0

user_features = {}
for u in users:
    feature = {}
    total_playtime_forever = 0
    total_playtime_2weeks = 0
    total_price = 0.0
    geners_vec = np.zeros(len(genres_to_idx), dtype=np.float32)
    specs_vec = np.zeros(len(specs_to_idx), dtype=np.float32)
    tags_vec = np.zeros(len(tags_to_idx), dtype=np.float32)

    for item in u.get('items', []):
        total_playtime_forever += item.get('playtime_forever', 0)
        total_playtime_2weeks += item.get('playtime_2weeks', 0)
        price_raw = game_dict.get(item['item_id'], {}).get('price', 0.0)
        try:
            price = float(price_raw)
        except (ValueError, TypeError):
            price = 0.0
        total_price += price
        for g in game_dict.get(item['item_id'], {}).get('genres', []):
            if g in genres_to_idx:
                geners_vec[genres_to_idx[g]] += 1.0
        for s in game_dict.get(item['item_id'], {}).get('specs', []):
            if s in specs_to_idx:
                specs_vec[specs_to_idx[s]] += 1.0
        for t in game_dict.get(item['item_id'], {}).get('tags', []):
            if t in tags_to_idx:
                tags_vec[tags_to_idx[t]] += 1.0
    tmp = u.get('items', [])
    feature['items_count'] = u.get('items_count', 0)
    feature['playtime_forever'] = total_playtime_forever/feature['items_count'] if feature['items_count']>0 else 0
    feature['playtime_2weeks'] = total_playtime_2weeks/feature['items_count'] if feature['items_count']>0 else 0
    feature['average_price'] = total_price/feature['items_count'] if feature['items_count']>0 else 0
    feature['genres_vec'] = geners_vec
    feature['specs_vec'] = specs_vec
    feature['tags_vec'] = tags_vec
    # No recommendation ratio because the data sparsity
    user_features[u['user_id']] = feature

print(len(user_features))
with open('user_features.pkl', 'wb') as f:
    pickle.dump(user_features, f, protocol=pickle.HIGHEST_PROTOCOL)

game_features = {}
for game in games:
    feature = {}
    genres_vec = np.zeros(len(genres_to_idx), dtype=np.float32)
    specs_vec = np.zeros(len(specs_to_idx), dtype=np.float32)
    tags_vec = np.zeros(len(tags_to_idx), dtype=np.float32)

    for g in game.get('genres', []):
        if g in genres_to_idx:
            genres_vec[genres_to_idx[g]] = 1.0
    for s in game.get('specs', []):
        if s in specs_to_idx:
            specs_vec[specs_to_idx[s]] = 1.0
    for t in game.get('tags', []):
        if t in tags_to_idx:
            tags_vec[tags_to_idx[t]] = 1.0
    price_raw = game_dict.get(game['id'], {}).get('price', 0.0)
    try:
        price = float(price_raw)
    except (ValueError, TypeError):
        price = 0.0
    feature['price'] = price
    feature['genres_vec'] = genres_vec
    feature['specs_vec'] = specs_vec
    feature['tags_vec'] = tags_vec
    feature['review_count'] = review_per_item.get(game['id'], 0)
    feature['recommend_ratio'] = recommand_per_item.get(game['id'], average_recommand)
    game_features[game['id']] = feature

print(len(game_features))
with open('game_features.pkl', 'wb') as f:
    pickle.dump(game_features, f, protocol=pickle.HIGHEST_PROTOCOL)

play_time_forever = {}
play_time_2weeks = {}
for u in users:
    for item in u.get('items', []):
        play_time_forever[(u['user_id'], item['item_id'])] = item.get('playtime_forever', 0)
        play_time_2weeks[(u['user_id'], item['item_id'])] = item.get('playtime_2weeks', 0)

training_data = []
for r in reviews:
    x = {}
    first_r = r['reviews'][0]
    user_id = r['user_id']
    item_id = first_r['item_id']
    user_feature = user_features[user_id]
    game_feature = game_features[item_id]
    cross_feature = {}
    cross_feature['playtime_forever'] = play_time_forever.get((user_id, item_id), 0)
    cross_feature['playtime_2weeks'] = play_time_2weeks.get((user_id, item_id), 0)

    x['user_feature'] = user_feature
    x['game_feature'] = game_feature
    x['cross_feature'] = cross_feature
    x['label'] = 1 if first_r['recommend'] == True else 0
    training_data.append(x)

with open('training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

for i, x in enumerate(training_data[:2]):
    print(f"Sample {i}:")
    for k, v in x['user_feature'].items():
        if isinstance(v, np.ndarray):
            print(f"User feature - {k}: array of shape {v.shape}")
        else:
            print(f"User feature - {k}: {v}")
    for k, v in x['game_feature'].items():
        if isinstance(v, np.ndarray):
            print(f"Game feature - {k}: array of shape {v.shape}")
        else:
            print(f"Game feature - {k}: {v}")
    for k, v in x['cross_feature'].items():
        print(f"Cross feature - {k}: {v}")
    print(f"Label: {x['label']}")
    print("-----")

# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# class SteamDataset(Dataset):
#     def __init__(self, users, games, reviews=None, max_items=100):
#         """
#         users: list of user dicts, each with keys 'user_id', 'items_count', 'items'
#         games: list of game dicts, each with keys 'id', 'genres', 'specs', 'tags', 'price'
#         reviews: list of review dicts (optional), each with 'user_id', 'reviews'
#         """
#         self.users = users
#         self.reviews = {r['user_id']: r['reviews'] for r in reviews} if reviews else {}
#         self.max_items = max_items

#         # 1. 构建 game_id -> game_info 的映射
#         self.game_dict = defaultdict(dict)
#         all_genres = set()
#         all_specs = set()
#         all_tags = set()

#         for g in games:
#             if 'id' not in g:
#                 continue
#             self.game_dict[g['id']] = g
#             all_genres.update(g.get('genres', []))
#             all_specs.update(g.get('specs', []))
#             all_tags.update(g.get('tags', []))

#         # 2. 生成 one-hot 编码索引
#         self.genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
#         self.spec2idx = {s: i for i, s in enumerate(sorted(all_specs))}
#         self.tag2idx = {t: i for i, t in enumerate(sorted(all_tags))}

#         self.num_genres = len(self.genre2idx) #12
#         self.num_specs = len(self.spec2idx) #31
#         self.num_tags = len(self.tag2idx) #148
#         import pdb; pdb.set_trace()
#         # 特征维度
#         # base 4 + max_items*(price + genres_onehot + specs_onehot + tags_onehot)
#         self.feature_dim = 4 + max_items * (1 + self.num_genres + self.num_specs + self.num_tags)

#     def encode_game_features(self, game_id):
#         game = self.game_dict.get(game_id, {})
#         # price
#         price = game.get('price', 0.0)

#         # genres one-hot
#         genres_vec = np.zeros(self.num_genres, dtype=np.float32)
#         for g in game.get('genres', []):
#             if g in self.genre2idx:
#                 genres_vec[self.genre2idx[g]] = 1.0

#         # specs one-hot
#         specs_vec = np.zeros(self.num_specs, dtype=np.float32)
#         for s in game.get('specs', []):
#             if s in self.spec2idx:
#                 specs_vec[self.spec2idx[s]] = 1.0

#         # tags one-hot
#         tags_vec = np.zeros(self.num_tags, dtype=np.float32)
#         for t in game.get('tags', []):
#             if t in self.tag2idx:
#                 tags_vec[self.tag2idx[t]] = 1.0

#         return np.concatenate([[price], genres_vec, specs_vec, tags_vec], axis=0)

#     def __len__(self):
#         return len(self.users)

#     def __getitem__(self, idx):
#         user = self.users[idx]
#         user_id = user['user_id']

#         # 基础特征
#         items_count = user.get('items_count', 0)
#         total_playtime_forever = sum(item.get('playtime_forever', 0) for item in user.get('items', []))
#         total_playtime_2weeks = sum(item.get('playtime_2weeks', 0) for item in user.get('items', []))
#         num_reviews = len(self.reviews.get(user_id, []))

#         base_features = [items_count, total_playtime_forever, total_playtime_2weeks, num_reviews]

#         # 每个 item 特征
#         item_features = []
#         items = user.get('items', [])
#         for i in range(self.max_items):
#             if i < len(items):
#                 item_id = items[i]['item_id']
#                 item_feat = self.encode_game_features(item_id)
#                 item_features.append(item_feat)
#             else:
#                 # pad with zeros
#                 item_features.append(np.zeros(1 + self.num_genres + self.num_specs + self.num_tags, dtype=np.float32))

#         features = np.concatenate([base_features] + item_features, axis=0)
#         return user_id, torch.tensor(features, dtype=torch.float32)
    

# # 使用示例
# dataset = SteamDataset(user, games=game, reviews=review, max_items=10)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# for user_ids, features in dataloader:
#     print(user_ids)
#     print(features.shape)
#     break