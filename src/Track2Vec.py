import numpy as np
import pandas as pd
from reclist.abstractions import RecModel
from gensim.models import Word2Vec
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers 
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import itertools
import time
from src.bi_lstm import BiLSTM


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

seed = 27
set_random_seed(seed)

class Track2Vec(RecModel):

    def __init__(self, items, users, top_k: int=100, vector_size: int=300, epochs: int=10, **kwargs):
        super(Track2Vec, self).__init__()
        """
        :param top_k: numbers of recommendation to return for each user. Defaults to 20.
        """
        # get tracks df
        self.items = items
        self.unique_track_ids = items.index.unique()
        
        # define mapping between track_ids to tokens
        self.track_id_to_token = {track_id: i for i, track_id in enumerate(self.unique_track_ids)}
        self.token_to_track_id = {i: track_id for track_id, i in self.track_id_to_token.items()}
        self.vocab_size = len(self.unique_track_ids)
        
        self.users = self._convert_user_info(users)
        self.top_k = top_k
        self.mappings = None
        self.epochs = epochs
        self.vector_size = vector_size

    def _convert_user_info(self, user_info):
        user_info['gender'] = user_info['gender'].fillna(value='n')
        user_info['country'] = user_info['country'].fillna(value='UNKNOWN')
        user_info['playcount'] = user_info['playcount'].fillna(value=0)

        return user_info
    
    def _update_sequences_with_tokens(self, sequences):
        return [[self.track_id_to_token.get(track_id, 0) for track_id in sequence] for sequence in sequences]
    
    def _prepare_data_for_bilstm(self, sequences):
        X, y = [], []
        
        # create sequences for training and labels
        for sequence in sequences:
            for i in range(1, len(sequence)):
                X.append(sequence[:i])
                y.append(sequence[i])
                
        # update sequences with tokens
        X_tokens = self._update_sequences_with_tokens(X)
        y_tokens = np.array([self.track_id_to_token[track_id] for track_id in y])

        # pad sequences
        max_sequence_length = max(len(x) for x in X_tokens)
        X_padded = pad_sequences(X_tokens, maxlen=max_sequence_length)
        
        return X_padded, y_tokens, max_sequence_length
    

    def train_playcount(self, df):
        p_1 = df[df['playcount'] <= 10].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_2 = df[(10 < df['playcount']) & (df['playcount'] <= 100)].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_3 = df[(100 < df['playcount']) & (df['playcount'] <= 1000)].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_4 = df[1000 < df['playcount']].groupby(['user_id'], sort=False)['track_id'].agg(list)
        
        # prepare data for biltsm models
        p_1_X, p_1_y, p_1_max_sequence_length = self._prepare_data_for_bilstm(p_1.tolist())
        # p_2_X, p_2_y, p_2_max_sequence_length = self._prepare_data_for_bilstm(p_2.tolist())
        # p_3_X, p_3_y, p_3_max_sequence_length = self._prepare_data_for_bilstm(p_3.tolist())
        # p_4_X, p_4_y, p_4_max_sequence_length = self._prepare_data_for_bilstm(p_4.tolist())
        
        # Train BiLSTM models WITHIN each segment
        self.model_1 = BiLSTM(vocab_size=self.vocab_size, 
                            max_sequence_length=p_1_max_sequence_length,
                            learning_rate=0.001,
                            vector_size=self.vector_size)
        # self.model_2 = BiLSTM(p_2.values.tolist(), p_2_max_sequence_length, 0.001, self.vector_size)    
        # self.model_3 = BiLSTM(p_3.values.tolist(), p_3_max_sequence_length, 0.001, self.vector_size)
        # self.model_4 = BiLSTM(p_4.values.tolist(), p_4_max_sequence_length, 0.001, self.vector_size)
        
        # train models cuncurrently
        self.model_1.train(p_1_X, p_1_y, batch_size=32, epochs=self.epochs)
        # self.model_2.train(p_2_X, p_2_y, batch_size=32, epochs=self.epochs)
        # self.model_3.train(p_3_X, p_3_y, batch_size=32, epochs=self.epochs)
        # self.model_4.train(p_4_X, p_4_y, batch_size=32, epochs=self.epochs)

        
        p = pd.concat([p_1, p_2, p_3, p_4])

        user_tracks = pd.DataFrame(p)
        user_tracks['playcount_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40)) 
        self.mappings = user_tracks.T.to_dict() # {"user_k" : {"track_id" : [...], "track_id_sampled" : [...]}}

    def train_gender(self, df):
        p_m = df[df['gender'] == 'm'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_f = df[df['gender'] == 'f'].groupby(['user_id'], sort=False)['track_id'].agg(list)
        p_n = df[(df['gender'] != 'm') & (df['gender'] != 'f')].groupby(['user_id'], sort=False)['track_id'].agg(list)

        self.mymodel_m = Word2Vec(p_m.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_f = Word2Vec(p_f.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_n = Word2Vec(p_n.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        
        p = pd.concat([p_m, p_f, p_n])

        user_tracks = pd.DataFrame(p)
        user_tracks['gender_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40))
        gender_dict = user_tracks.T.to_dict()
        for key in self.mappings.keys():
            self.mappings[key]['gender_track_id_sampled'] = gender_dict[key]['gender_track_id_sampled']

    def train_user_track_count(self, df):
        df_trackid = df.groupby(['user_id'], sort=False)['track_id'].agg(list)
        df = pd.DataFrame(df_trackid).join(df.groupby('user_id', as_index=True, sort=False)[['user_track_count']].sum(), on='user_id', how='left')
        self.train_df = df
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')

        p_1 = df[df['user_track_count'] <= 100]['track_id']
        p_2 = df[(100 < df['user_track_count']) & (df['user_track_count'] <= 1000)]['track_id']
        p_3 = df[1000 < df['user_track_count']]['track_id']

        self.mymodel_utc1 = Word2Vec(p_1.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_utc2 = Word2Vec(p_2.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)
        self.mymodel_utc3 = Word2Vec(p_3.values.tolist(), min_count=0, vector_size=self.vector_size, window=self.window, negative=self.negative, epochs=self.epochs, sg=0, workers=4)

        p = pd.concat([p_1, p_2, p_3])

        user_tracks = pd.DataFrame(p)
        user_tracks['utc_track_id_sampled'] = user_tracks['track_id'].apply(lambda x : random.choices(x, k=40))
        utc_dict = user_tracks.T.to_dict()
        for key in self.mappings.keys():
            self.mappings[key]['utc_track_id_sampled'] = utc_dict[key]['utc_track_id_sampled']

    def train(self, train_df: pd.DataFrame):
        df = train_df[['user_id', 'track_id', 'timestamp', 'user_track_count']].sort_values('timestamp')
        df = pd.DataFrame(df).join(self.users, on='user_id', how='left')
        
        self.train_playcount(df)
        # self.train_gender(df)
        # self.train_user_track_count(df)

    def pred_playcount(self, user, user_playcount, user_tracks):
    
        # Choose the right model based on user playcount
        # if user_playcount <= 10:
        #     model = self.model_1
        # elif 10 < user_playcount <= 100:
        #     model = self.model_2
        # elif 100 < user_playcount <= 1000:
        #     model = self.model_3
        # else:
        #     model = self.model_4

        # Predict the next track(s) using the model
        predictions = self.model_1.predict_next_n_tracks(user_tracks, self.token_to_track_id, n=self.top_k)
        
        return predictions

    def pred_gender(self, user, user_gender, user_tracks):
        if user_gender == 'm':
            get_user_embedding = np.mean([self.mymodel_m.wv[t] for t in user_tracks if t in self.mymodel_m.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k 
            user_predictions = [k[0] for k in self.mymodel_m.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif user_gender == 'f':
            get_user_embedding = np.mean([self.mymodel_f.wv[t] for t in user_tracks if t in self.mymodel_f.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k
            user_predictions = [k[0] for k in self.mymodel_f.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        else:
            get_user_embedding = np.mean([self.mymodel_n.wv[t] for t in user_tracks if t in self.mymodel_n.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k
            user_predictions = [k[0] for k in self.mymodel_n.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]

        user_predictions = list(filter(lambda x: x not in 
                                        self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
        return user_predictions

    def pred_user_track_count(self, user, user_track_count, user_tracks):
        if user_track_count <= 100:
            get_user_embedding = np.mean([self.mymodel_utc1.wv[t] for t in user_tracks if t in self.mymodel_utc1.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc1.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        elif 100 < user_track_count and user_track_count <= 1000:
            get_user_embedding = np.mean([self.mymodel_utc2.wv[t] for t in user_tracks if t in self.mymodel_utc2.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc2.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]
        else:
            get_user_embedding = np.mean([self.mymodel_utc3.wv[t] for t in user_tracks if t in self.mymodel_utc3.wv], axis=0)
            max_number_of_returned_items = len(self.mappings[user]["track_id"]) + self.top_k # filter out stuff from the user history
            user_predictions = [k[0] for k in self.mymodel_utc3.wv.most_similar(positive=[get_user_embedding], 
                                                                        topn=max_number_of_returned_items)]

        user_predictions = list(filter(lambda x: x not in 
                                        self.mappings[user]["track_id"], user_predictions))[0:self.top_k]
        return user_predictions

    def ensemble(self, pred_1, pred_2, pred_3):
        all_pred = list(itertools.chain(pred_1, pred_2, pred_3))
        counter = Counter(all_pred)
        counter_top_k = counter.most_common(self.top_k)
        pred = []
        for tuple in counter_top_k:
            if tuple[1] > 1:
                pred.append(tuple[0])

        for i in range(self.top_k):
            if pred_1[0] not in pred:
                pred.append(pred_1.pop(0))
            else:
                pred_1.pop(0)
            if pred_2[0] not in pred:
                pred.append(pred_2.pop(0))
            else:
                pred_2.pop(0)
            if pred_3[0] not in pred:
                pred.append(pred_3.pop(0))
            else:
                pred_3.pop(0)
        
        return pred[:100]

    def predict(self, user_ids: pd.DataFrame):
        user_ids = user_ids.copy()
        k = self.top_k
        
        pbar = tqdm(total=len(user_ids), position=0)
        predictions = []
        valid_user_ids = []  # List to collect valid user IDs
        for user in user_ids['user_id']:
            if user not in self.mappings:
                pbar.update(1)
                continue
            valid_user_ids.append(user)  # Add the valid user ID to the list
            user_tracks_playcount = self.mappings[user]['playcount_track_id_sampled']
            # user_tracks_gender = self.mappings[user]['gender_track_id_sampled']
            # user_tracks_utc = self.mappings[user]['utc_track_id_sampled']

            user_playcount = self.users.loc[[user]]['playcount'].values[0]
            # user_gender = self.users.loc[[user]]['gender'].values[0]
            # user_track_count = self.train_df.loc[user]['user_track_count']

            pred_1 = self.pred_playcount(user, user_playcount, user_tracks_playcount)
            # pred_2 = self.pred_gender(user, user_gender, user_tracks_gender)
            # pred_3 = self.pred_user_track_count(user, user_track_count, user_tracks_utc)

            # user_predictions = self.ensemble(pred_1, pred_2, pred_3)
            # predictions.append(user_predictions)
            predictions.append(pred_1)


            pbar.update(1)
        pbar.close()

        # users = user_ids["user_id"].values.reshape(-1, 1)
        valid_users_array = np.array(valid_user_ids).reshape(-1, 1)
        predictions = np.array(predictions)
        # predictions = np.concatenate([users, predictions], axis=1)
        predictions = np.concatenate([valid_users_array, predictions], axis=1)

        return pd.DataFrame(predictions, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')
