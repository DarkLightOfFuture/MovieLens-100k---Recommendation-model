import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from lightfm.data import Dataset
from lightfm import LightFM
import random

class RecommendationModel:
    def __init__(self, list_size = 15, emb_size = 64, is_test = False, simulation_learning_rate = 9e-5):
        self.list_size = list_size
        self.is_test = is_test
        self.emb_size = emb_size
        self.model = None
        self.init_df_size = None

        self._load_data()
        self._simulate_records(n=15_000, learning_rate=simulation_learning_rate)

        self._build_encoders_scalers()
        self._build_features_inputs()

        self._train_test_split()

    def _load_data(self):
        self.users_df = pd.read_csv("./data/userInfo.csv")
        self.movies_df = pd.read_csv("./data/movieInfo.csv", parse_dates=["release date"], dayfirst=True)
        self.reviews_df = pd.read_csv("./data/reviewInfo.csv")

        self.genres = self.movies_df.columns.tolist()[3:]

        # inner join on dataframes
        self.df = pd.merge(self.users_df, self.reviews_df, left_on="id", right_on="user id")
        self.df = self.df.drop(["id"], axis=1)

        self.df = pd.merge(self.df, self.movies_df, left_on="movie id", right_on="id")
        self.df = self.df.drop(["id"], axis=1)

        self.df['weighted rating'] = self.df['rating'] * self.df['watched']
        self.init_df_size = len(self.df)

    def _prepare_sgd_features(self):
        user_features = set()
        for _, row in self.df.iterrows():
            user_features.add(f"age_{row['age']}")
            user_features.add(f"occp_{row['occupation']}")
            user_features.add(f"gender_{row['gender']}")

        movie_features = set()
        for _, row in self.df.iterrows():
            movie_features.add(f"year_{row['release date'].year}")

            for genre in self.genres:
                movie_features.add(f"{genre}_{row[genre]}")

        return list(user_features), list(movie_features)

    def _get_user_features(self):
        user_features = {}
        for user_id in self.df["user id"].unique():
            user = self.df[self.df["user id"] == user_id].iloc[0]

            user_features[user_id] = [
                f"age_{user['age']}",
                f"occp_{user['occupation']}",
                f"gender_{user['gender']}"
            ]

        return user_features.items()

    def _get_movie_features(self):
        movie_features = {}
        for movie_id in self.df["movie id"].unique():
            movie = self.df[self.df["movie id"] == movie_id].iloc[0]

            movie_features[movie_id] = [f"year_{movie['release date'].year}"]
            movie_features[movie_id].extend([f"{genre}_{movie[genre]}" for genre in self.genres])

        return movie_features.items()

    def _simulate_records(self, n = 5000, learning_rate = 1e-4):
        user_features, movie_features = self._prepare_sgd_features()

        dataset = Dataset()
        dataset.fit(
            self.df["user id"].unique(), self.df["movie id"].unique(),
            user_features=user_features, item_features=movie_features
        )

        def create_interactions(val):
            return dataset.build_interactions(
                [(row["user id"], row['movie id'], row[val]) for _, row in self.df.iterrows()]
            )

        rating_interactions, _ = create_interactions("rating")
        watched_interactions, _ = create_interactions("watched")

        user_features_matrix = dataset.build_user_features(
            [(user_id, user_features) for user_id, user_features in self._get_user_features()]
        )
        movie_features_matrix = dataset.build_item_features(
            [(movie_id, movie_features) for movie_id, movie_features in self._get_movie_features()]
        )

        rating_model = LightFM(no_components=100, loss="warp", random_state=42, learning_rate=learning_rate)
        watched_model = LightFM(no_components=100, loss="warp", random_state=42, learning_rate=learning_rate)

        rating_model.fit(
            rating_interactions, user_features=user_features_matrix,
            item_features=movie_features_matrix, epochs=20
        )
        watched_model.fit(
            watched_interactions, user_features=user_features_matrix,
            item_features=movie_features_matrix, epochs=20
        )

        predictions = []
        all_users = list(self.df['user id'].unique())
        all_movies = list(self.df['movie id'].unique())
        existing_pairs = set(zip(self.df['user id'], self.df['movie id']))

        user_id_map = {uid: df_id for df_id, uid in enumerate(all_users)}
        movie_id_map = {mid: df_id for df_id, mid in enumerate(all_movies)}

        all_pairs = [(user_id, movie_id) for user_id in all_users for movie_id in all_movies]
        #randomizing of order (diversity of simulated data)
        random.shuffle(all_pairs)

        amount = 0
        for (user_id, movie_id) in all_pairs:
            if amount <= n and (user_id, movie_id) not in existing_pairs:
                user_df_id = user_id_map[user_id]
                movie_df_id = movie_id_map[movie_id]

                def create_prediction(model):
                    return model.predict(
                        user_df_id, [movie_df_id], user_features=user_features_matrix,
                        item_features=movie_features_matrix
                    )[0]

                rating_pred = create_prediction(rating_model)
                watched_pred = create_prediction(watched_model)

                rating_pred = np.clip(rating_pred * 2 + 3, 1, 5) # y = 2x + 3
                watched_pred = np.clip(watched_pred * .5 + .5, 0, 1) # y = 0.5x + 0.5

                prediction = {
                    'user id': user_id,
                    'movie id': movie_id,
                    'rating': rating_pred,
                    'watched': watched_pred,
                    'weighted rating': rating_pred * watched_pred,
                    'timestamp': int(pd.Timestamp.now().timestamp())
                }

                for col, val in self.df[self.df['user id'] == user_id].iloc[0].items():
                    if col not in prediction:
                        prediction[col] = val

                predictions.append(prediction)
                amount += 1

        if predictions:
            simulated_df = pd.DataFrame(predictions)
            self.df = pd.concat([self.df, simulated_df], ignore_index=True)

    def _build_encoders_scalers(self):
        self.user_id_enc = LabelEncoder()
        self.user_id_enc.fit(self.df["user id"])

        self.movie_id_enc = LabelEncoder()
        self.movie_id_enc.fit(self.df["movie id"])

        self.gender_enc = LabelEncoder()
        self.gender_enc.fit(self.df['gender'])

        # occupation
        self.occp_enc = LabelEncoder()
        self.occp_enc.fit(self.df['occupation'])

        self.age_scal = MinMaxScaler()
        self.age_scal.fit(self.df[['age']])

        self.year_scal = MinMaxScaler()
        self.year_scal.fit(self.df["release date"].dt.year.to_frame())

        self.rating_scal = MinMaxScaler()
        self.rating_scal = self.rating_scal.fit(self.df[["rating"]])

    def _build_feature_input(self, name, encoder, is_scaler = False):
        inpt = layers.Input(shape = (self.list_size,), name=f"{name}_input")

        if not is_scaler:
            emb = layers.Embedding(len(encoder.classes_), self.emb_size, name=f"{name}_emb")(inpt)
            return inpt, emb
        else:
            expanded = layers.Lambda(lambda x: tf.expand_dims(x, -1))(inpt)
            return inpt, expanded

    def _build_features_inputs(self):
        user_id_input, user_id_emb = self._build_feature_input("user_id", self.user_id_enc)
        movie_id_input, movie_id_emb = self._build_feature_input("movie_id", self.movie_id_enc)
        gender_input, gender_emb = self._build_feature_input("gender", self.gender_enc)
        # occupation
        occp_input, occp_emb = self._build_feature_input("occp", self.occp_enc)

        age_input, age_expanded = self._build_feature_input("age", self.age_scal, True)
        year_input, year_expanded = self._build_feature_input("year", self.year_scal, True)
        rating_input, rating_expanded = self._build_feature_input("rating", self.rating_scal, True)

        genres_input = []
        genres_expanded = []

        for genre in self.genres:
            genre_input = layers.Input(shape = (self.list_size,), name = f"{genre}_input")
            genre_expanded = layers.Lambda(lambda x: tf.expand_dims(x, -1))(genre_input)

            genres_input.append(genre_input)
            genres_expanded.append(genre_expanded)

        self.user_features = layers.concatenate([
            user_id_emb, gender_emb, occp_emb, age_expanded
        ], axis=-1)

        self.item_features = layers.concatenate([
            movie_id_emb, year_expanded, rating_expanded
        ] + genres_expanded, axis=-1)

        self.model_input = [
            user_id_input, movie_id_input, gender_input, occp_input,
            age_input, year_input, rating_input
        ] + genres_input

    def _add_padding(self, df):
        needed = self.list_size - len(df)

        if needed > 0:
            watched_movies = df["movie id"].unique()
            available_movies = self.movies_df[~self.movies_df["id"].isin(watched_movies)]

            padding = available_movies.sample(n = needed)
            padding_cols = padding.columns + ["movie id"]

            user_row = df.iloc[0]

            for col in df.columns:
                if col in padding_cols:
                    continue
                else:
                    padding[col] = df[col].iat[0]

            padding['movie id'] = padding['id']
            padding['rating'] = 1
            padding['weighted rating'] = 1
            padding['watched'] = 1

            return pd.concat([df, padding], ignore_index=True).sort_values("weighted rating", ascending=False)
        elif needed < 0:
            return df.sample(n = self.list_size).sort_values("weighted rating", ascending=False)
        else:
            return df.sort_values("weighted rating", ascending=False)

    def _build_model_data(self, df, data):
        data["user_id_input"].append(self.user_id_enc.transform(df["user id"].values))
        data["movie_id_input"].append(self.movie_id_enc.transform(df["movie id"].values))
        data["occp_input"].append(self.occp_enc.transform(df["occupation"].values))
        data["gender_input"].append(self.gender_enc.transform(df["gender"].values))
        data["age_input"].append(self.age_scal.transform(df[["age"]]))
        data["year_input"].append(self.year_scal.transform(df["release date"].dt.year.to_frame()))
        data["rating_input"].append(self.rating_scal.transform(df[["rating"]]))
        data["weighted_rating"].append(df["weighted rating"])


        for col in self.genres:
            data[f'{col}_input'].append(df[col])

    def _train_test_split(self):
        user_movies_count = self.df.groupby("user id").size()
        valid_users = user_movies_count[np.ceil(user_movies_count * .2) >= 2].index # min_test ewentualnie dodaÄ‡
        self.df = self.df[self.df['user id'].isin(valid_users)]

        self.train_data = {
            "user_id_input": [],
            "movie_id_input": [],
            "occp_input": [],
            "gender_input": [],
            "age_input": [],
            "year_input": [],
            "rating_input": [],
            "weighted_rating": []
        }

        for col in self.genres:
            self.train_data[f'{col}_input'] = []

        self.test_data = {
            "user_id_input": [],
            "movie_id_input": [],
            "occp_input": [],
            "gender_input": [],
            "age_input": [],
            "year_input": [],
            "rating_input": [],
            "weighted_rating": []
        }

        for col in self.genres:
            self.test_data[f'{col}_input'] = []

        for user_id in valid_users:
            user_movies = self.df[self.df['user id'] == user_id].copy()
            user_movies = user_movies.sort_values(by=["timestamp"], ascending=[True])

            split = int(.8 * len(user_movies))
            train_df = self._add_padding(user_movies[:split])
            test_df = self._add_padding(user_movies[split:])

            self._build_model_data(train_df, self.train_data)
            self._build_model_data(test_df, self.test_data)

    def build(self, loss_func=tfr.keras.losses.ListMLELoss, learning_rate = 1e-3,
              optimizer=optimizers.Adam, activation="relu", neurons=64, dropout_rate=0.3):

        # User tower
        user_dense1 = layers.Dense(neurons, activation=activation)(self.user_features)
        user_bn1 = layers.BatchNormalization()(user_dense1)
        user_drop1 = layers.Dropout(dropout_rate)(user_bn1)
        user_dense2 = layers.Dense(neurons // 2, activation=activation)(user_drop1)

        # Item tower
        item_dense1 = layers.Dense(neurons, activation=activation)(self.item_features)
        item_bn1 = layers.BatchNormalization()(item_dense1)
        item_drop1 = layers.Dropout(dropout_rate)(item_bn1)
        item_dense2 = layers.Dense(neurons // 2, activation=activation)(item_drop1)

        # Interaction layer
        interaction = layers.multiply([user_dense2, item_dense2])

        # Final layers
        combined = layers.concatenate([user_dense2, item_dense2, interaction], axis=-1)

        dense1 = layers.Dense(neurons // 2, activation=activation)(combined)
        bn1 = layers.BatchNormalization()(dense1)
        drop1 = layers.Dropout(dropout_rate)(bn1)

        dense2 = layers.Dense(neurons // 4, activation=activation)(drop1)
        drop2 = layers.Dropout(dropout_rate * 0.5)(dense2)

        output = layers.Dense(1, activation="linear")(drop2)
        output = layers.Lambda(lambda x: tf.squeeze(x, -1))(output)

        self.model = tf.keras.Model(inputs=self.model_input, outputs=output)

        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss_func(),
            metrics=[tfr.keras.metrics.NDCGMetric(name="ndcg"),
                     tfr.keras.metrics.MeanAveragePrecisionMetric(name="map")])

        self.is_test and self.model.summary()

        return self._train()

    def _train(self):
        for key in self.train_data:
            self.train_data[key] = np.array(self.train_data[key])

        for key in self.test_data:
            self.test_data[key] = np.array(self.test_data[key])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3,
                factor=0.5,
                monitor='val_loss'
            )
        ]

        results = self.model.fit(
            self.train_data, self.train_data['weighted_rating'],
            batch_size=32,
            epochs=3 if self.is_test else 100,
            validation_data=(self.test_data, self.test_data['weighted_rating']),
            callbacks=callbacks,
            verbose=False
        )

        return results.history

    def recommend_movies(self, user_id_raw, top_n=10):
        if self.model == None:
            raise Exception("Model hasn't been built.")

        try:
            user_encoded_id = self.user_id_enc.transform([user_id_raw])[0]
            # Fetch user's features
            user_info = self.users_df[self.users_df['id'] == user_id_raw].iloc[0]
            gender_encoded = self.gender_enc.transform([user_info['gender']])[0]
            occp_encoded = self.occp_enc.transform([user_info['occupation']])[0]
            age_scaled = self.age_scal.transform([[user_info['age']]])[0][0]

        except (ValueError, IndexError):
            print(f"User with ID {user_id_raw} was not found.")
            return pd.DataFrame()

        watched_movies_ids = self.reviews_df.loc[self.reviews_df['user id'] == user_id_raw, ['movie id']].values.tolist()

        all_movie_ids = self.movies_df['id'].unique()
        unwatched_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in watched_movies_ids]

        unwatched_movies_df = self.movies_df[self.movies_df['id'].isin(unwatched_movie_ids)].copy()

        if unwatched_movies_df.empty:
            return pd.DataFrame()

        # Remove movies (from possible recommendation) which reviews weren't in training.
        unwatched_movies_df['movie_encoded'] = unwatched_movies_df['id'].apply(
            lambda x: self.movie_id_enc.transform([x])[0] if x in self.movie_id_enc.classes_ else -1
        )
        unwatched_movies_df = unwatched_movies_df[unwatched_movies_df['movie_encoded'] != -1]

        if unwatched_movies_df.empty:
            return pd.DataFrame()

        num_unwatched_movies = len(unwatched_movies_df)

        # Process movies in batches to handle listwise loss function constraint
        batch_size = self.list_size
        all_predictions = []

        for i in range(0, num_unwatched_movies, batch_size):
            batch_end = min(i + batch_size, num_unwatched_movies)
            batch_movies = unwatched_movies_df.iloc[i:batch_end].copy()
            current_batch_size = len(batch_movies)

            # Fill batch to list_size if necessary
            if current_batch_size < self.list_size:
                # Repeat last movie to fill the batch
                last_movie = batch_movies.iloc[-1:].copy()
                padding_needed = self.list_size - current_batch_size

                padding = pd.concat([last_movie] * padding_needed, ignore_index=True)
                batch_movies = pd.concat([batch_movies, padding], ignore_index=True)

            # Prepare batch inputs
            user_id_input_arr = np.full(self.list_size, user_encoded_id)
            movie_id_input_arr = batch_movies['movie_encoded'].values

            batch_movies['release date'] = pd.to_datetime(batch_movies['release date'], dayfirst=True).dt.year
            batch_movies['release_year_scaled'] = self.year_scal.transform(batch_movies[['release date']])
            year_input_arr = batch_movies['release_year_scaled'].values

            gender_input_arr = np.full(self.list_size, gender_encoded)
            occp_input_arr = np.full(self.list_size, occp_encoded)
            age_input_arr = np.full(self.list_size, age_scaled)

            genre_input_arrs = {}
            for genre in self.genres:
                genre_input_arrs[f'{genre}_input'] = np.array([batch_movies[genre].values])

            predict_inputs = {
                'user_id_input': np.array([user_id_input_arr]),
                'movie_id_input': np.array([movie_id_input_arr]),
                'gender_input': np.array([gender_input_arr]),
                'occp_input': np.array([occp_input_arr]),
                'age_input': np.array([age_input_arr]),
                'year_input': np.array([year_input_arr]),
                'rating_input': np.array([np.zeros(self.list_size)])
            }
            predict_inputs.update(genre_input_arrs)

            # Get predictions for this batch
            batch_predictions = self.model.predict(predict_inputs, verbose=0).flatten()

            # Exclude padding predictions
            all_predictions.extend(batch_predictions[:current_batch_size])

        predictions = np.array(all_predictions)

        recommendations = pd.DataFrame({
            'movie_id': unwatched_movies_df['id'],
            'title': unwatched_movies_df['title'],
            'predicted_rating': predictions
        })

        recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
        return recommendations.head(top_n)