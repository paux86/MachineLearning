# Matt Brierley
# Final Project
# Comp 478, Spring 2021

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(42)


class Project:
    def load_data(self, min_reviews, split=True, target_userId=None):
        # load csv files
        movies = pd.read_csv('ml-latest-small/movies.csv')
        ratings = pd.read_csv('ml-latest-small/ratings.csv')
        tags = pd.read_csv('ml-latest-small/tags.csv')

        # for collaborative filtering, theres probably a more efficient built-in way I could have done this, its quite slow
        if target_userId:
            # get list of movies each user has watched
            # for each other user (one not passed), compare lists, and sum 0.5-(user1rating/5 - user2rating/5) for each movie in both lists
            # create new ratings list consisting of x most similar users, or add similar users until x number of total ratings
            user_movies = ratings.groupby('userId')['movieId'].apply(list).to_frame().reset_index()
            target_user = user_movies[user_movies.userId == target_userId]
            user_movies = user_movies[user_movies.userId != target_userId]
            similarity_scores = {}

            '''
            # print target top 10 for reference
            target_top_movies = ratings[ratings['userId'] == target_userId].nlargest(10, 'movieId')
            print(f"Top rated movies for user {target_userId}:\n")
            for index, row in target_top_movies.iterrows():
                movie_id = (int)(row['movieId'])
                print(f"{movies[movies['movieId'] == movie_id]['title'].item()} - Rating: {row['rating']}")
            '''

            for id in user_movies.userId.unique():
                similarity_scores[id] = 0
            
            for movie in target_user['movieId'][0]:
                target_rating = ratings.query('userId == @target_userId and movieId == @movie')['rating'].item()
                for index, row in user_movies.iterrows():
                    user_id = row['userId']
                    if movie in row['movieId']:
                        user_rating = ratings.query('userId == @user_id and movieId == @movie')['rating'].item()
                        similarity_scores[user_id] += (0.5 - (abs(target_rating - user_rating) / 5))
            
            similar_users = sorted(similarity_scores, key=similarity_scores.get, reverse=True)[:100]
            similar_users.append(target_user)        
            ratings = ratings[ratings['userId'].isin(similar_users)]

        # make list of tags per title
        tags = tags.drop(columns=['userId', 'timestamp'])
        tags = tags.drop_duplicates()
        tags = tags.groupby('movieId')['tag'].apply(list).to_frame()

        # average rating per movieId
        ratings = ratings.groupby('movieId').filter(lambda x: len(x) >= min_reviews)
        ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        
        # build full lens df
        movie_ratings = pd.merge(movies, ratings)
        lens = pd.merge(movie_ratings, tags, on='movieId', how='outer')
        lens = lens.dropna(subset=['title'])
        lens['tag'] = lens['tag'].fillna('Unknown')
        #lens = lens.drop(columns=['userId', 'timestamp']) # if not averaging rating
        
        # split and one hot encode genres
        lens['genres'] = lens['genres'].apply(lambda x: x.split('|'))
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(sparse_output=True)
        lens = lens.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(lens.pop('genres')),
                index=lens.index,
                columns=mlb.classes_).add_prefix('genre_'))
        if 'genre_(no genres listed)' in lens.columns:
            lens = lens.drop(columns=['genre_(no genres listed)'])
        
        # one hot encode tags, not sure if i could do both at the same time
        lens = lens.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(lens.pop('tag')),
                index=lens.index,
                columns=mlb.classes_).add_prefix('tag_'))
        
        if split == False:
            return lens

        titles = lens['title']
        lens = lens.drop(columns=['movieId', 'title'])

        y = lens['rating']
        X = lens.drop(columns=['rating'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        return titles, X_train, X_test, y_train, y_test

    def train_decision_tree(self):
        # Decision Tree
        print("DecisionTreeRegressor")
        
        best_dtr_config = []
        for min_reviews in range(0,16):
            print(f"\nMin Reviews: {min_reviews}\n")

            titles, X_train, X_test, y_train, y_test = self.load_data(min_reviews)
            X_train_holdout, X_train_validate, y_train_holdout, y_train_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            criterion = ['mse'] # when tuning for mse, mse criterion always comes out ahead, so removed others to save a ton of time
            max_depth = [None, 2, 4, 6, 8, 10]
            max_leaf_nodes = [None, 5, 10, 20, 100]
            min_samples_leaf = [1, 10, 20, 40]
            min_samples_split = [2, 10, 20, 40]

            for c in criterion:
                for max_d in max_depth:
                    for max_ln in max_leaf_nodes:
                        for min_sl in min_samples_leaf:
                            for min_sp in min_samples_split:
                                dtr = DecisionTreeRegressor(random_state=42, criterion=c, max_depth=max_d, max_leaf_nodes=max_ln, min_samples_leaf=min_sl, min_samples_split=min_sp)
                                dtr.fit(X_train_holdout, y_train_holdout)
                                y_pred = dtr.predict(X_train_validate)

                                mse = mean_squared_error(y_train_validate, y_pred) # should be near 0
                                r2 = r2_score(y_train_validate, y_pred) # should be near 1

                                if not best_dtr_config or mse < best_dtr_config[0]:
                                    best_dtr_config = [mse, r2, c, max_d, max_ln, min_sl, min_sp, min_reviews]
                                    print(f"New best MSE: {best_dtr_config}, Combined: {mse + (1-r2)}")
        print(best_dtr_config) # 'mse', 6, None, 10, 40, 15

        titles, X_train, X_test, y_train, y_test = self.load_data(best_dtr_config[7])
        dtr = DecisionTreeRegressor(random_state=42, criterion=best_dtr_config[2], max_depth=best_dtr_config[3], max_leaf_nodes=best_dtr_config[4], min_samples_leaf=best_dtr_config[5], min_samples_split=best_dtr_config[6])
        dtr.fit(X_train, y_train)
        y_pred = dtr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"Best MSE on test data with DecisionTreeRegressor: {mse}")
        print(f"Best R2 on test data with DecisionTreeRegressor: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('DecisionTreeRegressor')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()
        
        return dtr
         
    def get_trained_decision_tree(self):
        best_dtr_config = ['mse', 6, None, 10, 40, 12] # 15 is probably overfit, better results on 12 (2nd best in training)

        titles, X_train, X_test, y_train, y_test = self.load_data(best_dtr_config[5])

        dtr = DecisionTreeRegressor(random_state=42, criterion=best_dtr_config[0], max_depth=best_dtr_config[1], max_leaf_nodes=best_dtr_config[2], min_samples_leaf=best_dtr_config[3], min_samples_split=best_dtr_config[4])
        dtr.fit(X_train, y_train)
        
        return dtr, titles, X_train, X_test, y_train, y_test

    def train_knn(self):
        # KNN
        print("KNeighborsRegressor")
        
        best_knnr_config = []
        for min_reviews in range(0,16):
            print(f"\nMin Reviews: {min_reviews}\n")

            titles, X_train, X_test, y_train, y_test = self.load_data(min_reviews)
            X_train_holdout, X_train_validate, y_train_holdout, y_train_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            n_neighbors = list(range(1,31))
            leaf_size = [1] # list(range(1,50)) # doesn't appear to have any impact on this dataset
            p=[1,2]

            for n in n_neighbors:
                for l in leaf_size:
                    for param in p:
                        knnr = KNeighborsRegressor(n_neighbors=n, leaf_size=l, p=param)
                        knnr.fit(X_train_holdout, y_train_holdout)
                        y_pred = knnr.predict(X_train_validate)

                        mse = mean_squared_error(y_train_validate, y_pred) # should be near 0
                        r2 = r2_score(y_train_validate, y_pred) # should be near 1

                        if not best_knnr_config or mse < best_knnr_config[0]:
                            best_knnr_config = [mse, r2, n, l, param, min_reviews]
                            print(f"New best MSE: {best_knnr_config}, Combined: {mse + (1-r2)}")
        print(best_knnr_config) # [17, 1, 1, 15]
        
        titles, X_train, X_test, y_train, y_test = self.load_data(best_knnr_config[5])

        knnr = KNeighborsRegressor(n_neighbors=best_knnr_config[2], leaf_size=best_knnr_config[3], p=best_knnr_config[4])
        knnr.fit(X_train, y_train)
        y_pred = knnr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"\nBest MSE on test data with KNeighborsRegressor: {mse}")
        print(f"Best R2 on test data with KNeighborsRegressor: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('KNeighborsRegressor')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()
    
    def get_trained_knn(self):
        best_knnr_config = [17, 1, 1, 11] # 15 seems overfit, much better results with 11 (2nd best in training)

        titles, X_train, X_test, y_train, y_test = self.load_data(best_knnr_config[3])

        knnr = KNeighborsRegressor(n_neighbors=best_knnr_config[0], leaf_size=best_knnr_config[1], p=best_knnr_config[2])
        knnr.fit(X_train, y_train)
        
        return knnr, titles, X_train, X_test, y_train, y_test

    def train_linear_regression(self):
        # Linear Regression
        print("LinearRegression")
        
        best_lr_config = []
        for min_reviews in range(0,16):
            print(f"\nMin Reviews: {min_reviews}")

            titles, X_train, X_test, y_train, y_test = self.load_data(min_reviews)
            X_train_holdout, X_train_validate, y_train_holdout, y_train_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            normalize = [False, True]

            for n in normalize:
                lr = LinearRegression(normalize=n)
                lr.fit(X_train_holdout, y_train_holdout)
                y_pred = lr.predict(X_train_validate)

                mse = mean_squared_error(y_train_validate, y_pred) # should be near 0
                r2 = r2_score(y_train_validate, y_pred) # should be near 1

                if not best_lr_config or mse < best_lr_config[0]:
                    best_lr_config = [mse, r2, n, min_reviews]
                    print(f"New best MSE: {best_lr_config}, Combined: {mse + (1-r2)}")
        print(best_lr_config) # [False, 14]
        
        titles, X_train, X_test, y_train, y_test = self.load_data(best_lr_config[3])

        lr = LinearRegression(normalize=best_lr_config[2])
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"Best MSE on test data with LinearRegression: {mse}")
        print(f"Best R2 on test data with LinearRegression: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('LinearRegression')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

    def get_trained_linear_regression(self):
        best_lr_config = [False, 6] # 14 was best from training, but is overfit. 6 second best, and better results

        titles, X_train, X_test, y_train, y_test = self.load_data(best_lr_config[1])

        lr = LinearRegression(normalize=best_lr_config[0])
        lr.fit(X_train, y_train)

        return lr, titles, X_train, X_test, y_train, y_test

    def train_nn(self):
        # NN
        print("Neural Network\n")

        # Train Learning Rate
        titles, X_train, X_test, y_train, y_test = self.load_data(15)
        X_train_holdout, X_train_validate, y_train_holdout, y_train_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        '''
        # coarse search
        max_count = 100
        for count in range(max_count):
            model = keras.Sequential()
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))

            lr = 10**np.random.uniform(-3, -6)
            model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=lr), metrics=["mse"])
            model.fit(X_train_holdout, y_train_holdout, batch_size=32, epochs=5, validation_data=(X_train_validate, y_train_validate), verbose=0)
            scores = model.evaluate(X_train_validate, y_train_validate, verbose=0)
            print(f"lr: {lr}, loss: {scores[0]}, val_mse: {scores[1]}")

        # fine search
        max_count = 100
        for count in range(max_count):
            model = keras.Sequential()
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(32, activation='relu'))
            model.add(layers.Dense(1))

            lr = 10**np.random.uniform(-3, -4)
            model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=lr), metrics=["mse"])
            model.fit(X_train_holdout, y_train_holdout, batch_size=32, epochs=10, validation_data=(X_train_validate, y_train_validate), verbose=0)
            scores = model.evaluate(X_train_validate, y_train_validate, verbose=0)
            print(f"lr: {lr}, loss: {scores[0]}, val_mse: {scores[1]}")
        '''
        lr = 9e-4 # from observation
        
        # Train layers
        # tested 2-4 layers and various node amounts, and found the best results with 3 layers of 32 nodes
        activation_functions = ['relu', 'elu', 'tanh']
        dropout_layer = [round(x * 0.1, 1) for x in range(0,10)] # 0.0 for no dropout
        best_nn_model_config = []

        for a1 in activation_functions:
            for a2 in activation_functions:
                for a3 in activation_functions:
                    for d in dropout_layer:
                        model = keras.Sequential()
                        model.add(layers.Dense(32, activation=a1))
                        model.add(layers.Dense(32, activation=a2))
                        model.add(layers.Dense(32, activation=a3))
                        model.add(layers.Dropout(d))
                        model.add(layers.Dense(1))

                        model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=lr), metrics=["mse"])
                        model.fit(X_train_holdout, y_train_holdout, batch_size=32, epochs=10, validation_data=(X_train_validate, y_train_validate), verbose=0)
                        
                        y_pred = model.predict(X_train_validate)
                        mse = mean_squared_error(y_train_validate, y_pred)

                        if not best_nn_model_config or mse < best_nn_model_config[0]:
                            best_nn_model_config = [mse, a1, a2, a3, d]
                            print(f"New best model found: {best_nn_model_config}")
        print(f"\nBest trained model:\nLearning rate: {lr}\nLayer1 activation: {best_nn_model_config[1]}\nLayer2 activation: {best_nn_model_config[2]}\nLayer3 activation: {best_nn_model_config[3]}\nDropout: {best_nn_model_config[4] if best_nn_model_config[4] else 'None'}")

        # Test vs Test data
        model = keras.Sequential()
        model.add(layers.Dense(32, activation=best_nn_model_config[1]))
        model.add(layers.Dense(32, activation=best_nn_model_config[2]))
        model.add(layers.Dense(32, activation=best_nn_model_config[3]))
        model.add(layers.Dropout(best_nn_model_config[4]))
        model.add(layers.Dense(1))

        model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=lr), metrics=["mse"])
        model.fit(X_train_holdout, y_train_holdout, batch_size=32, epochs=10, validation_data=(X_train_validate, y_train_validate), verbose=0)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"Best MSE on test data with Neural Network: {mse}")
        print(f"Best R2 on test data with Neural Network: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('Neural Network')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

    def get_trained_nn(self, fit=True):
        best_nn_model_config = ['elu', 'tanh', 'tanh', 0.0, 9e-4, 12] # 15 was overfit, 12 was 2nd best in training

        # Test vs Test data
        model = keras.Sequential()
        model.add(layers.Dense(32, activation=best_nn_model_config[0]))
        model.add(layers.Dense(32, activation=best_nn_model_config[1]))
        model.add(layers.Dense(32, activation=best_nn_model_config[2]))
        model.add(layers.Dropout(best_nn_model_config[3]))
        model.add(layers.Dense(1))

        model.compile(loss="mse", optimizer=keras.optimizers.RMSprop(learning_rate=best_nn_model_config[4]), metrics=["mse"])
        
        if not fit:
            return model
        
        titles, X_train, X_test, y_train, y_test = self.load_data(best_nn_model_config[5])
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

        return model, titles, X_train, X_test, y_train, y_test

    def display_trained_models(self):
        # Decision Tree
        model, titles, X_train, X_test, y_train, y_test = proj.get_trained_decision_tree()

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"\nBest MSE on test data with DecisionTreeRegressor: {mse}")
        print(f"Best R2 on test data with DecisionTreeRegressor: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('Decision Tree')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

        # KNN
        model, titles, X_train, X_test, y_train, y_test = proj.get_trained_knn()

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"\nBest MSE on test data with KNeighborsRegressor: {mse}")
        print(f"Best R2 on test data with KNeighborsRegressor: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('KNN')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

        # LR
        model, titles, X_train, X_test, y_train, y_test = proj.get_trained_linear_regression()

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"\nBest MSE on test data with LinearRegression: {mse}")
        print(f"Best R2 on test data with LinearRegression: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('Linear Regression')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()

        # NN
        model, titles, X_train, X_test, y_train, y_test = proj.get_trained_nn()

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred) # should be near 0
        r2 = r2_score(y_test, y_pred) # should be near 1

        print(f"\nBest MSE on test data with Neural Network: {mse}")
        print(f"Best R2 on test data with Neural Network: {r2}")
        print(f"Adjusted combination: {mse + (1-r2)}\n")

        plt.scatter(y_test, y_pred)
        plt.xlim(1.5,5.5)
        plt.ylim(1.5,5.5)
        plt.title('Neural Network')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.show()
    
    def collaborative_recommendations(self, userId, recommendations=10, min_reviews=10):
        training_data = proj.load_data(min_reviews=min_reviews, target_userId=userId, split=False)
        training_data_y = training_data['rating']
        training_data_titles = training_data['title']
        training_data = training_data.drop(columns=['rating', 'movieId', 'title'])

        test_data = proj.load_data(min_reviews=min_reviews, split=False)
        test_data_y = test_data['rating']
        test_data_titles = test_data['title']
        test_data = test_data.drop(columns=['rating', 'movieId', 'title'])
        test_data = test_data[training_data.columns]
        
        model = self.get_trained_nn(fit=False)
        model.fit(training_data, training_data_y, batch_size=32, epochs=10, verbose=0)
        y_pred = model.predict(test_data)
        
        ind = np.argpartition(y_pred, -recommendations, axis=None)[-recommendations:]
        for i in ind:
            print(f"{test_data_titles[i]} - Predicted rating: {y_pred[i].item()}")


proj = Project()

print("Rating Prediction Training")
print("==========================")

# Show training
#proj.train_decision_tree()
#proj.train_knn()
#proj.train_linear_regression()
#proj.train_nn()

# Show stats on trained models
proj.display_trained_models()

# Show a few predictions on trained NN model
print("Predictions from trained NN model")
print("=================================")
model, titles, X_train, X_test, y_train, y_test = proj.get_trained_nn()

for i in range(10):
    movie_data = X_test[i:i+1]
    movie_index = movie_data.index[0]
    movie_title = titles[movie_index]
    predicted = model.predict(movie_data)

    print(f"\nMovie: {movie_title}")
    print(f"Rating: {y_test[movie_index]}, Predicted: {predicted[0,0]}")

# Recommendation System
print("\nMovie recommendations - KNearest")
print("================================")

data = proj.load_data(min_reviews=10, split=False)
titles = data['title']
data = data.drop(columns=['movieId', 'title', 'rating'])

from sklearn.neighbors import NearestNeighbors
recommender = NearestNeighbors()
recommender.fit(data)

for i in range(10):
    movie_data = data[i:i+1]
    movie_index = movie_data.index[0]
    movie_title = titles[movie_index]

    print(f"\nMovie: {movie_title}")

    recommendations = recommender.kneighbors(X=movie_data, n_neighbors=10, return_distance=False)

    print("\nRecommendations:")
    for i in recommendations[0]:
        if i != movie_index:
            print(titles[i])

print("\nMovie recommendations - Collaborative Filtering")
print("===============================================\n")

proj.collaborative_recommendations(userId=1, recommendations=10)