import pandas as pd
import prepare_movielens
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def train_xgboost_model(train: pd.DataFrame):
    print(train)
    X, y = train[["userId", "movieId", "genres"]], train["rating"]
    column_transformer = ColumnTransformer(
        [
            ("genre_tr", CountVectorizer(analyzer=set), "genres"),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    model = Pipeline(
        [
            ("encode_genres", column_transformer),
            ("xgb", xgb.XGBRegressor()),
        ]
    )
    model.fit(X, y)
    print(model.get_params())
    return model


def load_and_preprocess_movielens_df():
    movies = pd.read_csv(prepare_movielens.MOVIES_FILENAME)
    train = pd.read_csv(prepare_movielens.TRAIN_FILENAME)
    test = pd.read_csv(prepare_movielens.TEST_FILENAME)

    def transform(df: pd.DataFrame):
        joined = df.join(movies.set_index("movieId"), "movieId", how="inner")
        joined["genres"] = joined["genres"].str.split("|")
        without_title = joined.drop("title", axis=1)
        return without_title

    train_transformed = transform(train)
    test_transformed = transform(test)

    return train_transformed, test_transformed


if __name__ == "__main__":
    train_transformed, test_transformed = load_and_preprocess_movielens_df()

    model = train_xgboost_model(train_transformed)
    X, y = test_transformed[["userId", "movieId", "genres"]], test_transformed["rating"]
    preds = model.predict(X)
    print(f"RMSE: {mean_squared_error(y, preds, squared=False)}")
    print(f"min{min(preds)}, max={max(preds)}")
