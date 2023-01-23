import merlin.models.tf as mm
import nvtabular as nvt
import prepare_movielens
import tensorflow as tf
from merlin.schema.tags import Tags
from tensorflow.keras import regularizers

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def train_dlrm_model(train: nvt.Dataset, test: nvt.Dataset, epochs=1):
    model = mm.DLRMModel(
        schema=train.schema,
        embedding_dim=64,
        top_block=mm.MLPBlock([128, 64, 32]),
        bottom_block=mm.MLPBlock([128, 64]),
        prediction_tasks=mm.RegressionTask("rating"),
    )
    model.compile(optimizer="adam")
    model.fit(train, validation_data=test, batch_size=1024, epochs=epochs)
    return model

def train_dlrm_model_l2(train: nvt.Dataset, test: nvt.Dataset, epochs=1):
    model = mm.DLRMModel(
        schema=train.schema,
        embedding_dim=64,
        top_block=mm.MLPBlock([128, 64, 32], kernel_regularizer=regularizers.L2()),
        bottom_block=mm.MLPBlock([128, 64], kernel_regularizer=regularizers.L2()),
        prediction_tasks=mm.RegressionTask("rating"),
    )
    model.compile(optimizer="adam")
    model.fit(train, validation_data=test, batch_size=1024, epochs=epochs)
    return model

def load_and_preprocess_movielens():
    movies = nvt.Dataset(prepare_movielens.MOVIES_FILENAME)
    train = nvt.Dataset(prepare_movielens.TRAIN_FILENAME, part_size="5M")
    test = nvt.Dataset(prepare_movielens.TEST_FILENAME, part_size="5M")

    features = (
        ["userId", "movieId"]
        >> nvt.ops.JoinExternal(
            movies,
            on=["movieId"],
            on_ext=["movieId"],
            columns_ext=["movieId", "genres"],
        )
        >> nvt.ops.Categorify()
    )
    features += ["rating"] >> nvt.ops.AddMetadata(tags=[Tags.REGRESSION, Tags.TARGET])
    workflow = nvt.Workflow(features)

    train_transformed = workflow.fit_transform(train)
    test_transformed = workflow.fit_transform(test)

    return train_transformed, test_transformed


if __name__ == "__main__":
    train_transformed, test_transformed = load_and_preprocess_movielens()

    model = train_dlrm_model(train_transformed, test_transformed)
    metrics = model.evaluate(test_transformed, batch_size=1024, return_dict=True)
    print(metrics)
    preds = model.predict(test_transformed, batch_size=1024)
    print(f"min{min(preds)}, max={max(preds)}")
