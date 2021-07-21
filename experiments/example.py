"""
example.py
=======================
Example of how the training loop should operate.
"""
from sklearn.ensemble import RandomForestClassifier
from src.compute_statistics import marginal_statistics_from_dataframe
from src import generate


def load_dataframes():
    """This should return the dataframes as processed and saved by get_data."""
    pass


def evaluate(train_data, test_data):
    """Custom evaluation."""
    pass


def save_results():
    """Save the results to some text/json file."""
    pass


def run(dataset_name="admissionprediction", regularisation="emm"):
    """Example generation, model build, and evaluation loop."""
    # Load the data to build statistics from, real training data, real test data.
    frame_stat, frame_train, frame_test = load_dataframes()

    # Compute the marginal statistics from frame_stat
    target_statistics = marginal_statistics_from_dataframe(frame_stat)

    # Now generate a weighted corpus from the training set using these statistics
    reweighted_corpus = generate.generate_weighted_dataset_from_marginals(target_statistics)

    # Do a couple of other models
    independent_corpus = generate.generate_dataset_using_independent_normals(target_statistics)
    other_corpus = generate.generate_dataset_using_some_better_method(target_statistics)

    # Get ready to train a model and save results
    runs = {
        'full_training': frame_train,
        'emm': reweighted_corpus,
        'independent': independent_corpus,
        'other': other_corpus
    }

    # For saving the results
    results = dict.fromkeys(runs.keys())

    # Run all the models
    for name, dataset in runs.items():
        data, labels = dataset.drop('labels', axis=1)
        labels = dataset['labels']

        # Train
        model = RandomForestClassifier()
        model.fit(data, labels)

        # Evaulate and dump results to results
        results[name] = evaluate(train, test)

    # Print rseults and save
    print(results)
    save_results(results)