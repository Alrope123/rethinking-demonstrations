import os
import argparse
import random
import json
import numpy as np

from collections import defaultdict, Counter

from templates import apply_template

def main(args):
    assert args.variant in [
        "gold", "random", # main experiments in Section 4
        "75_correct", "50_correct", "25_correct", "0_correct", # ablations in Section 4
        "gold_w_template", "random_w_template", # ablations in Section 4
        "ood_inputs", "random_english_words", "random_labels_only", "no_labels", # Section 5
        "random_english_words_gold_labels", "permutated_labels", "random_true_distribution"
    ]
    if args.variant in ["gold_w_template", "random_w_template"]:
        assert args.method is not None, "Please specify `--method` with the inference method (`direct` or `channel`) for using the template."
        assert args.method in ["direct", "channel"], "Please make sure to use either `direct` or `channel`."

    if args.variant=="gold":
        print ("No need to run `create_data.py` --- you can use the original data as it is.")
        return

    if args.variant=="ood_inputs":
        # load sources for OOD inputs
        assert args.corpus_path is not None, \
        """
        Please note that you need to specify the path to the corpus from which the OOD inputs will be sampled.
        It should be a .txt file where each line contains a sentence (plain text).
        """
        grouped_samples = defaultdict(list)
        with open(args.corpus_path, "r") as f:
            random_texts = []
            random_text_lens = []
            for line in f:
                line = line.strip()
                random_texts.append(line)
                random_text_lens.append(len(line.split()))
            random_text_lens = np.array(random_text_lens)

    elif args.variant in ["random_english_words", "random_english_words_gold_labels"]:
        from english_words import english_words_set
        english_words_set = sorted(english_words_set)

    datasets = args.dataset.split(',')
    new_datasets = [dataset + "_" + args.variant + (("_" + args.method) if args.method is not None else "") for dataset in datasets]
    assert len(datasets) == len(new_datasets)

    ################################################################################################################

    seeds = args.seed.split(',')
    perfs = []
    for dataset_idx, (dataset, new_dataset) in enumerate(zip(datasets, new_datasets)):

        # contruct and save a new config file and data directory
        config_file = os.path.join(args.config_dir, "tasks")
        assert os.path.exists(config_file), config_file
        with open(os.path.join(config_file, "{}.json".format(dataset)), "r") as f:
            config = json.load(f)

        # in case of random English words, we will create a config file and data directory
        # for each random seed later on (since the data is different across seeds)
        if args.variant not in ["random_english_words", "random_english_words_gold_labels"]:
            with open(os.path.join(config_file, "{}.json".format(new_dataset)), "w") as f:
                json.dump(config, f)

            new_dataset_dir = os.path.join(args.data_dir, new_dataset)
            if not os.path.exists(new_dataset_dir):
                os.mkdir(new_dataset_dir)
        
        # load full training data to get the distribution of the labels
        if args.variant=="random_true_distribution":
            full_train_data_path = os.path.join(args.data_dir, dataset, "{}_16384_100_train.jsonl".format(dataset))
            assert os.path.exists(full_train_data_path), "Please generate full training data first by running _build_gym.py with k=16384."
            full_train_data_labels = []
            with open(full_train_data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    assert dp["task"]==dataset
                    full_train_data_labels.append(dp["output"])
            train_label_counter = Counter(full_train_data_labels)
            train_label_distribution = {label : train_label_counter[label] / len(full_train_data_labels) for label in train_label_counter}

        for seed in seeds:
            # random seed
            np.random.seed(int(seed))

            if args.variant in ["random_english_words", "random_english_words_gold_labels"]:
                new_dataset = new_datasets[dataset_idx] + "_seed={}".format(seed)

            # read the original training and test data
            # note that we are modifying the training data only,
            # and the test data will always be the same
            # (we are creating duplicates only for convenience)
            train_data = []
            train_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "train"))
            with open(train_data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    assert dp["task"]==dataset
                    dp["task"] = new_dataset
                    train_data.append(dp)

            test_data = []
            test_data_path = os.path.join(args.data_dir, dataset, "{}_{}_{}_{}.jsonl".format(dataset, args.k, seed, "test"))
            with open(test_data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    assert dp["task"]==dataset
                    dp["task"] = new_dataset
                    test_data.append(dp)

            # apply templates to inputs and labels
            if args.variant in ["gold_w_template", "random_w_template"]:
                for dp in train_data:
                    apply_template(dp, dataset, args.method)
                for dp in test_data:
                    apply_template(dp, dataset, args.method)

            # now, for random_english_words, create a config file and data directory
            if args.variant in ["random_english_words", "random_english_words_gold_labels"]:
                new_dataset_dir = os.path.join(args.data_dir, new_dataset)
                if not os.path.exists(new_dataset_dir):
                    os.mkdir(new_dataset_dir)

                if config["task_type"]=="classification":
                    new_options = list(np.random.choice(english_words_set, size=len(config["options"]), replace=False))
                    new_mapping = {option: new_option for option, new_option in zip(config["options"], new_options)}
                    new_config = config.copy()
                    new_config["options"] = new_options

                    with open(os.path.join(config_file, "{}.json".format(new_dataset)), "w") as f:
                        json.dump(new_config, f)

                    for i, dp in enumerate(train_data):
                        train_data[i]["output"] = new_mapping[dp["output"]]
                        train_data[i]["options"] = [new_mapping[option] for option in dp["options"]]

                    if args.variant == "random_english_words_gold_labels":
                        # also modify the test data for classification tasks
                        for i, dp in enumerate(test_data):
                            test_data[i]["output"] = new_mapping[dp["output"]]
                            test_data[i]["options"] = [new_mapping[option] for option in dp["options"]]

                elif config["task_type"]=="multi-choice":
                    with open(os.path.join(config_file, "{}.json".format(new_dataset)), "w") as f:
                        json.dump(config, f)

                    shuffled_indices = np.random.permutation(range(len(english_words_set)))
                    shuffled_options = [english_words_set[i] for i in shuffled_indices]
                    offset = 0
                    for i, dp in enumerate(train_data):
                        new_options = shuffled_options[offset:offset+len(dp["options"])]
                        offset += len(dp["options"])
                        train_data[i]["output"] = new_options[dp["options"].index(dp["output"])]
                        train_data[i]["options"] = new_options

                else:
                    raise NotImplementedError()

            # modify both train input and test input for permutated_labels with classification tasks
            if args.variant == "permutated_labels" and config["task_type"]=="classification":
                old_options = config["options"]
                new_options = [old_options[(i+1)%len(old_options)] for i in range(len(old_options))]
                new_mapping = {old_option: new_option for old_option, new_option in zip(old_options, new_options)}

                for i, dp in enumerate(train_data):
                    train_data[i]["output"] = new_mapping[dp["output"]]                    
                for i, dp in enumerate(test_data):
                    test_data[i]["output"] = new_mapping[dp["output"]]
                    

            ## modify labels in the training data

            if args.variant in ["75_correct", "50_correct", "25_correct"]:
                num_correct = args.k * int(args.variant.split("_")[0]) // 100
                indices_correct = np.random.permutation(range(args.k))[:num_correct]

            for dp_idx, dp in enumerate(train_data):
                if args.variant in ["gold", "gold_w_template", "permutated_labels", "random_english_words_gold_labels"] or \
                        (args.variant in ["75_correct", "50_correct", "25_correct"] and dp_idx in indices_correct):
                    # assign correct label
                    pass
                elif args.variant.endswith("_correct"):
                    # assign incorrect label
                    dp["output"] = dp["options"][np.random.choice([i for i in range(len(dp["options"])) if dp["options"][i] != dp["output"]])]
                elif args.variant=="no_labels":
                    # assign empty label
                    dp["output"] = ""
                    dp["options"] = [""]
                elif args.variant=="random_true_distribution":
                    # assign random labels according to the distribution in the training data
                    dp["output"] = np.random.choice(list(train_label_distribution.keys()), p=list(train_label_distribution.values()))
                else:
                    # assign random label
                    dp["output"] = np.random.choice(dp["options"])

            ## modify inputs in the training data

            if args.variant=="random_labels_only":
                for dp in train_data:
                    dp["input"] = ""

            elif args.variant=="ood_inputs":
                new_train_data = []
                for dp in test_data:
                    l = len(dp["input"].split())
                    prob = np.exp(-np.power(random_text_lens-l, 2)/50)
                    prob /= np.sum(prob)
                    samples = np.random.choice(random_texts, size=args.k, replace=False, p=prob)
                    assert len(samples)==len(train_data)
                    new_train_data.append([])
                    for train_dp, sample in zip(train_data, samples):
                        new_train_data[-1].append({"task": train_dp["task"],
                                                    "input": sample,
                                                    "output": train_dp["output"],
                                                    "options": train_dp["options"]})
                train_data = new_train_data

            # write the modified data
            with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "train")), "w") as f:
                for dp in train_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

            with open(os.path.join(new_dataset_dir, "{}_{}_{}_{}.jsonl".format(new_dataset, args.k, seed, "test")), "w") as f:
                for dp in test_data:
                    f.write(json.dumps(dp))
                    f.write("\n")

            print ("Done for %s seed=%s" % (new_dataset, seed))


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")
    parser.add_argument("--variant", type=str, default="random", required=True)
    parser.add_argument("--method", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--config_dir", type=str, default="config")
    parser.add_argument("--corpus_path", type=str, default=None)

    args = parser.parse_args()

    main(args)
