# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?

This includes an original implementation of "[Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?][paper]" by [Sewon Min][sewon], [Xinxi Lyu][xinxi], [Ari Holtzman][ari], [Mikel Artetxe][mikel], [Mike Lewis][mike], [Hannaneh Hajishirzi][hanna], and [Luke Zettlemoyer][luke].

This code provides:
- Codes for creating the variants of the demonstrations used in the experiments.
- Commands to run the models and get numbers reported in the paper, based on the [MetaICL][metaicl] codebase.

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{ min2022rethinking,
    title={ Rethinking the Role of Demonstrations: What makes In-context Learning Work? },
    author={ Min, Sewon and Lyu, Xinxi and Holtzman, Ari and Artetxe, Mikel and Lewis, Mike and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={ 2022 }
}
```

### Announcements
* 07/25/2022: The code also supports running GPT-3 now.
* 02/25/2022: The code supports running GPT-2, MetaICL and GPT-J for now. Please contact authors for running other models.

## Content

1. [Preparation](#preparation)
2. [Reproducing Main Experiments](#reproducing-main-experiments) (Section 4.1 of the paper)
    * [No Demonstrations](#no-demonstrations)
    * [Demonstrations with gold labels](#demonstrations-with-gold-labels)
    * [Demonstrations with random labels](#demonstrations-with-random-labels)
3. [Reproducing Ablations](#reproducing-ablations) (Section 4.2 of the paper)
    * [Number of correct labels](#number-of-correct-labels)
    * [Number of input-label pairs in the demonstrations](#number-of-input-label-pairs-in-the-demonstrations)
    * [Using manual templates](#using-manual-templates)
4. [Reproducing Analysis](#reproducing-analysis) (Section 5 of the paper)
    * [Demonstrations with OOD input text](#demonstrations-with-ood-input-text)
    * [Demonstrations with random english words](#demonstrations-with-random-english-words)
    * [Demonstrations with random labels only (no inputs)](#demonstrations-with-random-labels-only-no-inputs)
    * [Demonstrations with no labels (inputs only)](#demonstrations-with-no-labels-inputs-only)

## Preparation

The code is tested with python 3.8.

The data and the code are based on the MetaICL codebase.
```bash
git remote add metaicl https://github.com/facebookresearch/MetaICL.git
git pull metaicl main
```

Install the data dependencies and download the data.
```bash
conda conda create --name metaicl-data python=3.8
conda activate metaicl-data
pip install datasets==1.4.0 wget
cd preprocess
python _build_gym.py --build --n_proc=40 --do_test
```

This uses `k=16` by default. If you want to run ablations with varying `k`, please also run the following.
```bash
python _build_gym.py --build --n_proc=40 --do_test --test_k {4|8|32}
```

After preprocesisng is done, come back to the main directory.
```bash
cd ../
conda deactivate
```

Now, install the model dependencies to run the model. Please note that the Transformer version is not compatible to the datasets library used to download the data, so make sure to use a different environment.
```
conda conda create --name metaicl python=3.8
conda activate metaicl
pip install torch==1.9.0
pip install git+https://github.com/huggingface/transformers.git@c37573806ab3526dd805c49cbe2489ad4d68a9d7
```

(Optional) Install OpenAI Python Library for running GPT-3
```
pip install openai
```

## Reproducing Main Experiments

This is for reproducing experiments in Section 4.1 of the paper.
Evaluation datasets are:
* Classification (16 datasets): `financial_phrasebank`,`poem_sentiment`,`glue-wnli`,`climate_fever`,`glue-rte`,`superglue-cb`,`sick`,`medical_questions_pairs`,`glue-mrpc`,`hate_speech18`,`ethos-national_origin`,`ethos-race`,`ethos-religion`,`tweet_eval-hate`,`tweet_eval-stance_atheism`,`tweet_eval-stance_feminist`
* Multi-choice (10 datasets): `quarel`,`openbookqa`,`qasc`,`commonsense_qa`,`ai2_arc`,`codah`,`superglue-copa`,`dream`,`quartz-with_knowledge`,`quartz-no_knowledge`

#### No Demonstrations

To run the evaluation of No-Demonstrations:

```bash
# Direct GPT-2 Large
python test.py --dataset {dataset} --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot
# Channel GPT-2 Large
python test.py --dataset {dataset} --gpt2 gpt2-large --method channel --out_dir out/gpt2-large --do_zeroshot
# Direct MetaICL
python test.py --dataset {dataset} --gpt2 metaicl --method direct --out_dir out/metaicl --do_zeroshot
# Channel MetaICL
python test.py --dataset {dataset} --gpt2 channel-metaicl --method channel --out_dir out/channel-metaicl --do_zeroshot
# Direct GPT-J
python test.py --dataset {dataset} --gpt2 gpt-j-6B --method direct --out_dir out/gpt-j --do_zeroshot
# Channel GPT-J
python test.py --dataset {dataset} --gpt2 gpt-j-6B --method channel --out_dir out/gpt-j --do_zeroshot
# GPT-3
python test_gpt3.py --dataset {dataset} --gpt3 {ada|babbage|curie|davinci} --method {direct|channel} --out_dir out/gpt3 --do_zeroshot --api {API key}
```
Note that `test.py` and `test_gpt3.py` does not support multi-gpu for inference.

Other useful flags:
* `--test_batch_size`: can be adjusted based on your GPU memory availability. With a 32GB GPU, you can use 64 for GPT-2 Large & MetaICL, and 16 for GPT-J **with no demonstrations**. Later, when you run the code **with demonstrations**, decreasing the batch size by 4 times typically works, e.g., 16 (GPT-2 Large & MetaICL) and 4 (GPT-J) with a 32GB GPU.
* `--log_file`: if you want to save logs in a file, you can specify the path to the log file.

Notes for running GPT-3:
* You can create/check your OpenAI API keys by visiting [this link](https://beta.openai.com/account/api-keys).
* Running with GPT-3 can be expensive, and different models of GPT-3 comes with different costs. Please check [this link](https://openai.com/api/pricing/) to evaluate the cost before running each experiment.
* The responses from the GPT-3 API are cached in the `out_dir`.

From now on, we will use the above commands as a default and tell you which flags you need to add.


#### Demonstrations with gold labels

Run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87`.

#### Demonstrations with random labels

Create the demonstrations with random labels via:
```bash
python create_data.py --variant random --dataset {dataset}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_random`.

## Reproducing Ablations

This is for reproducing experiments in Section 4.2 of the paper.
Evaluation datasets are:
* Classification (5 datasets): `poem_sentiment`,`glue-rte`,`sick`,`glue-mrpc`,`tweet_eval-hate`
* Multi-choice (4 datasets): `openbookqa`,`commonsense_qa`,`ai2_arc`,`superglue-copa`

#### Number of correct labels

Create the demonstrations with varying number of correct labels via:
```bash
python create_data.py --variant {75|50|25|0}_correct --dataset {dataset}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_{75|50|25|0}_correct`.

#### Number of input-label pairs in the demonstrations

(Note that you should have run preprocessing with varying `k` to run this ablation. If you have not done this, please re-visit the [Preparation](#preparation) section.)

Create the demonstrations with varying `k` via:
```bash
python create_data.py --variant random --dataset {dataset} --k {4|8|16|32}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k {4|8|16|32} --seed 100,13,21,42,87 --dataset {dataset}_random`.

#### Using manual templates

Create the demonstrations with varying type of labels and inference method via:
```bash
python create_data.py --variant {gold|random}_w_template --dataset {dataset} --method {direct|channel}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_{gold|random}_w_template_{direct|channel}`.

## Reproducing Analysis

This is for reproducing experiments in Section 5 of the paper.
Evaluation datasets are:
* Classification (5 datasets): `poem_sentiment`,`glue-rte`,`sick`,`glue-mrpc`,`tweet_eval-hate`
* Multi-choice (4 datasets): `openbookqa`,`commonsense_qa`,`ai2_arc`,`superglue-copa`

#### Demonstrations with OOD input text

First, you need a corpus file in a .txt format, where each line is a sentence (in the plain text).
In the paper, we used samples from the English portion of CC News, which we are unable to release here.
Please visit [this link](https://commoncrawl.org/2016/10/news-dataset-available/) to learn more about how to download the CC News corpus.

Create the demonstrations with OOD input text via:
```bash
python create_data.py --variant ood_inputs --dataset {dataset} --corpus_path {corpus_path}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_ood_inputs`.

#### Demonstrations with random english words

Create the demonstrations with random English words as labels via:
```bash
python create_data.py --variant random_english_words --dataset {dataset}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed {seed} --dataset {dataset}_random_english_words_seed={seed}`, where `seed` can be one of 100, 13, 21, 42, and 87.

#### Demonstrations with random labels only (no inputs)

Create the demonstrations with random labels only via:
```bash
python create_data.py --variant random_labels_only --dataset {dataset}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_random_labels_only`.

#### Demonstrations with no labels (inputs only)

Create the demonstrations with no labels via:
```bash
python create_data.py --variant no_labels --dataset {dataset}
```
Then, run the commands same as [default commands](#no-demonstrations) but add `--use_demonstrations --k 16 --seed 100,13,21,42,87 --dataset {dataset}_no_labels`.


[paper]: https://arxiv.org/abs/2202.12837
[sewon]: http://shmsw25.github.io/
[xinxi]: https://alrope123.github.io/
[ari]: https://ari-holtzman.github.io/
[mikel]: https://scholar.google.com/citations?user=N5InzP8AAAAJ&hl=en
[mike]: https://ai.facebook.com/people/mike-lewis/
[hanna]: https://homes.cs.washington.edu/~hannaneh/index.html
[luke]: https://www.cs.washington.edu/people/faculty/lsz

[metaicl]: https://github.com/facebookresearch/MetaICL
