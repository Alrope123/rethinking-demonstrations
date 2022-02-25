import string

TEMPLATES = {
    "financial_phrasebank": {
        "direct" : ("{}", "The sentiment is: {}"),
        "channel": ("{}", "The sentiment is: {}")
    },
    "poem_sentiment": {
        "direct" : ("{}", "The sentiment is: {}"),
        "channel": ("{}", "The sentiment is: {}")
    },
    "glue-mrpc": {
        "direct" : ("{}\nThe question is: {} True or False?", "The answer is: {}"),
        "channel": ("The question is: {} True or False?\n{}", "The answer is: {}")
    },
    "glue-rte": {
        "direct" : ("{}\nThe question is: {} True or False?", "The answer is: {}"),
        "channel": ("The question is: {} True or False?\n{}", "The answer is: {}")
    },
    "sick": {
        "direct" : ("{}\nThe question is: {} True or False?", "The answer is: {}"),
        "channel": ("The question is: {} True or False?\n{}", "The answer is: {}")
    },
    "tweet_eval-hate": {
        "direct" : ("Tweet: {}", "Sentiment: {}"),
        "channel": ("Tweet: {}", "Sentiment: {}"),
    },
    "openbookqa": {
        "direct" : ("The question is: {}", "The answer is: {}"),
        "channel": ("The question is: {}", "The answer is: {}")
    },
    "ai2_arc": {
        "direct" : ("The question is: {}", "The answer is: {}"),
        "channel": ("The question is: {}", "The answer is: {}")
    },
    "codah": {
        "direct" : ("The question is: {}", "The answer is: {}"),
        "channel": ("The question is: {}", "The answer is: {}")
    },
    "commonsense_qa": {
        "direct" : ("The question is: {}", "The answer is: {}"),
        "channel": ("The question is: {}", "The answer is: {}")
    }
}

def apply_template(dp, dataset, method):
    if dataset.startswith("superglue-copa"):
        if method == "direct":
            if dp["input"].startswith("Cause: "):
                dp["input"] = dp["input"][7:-1] + " so"
                dp["output"] = dp["output"][8].lower() + dp["output"][9:]
                for i, options in enumerate(dp["options"]):
                    dp["options"][i] = dp["options"][i][8].lower() + dp["options"][i][9:]
            elif dp["input"].startswith("Effect: "):
                dp["input"] = dp["input"][8:-1] + " because"
                dp["output"] = dp["output"][7].lower() + dp["output"][8:]
                for i, options in enumerate(dp["options"]):
                    dp["options"][i] = dp["options"][i][7].lower() + dp["options"][i][8:]
            else:
                raise NotImplementedError()
        elif method == "channel":
            if dp["output"].startswith("Cause: "):
                dp["output"] = dp["output"][7:-1] + " so"
                dp["input"] = dp["input"][8].lower() + dp["input"][9:]
                for i, options in enumerate(dp["options"]):
                    dp["options"][i] = dp["options"][i][7:-1] + " so"
            elif dp["output"].startswith("Effect: "):
                dp["output"] = dp["output"][8:-1] + " because"
                dp["input"] = dp["input"][7].lower() + dp["input"][8:]
                for i, options in enumerate(dp["options"]):
                    dp["options"][i] =  dp["options"][i][8:-1] + " because"
        else:
            raise NotImplementedError(o)
    elif dataset.startswith("glue") or dataset.startswith("sick"):
        def map_option(option):
            if option in ["equivalent", "entailment"]:
                return "True"
            if option in ["not_equivalent", "not_entailment", "contradiction"]:
                return "False"
            if option in ["neutral"]:
                return "Not sure"
            raise NotImplementedError(option)
        dp["input"] = dp["input"].replace("sentence 1: ", "").replace("sentence 2: ", "")
        splits = dp["input"].split(" [SEP] ")
        if method=="channel":
            splits = [splits[1], splits[0]]
        splits = [split if split[-1] in string.punctuation else split+"." for split in splits]
        dp["input"] = TEMPLATES[dataset][method][0].format(splits[0], splits[1])
        dp["output"] = TEMPLATES[dataset][method][1].format(map_option(dp["output"]))
        for i, options in enumerate(dp["options"]):
            dp["options"][i] =TEMPLATES[dataset][method][1].format(map_option(dp["options"][i]))
    else:
        def map_option(option):
            if dataset=="tweet_eval-hate":
                return {"hate": "against", "non-hate": "favor"}[option]
            return option
        dp["input"] = TEMPLATES[dataset][method][0].format(dp["input"])
        dp["output"] = TEMPLATES[dataset][method][1].format(map_option(dp["output"]))
        for i, options in enumerate(dp["options"]):
            dp["options"][i] =TEMPLATES[dataset][method][1].format(map_option(dp["options"][i]))





