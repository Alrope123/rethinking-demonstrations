import time
import sys
import numpy as np
import torch
import json
import openai
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import GPT2Tokenizer

class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger


    def prepare_data(self, train_data, test_data, method, batch_size=10, dp_sep="\n", max_length=1024):
        # format demonstrations
        demonstrations = ""
        for dp in train_data:
            if method=="direct":
                demonstrations += "{}{}{}\n\n\n".format(dp["input"], dp_sep, dp["output"])
            elif method=="channel":
                demonstrations += "{}{}{}\n\n\n".format(dp["output"], dp_sep, dp["input"])
            else:
                raise NotImplementedError()

        # append demonstrations and separate options
        inputs = []
        outputs = []
        metadata = []
        for dp in test_data:
            prompt = dp["input"]
            options = dp["options"]

            indices = [i for i in range(len(inputs), len(inputs) + len(options))]
            metadata.append({"indices": indices, "options": options})

            if method=="direct":
                inputs += [demonstrations + prompt + dp_sep for option in options]
                outputs += [option for option in options]
            elif method=="channel":
                inputs += [demonstrations + option + dp_sep for option in options]
                outputs += [prompt for option in options]
            else:
                raise NotImplementedError()

        # truncate inputs
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            input_ids = self.tokenizer.encode(inp)
            output_ids = self.tokenizer.encode(out)
            if (len(input_ids) + len(output_ids) > max_length):
                input_ids = input_ids[len(input_ids)+len(output_ids) - max_length:]
                assert len(input_ids)+len(output_ids) == max_length
            inputs[i] = self.tokenizer.decode(input_ids)

        if self.logger is not None:
            self.logger.info("Checking the first example...")
            self.logger.info(inputs[0] + "" + outputs[0])

        # construct a dataloader
        dataset = zip(inputs, outputs)
        input_chunks = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        output_chunks = [outputs[i : i + batch_size] for i in range(0, len(outputs), batch_size)]
        dataloader = [(input_chunks[i], output_chunks[i]) for i in range(0, len(input_chunks))]

        return dataloader, metadata


    def do_inference(self, dataloader):
        losses = []
        cache = []
        cost = 0
        for inputs, outputs in dataloader:
            data = [inp + out for inp, out in zip(inputs, outputs)]
            response = self.gpt3(data)
            for choice in response["choices"]:
                cost += len(choice["logprobs"]["tokens"]) * 0.00006
            print("current cost = " + str(cost))
            cache.append((data, response))
            # get the beginning of the target from the response (based on tokenization)
            for inp, outp, out in zip(inputs, outputs, response["choices"]):
                assert inp+outp==out["text"]
                i = 0
                while out['logprobs']['text_offset'][i] < len(inp):
                    i += 1
                loss = -sum(out['logprobs']["token_logprobs"][i:])
                losses.append(loss / (len(out['logprobs']['text_offset']) - i))
        return losses, cache


    def do_predict(self, losses, metadata):
        predictions = []
        for dp in metadata:
            curr_label_losses = [losses[index] for index in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response
