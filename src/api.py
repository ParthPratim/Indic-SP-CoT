import time
import os
import openai
from typing import List
import asyncio


API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = None
API_BASE = os.getenv("OPENAI_API_BASE",None)
openai.api_base = "https://api.openai.com/v1"




# # if not API_KEY.startswith("sk"):
# API_BASE = os.environ["OPENAI_API_BASE"]
# openai.api_base = API_BASE




class APIModel(object):
    def __init__(self):
        self.api_key = None
        self.model_name = None

    def get_response(self, *args, **kwargs):
        raise NotImplementedError


class ChatGPT(APIModel):
    def __init__(self, model_name):
        super(ChatGPT, self).__init__()
        self.model = model_name

    @staticmethod
    def pack_message(prompt: str, role: str = "user", binary: bool = False):
        if binary:
            return [
                {"role": "system", "content": "You should give answers as direct and short as possible. Your answer must be either 'true' or 'false'. "},
                {"role": role, "content": prompt}
            ]
        else:
            return [
                {"role": "system", "content": "You should use your knowledge to answer the question to the best of your ability, not refuse to answer, even though you know your knowledge is sometimes out of date. If some references are uncertain, answer all possible cases rather than requesting further information."},
                {"role": role, "content": prompt}
            ]

    def get_response(
            self,
            prompt,
            max_tokens,
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            top_p=1,
            ban_pronoun=False,
            max_retry: int = 100,
            role: str = "user",
            binary: bool = False,
    ):
        response = None
        if binary:
            message = self.pack_message(prompt, role=role, binary=True)
        else:
            message = self.pack_message(prompt, role=role)
        
        if ban_pronoun:
            ban_token_ids = [1119, 1375, 679, 632, 770, 2312, 5334, 2332,
                             2399, 6363, 484, 673, 339, 340, 428, 777, 511,
                             607, 465, 663, 2990, 3347, 1544, 1026, 1212,
                             4711, 14574, 9360, 6653, 20459, 9930, 7091, 258,
                             270, 5661, 27218, 24571, 372, 14363, 896,
                             # 464,1169,383,262
                             ]
            logit_bias = {str(tid): -100 for tid in ban_token_ids}

            for _ in range(max_retry):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                        logit_bias=logit_bias,
                        
                    )
                    break
                except:
                    time.sleep(0.01)
        else:
            for _ in range(max_retry):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                    )

                    break
                except Exception as e:
                    print(e)
                    time.sleep(0.01)

        return response.choices[0].message.content if response else ""


class CompleteGPT(APIModel):
    def __init__(self, model_name):
        super(CompleteGPT, self).__init__()
        self.model = model_name

    def get_response(
            self,
            prompt,
            max_tokens,
            temperature=0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            top_p=1,
            ban_pronoun=False,
            max_retry: int = 50,
            binary: bool = False,
    ):
        response = None
        if ban_pronoun:
            ban_token_ids = [1119, 1375, 679, 632, 770, 2312, 5334, 2332,
                             2399, 6363, 484, 673, 339, 340, 428, 777, 511,
                             607, 465, 663, 2990, 3347, 1544, 1026, 1212,
                             4711, 14574, 9360, 6653, 20459, 9930, 7091, 258,
                             270, 5661, 27218, 24571, 372, 14363, 896,
                             # 464,1169,383,262
                             ]
            logit_bias = {str(tid): -100 for tid in ban_token_ids}

            for _ in range(max_retry):
                try:
                    response = openai.Completion.create(
                        engine=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                        logit_bias=logit_bias,
                    )
                    break
                except:
                    time.sleep(0.01)
        else:
            for _ in range(max_retry):
                try:
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                    )
                    break
                except:
                    time.sleep(0.01)

        return response["choices"][0]["text"] if response else ""


async def get_chat_response(
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stop: List[str] = None,
        top_p: float = 1,
):

    if model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo"]:
        message = [
            {
                "role": "user",
                "content": "You should use your knowledge to answer the question to the best of your ability, not refuse to answer, even though you know your knowledge is sometimes out of date. "
                           "If some references are uncertain, answer all possible cases rather than requesting further information. "
                           "No further search or extra information will be provided, you should answer it anyway based on your knowledge. "
                           "Answer ready if you understand the requirement."
             },
            {"role": "assistant", "content": "Ready."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                top_p=top_p
            )
            return response.choices[0].message.content
        except:
            return "RESPONSE_ERROR"

    elif model in ["text-davinci-003", "text-davinci-002"]:
        try:
            response = await openai.Completion.acreate(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
            return response["choices"][0]["text"] if response else ""
        except:
            return "RESPONSE_ERROR"
    else:
        return "Unsupported Model Name!"






