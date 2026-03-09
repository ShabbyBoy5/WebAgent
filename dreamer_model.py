"""
Drop-in replacement for WebWorldModel that uses Dreamer-7B locally
via HuggingFace transformers with 4-bit quantization.

Usage:
    from dreamer_model import DreamerWorldModel, encode_image
    model = DreamerWorldModel()
    result = model.multiple_step_change_prediction(...)
"""

import re
import base64
import torch
from PIL import Image
from io import BytesIO
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_model():
    print("Loading Dreamer-7B with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "osunlp/Dreamer-7B",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = Qwen2VLProcessor.from_pretrained("osunlp/Dreamer-7B")
    print("Dreamer-7B loaded.")
    return model, processor


class DreamerWorldModel:
    def __init__(self):
        self.model, self.processor = _load_model()

    def _call_model(self, messages: list, max_new_tokens: int = 2048) -> str:
        hf_messages = []
        images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                hf_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                hf_content = []
                for item in content:
                    if item["type"] == "text":
                        hf_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                        else:
                            raise ValueError(f"Unsupported image URL format: {url[:50]}")
                        images.append(img)
                        hf_content.append({"type": "image"})
                hf_messages.append({"role": role, "content": hf_content})

        text = self.processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def state_change_prediction_in_website(self, screenshot, task, action_description, format="change"):
        prompt_text = " Please predict the changes after {}.".format(action_description)
        if format == "change":
            system = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the changes that will occur on the webpage after the operation is performed, such as the appearance of new elements, the disappearance of existing elements, or changes in the content of existing elements. The operation type and the element to operate will be provided in the prompt. Directly output 'State changes: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "html":
            system = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should generate the html code for the new webpage, highlight the most likely elements appearing in the new page. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "accessibility":
            system = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should describe the new webpage as an accessibility tree, highlight the most likely elements appearing in the new page. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": screenshot}},
            ]},
        ]
        return self._call_model(messages)

    def action_proposal_in_imagination(self, screenshot, task, imaginations, format="change"):
        if format == "change":
            prompt_text = "The above image is a screenshot of a web page. You are required to complete the following task: {}\n ".format(task)
        else:
            prompt_text = "You are required to complete the following task: {}\n ".format(task)
        prompt_text += "PREVIOUS ACTIONS: \n"
        for i, item in enumerate(imaginations):
            prompt_text += str(i + 1) + " : " + item[0] + "\n"
        prompt_text += "\n"

        if format == "change":
            prompt_text += "The above image is the screenshot before actually performing all the previous actions. The current webpage should be a combination of the initial screenshot with the following changes."
            prompt_text += "The webpage has gone through several changes caused by previous actions:\n\n"
            for i, item in enumerate(imaginations):
                prompt_text += "ACTION {}: \n{}\n{}\n\n".format(str(i + 1), item[0], item[1])
            prompt_text += "Based on the initial screenshot and the changes to the webpage, please predict a single next step action to complete the given task. Please don't repeat any action from PREVIOUS ACTIONS. Please directly specify the operation in a short natural language description, including the operation type and the element to operate. Don't output anything else."
            messages = [
                {"role": "system", "content": "You are an autonomous intelligent agent tasked with navigating a web browser."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": screenshot}},
                ]},
            ]
        else:
            last_step = imaginations[-1]
            prompt_text += "The current state of the webpage is as follows: \n{}\n".format(last_step[1])
            prompt_text += "Based on the current state of the webpage, please predict a single next step action to complete the given task. Please don't repeat any action from PREVIOUS ACTIONS. Please directly specify the operation in a short natural language description. Don't output anything else."
            messages = [
                {"role": "system", "content": "You are an autonomous intelligent agent tasked with navigating a web browser."},
                {"role": "user", "content": prompt_text},
            ]

        return self._call_model(messages)

    def state_change_prediction_in_imagination(self, screenshot, task, imaginations, action, format="change"):
        prompt_text = " Please predict the changes after action: {}".format(action)
        for i, item in enumerate(imaginations):
            prompt_text += "ACTION {}: \n{}\n{}\n\n".format(str(i + 1), item[0], item[1])
        prompt_text += "Based on the initial screenshot and the changes to the webpage, please predict the changes after action: {}".format(action)

        if format == "change":
            system = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage, a sequence of actions and state changes, and an operation to perform. Predict the new changes. Directly output 'State changes: ...' and don't output anything else."
        elif format == "html":
            system = "You are an agent that predicts the effect of an action on a webpage. Generate the html code for the new webpage. Directly output 'New webpage: ...' and don't output anything else."
        elif format == "accessibility":
            system = "You are an agent that predicts the effect of an action on a webpage. Describe the new webpage as an accessibility tree. Directly output 'New webpage: ...' and don't output anything else."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": screenshot}},
            ]},
        ]
        return self._call_model(messages)

    def multiple_step_change_prediction(self, screenshot, screenshot_path, task,
                                        action_description, format="change", k=0):
        rtn_str = ''
        imagination_list = []
        rtn_str += "Proposed New Action: \n" + action_description + '\n'

        change_description = self.state_change_prediction_in_website(screenshot, task, action_description, format)
        rtn_str += "Predicted new webpage: \n" + change_description + '\n'
        imagination_list.append([action_description, change_description])

        for i in range(k):
            rtn_str += "==" * 25 + "STEP: " + str(i) + "==" * 25 + '\n'
            proposed_action = self.action_proposal_in_imagination(screenshot, task, imagination_list, format)
            rtn_str += "Proposed New Action: \n" + proposed_action + '\n'
            imagined_state = self.state_change_prediction_in_imagination(screenshot, task, imagination_list, proposed_action, format)
            rtn_str += "Predicted new webpage: \n" + imagined_state + '\n'
            if "stop" in proposed_action.lower():
                break
            imagination_list.append([proposed_action, imagined_state])

        return rtn_str


if __name__ == "__main__":
    model = DreamerWorldModel()

    screenshot_path = "demo_data/shopping_0.png"
    screenshot = "data:image/jpeg;base64," + encode_image(screenshot_path)
    action_description = "type 'red blanket' in the search bar and click search"
    task = "Buy the least expensive red blanket (in any size) from 'Blankets & Throws' category."

    result = model.multiple_step_change_prediction(
        screenshot, screenshot_path, task, action_description,
        format="accessibility", k=1
    )
    print(result)
