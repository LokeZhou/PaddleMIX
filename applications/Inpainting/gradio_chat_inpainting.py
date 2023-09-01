# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import uuid

import gradio as gr
from PIL import Image

from paddlemix import Appflow


class ConversationBot:
    def __init__(self):

        self.pipe = Appflow(
            app="inpainting",
            models=[
                "THUDM/chatglm-6b",
                "GroundingDino/groundingdino-swint-ogc",
                "Sam/SamVitH-1024",
                "stabilityai/stable-diffusion-2-inpainting",
            ],
        )
        self.image_pil = None
        self.prompt = None

    def run_text(self, text, state):
        warning_prompt = None

        if self.image_pil is None:
            warning_prompt = "Please upload image"

        if warning_prompt is not None or text == "":
            state = state + [("Please upload image", "or Please input prompt")]

            return state, state

        self.prompt = text

        AI_prompt = "Received.  "
        state = state + [(text, AI_prompt)]
        result = self.run()

        if result["state"]:
            image_filename = f"{str(uuid.uuid4())[:8]}.png"
            result["result"].save(image_filename)
            state = state + [(f"![](/file={image_filename})", "Done!")]
        else:
            state = state + [("Please provide detailed prompts", "exp: xx is changed to xx")]

        return state, state

    def run_image(self, image, state):
        self.image_pil = Image.open(image.name)
        AI_prompt = "Upload....  "
        state = state + [(AI_prompt, f"![](/file={image.name})")]

        return state, state

    def run(self):
        image_pil = self.image_pil.convert("RGB")
        result = self.pipe(image=image_pil, prompt=self.prompt)

        return result

    def clear(self):
        self.image_pil = None
        self.prompt = None


bot = ConversationBot()
with gr.Blocks(css="#chatInpainting {overflow:auto; height:500px;}") as demo:
    gr.Markdown("<h3><center>Chat Inpainting</center></h3>")
    gr.Markdown(
        """This is a demo to the work [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX).<br>
            """
    )

    chatbot = gr.Chatbot(elem_id="chatbot", label="ChatBot")
    state = gr.State([])

    with gr.Row(visible=True) as input_raws:
        with gr.Column(scale=0.7):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                container=False
            )

        with gr.Column(scale=0.10, min_width=0):
            btn = gr.UploadButton("🖼️Upload", file_types=["image"])
        with gr.Column(scale=0.10, min_width=0):
            clear = gr.Button("🔄Clear️")

    btn.upload(bot.run_image, [btn, state], [chatbot, state])
    txt.submit(bot.run_text, [txt, state], [chatbot, state])
    txt.submit(lambda: "", None, txt)

    clear.click(bot.clear)
    clear.click(lambda: [], None, chatbot)
    clear.click(lambda: [], None, state)


demo.launch(share=True)
