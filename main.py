import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import numpy as np
import gradio as gr
import torch

from nlp_tools import *
from image_tools import *


def img_detect(img, threshold=0.5):
    return object_detection(img, model, threshold=threshold, device=device)


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image()
                url_input = gr.Textbox(value='https://loremflickr.com/720/720')
                url_button = gr.Button("from URL") 

                with gr.Row():
                    image_button = gr.Button("Flip Image")
                    det_button = gr.Button("detect")

                sli = gr.Slider(0, 1.0, value=0.5, step=0.1, label='threshold')
                    
            image_output = gr.Image(shape=())

    with gr.Tab("NLP"):
        text_input = gr.Textbox()
        # 'Borat Sagdiev is a famous journalist from Kazakhstan.'
        with gr.Row():
            b1_upper = gr.Button("Uppercase")
            b1_lower = gr.Button("Lowercase")
            b1_flip = gr.Button("Flip")
            b1_en2es = gr.Button("En -> Es")
            b1_ner = gr.Button("NER")
        # text_output = gr.Textbox()
        text_output = gr.HighlightedText()

#     with gr.Accordion("Open for More!"):
        # gr.Markdown("Look at me...")

    b1_upper.click(lambda x: x.upper(), inputs=text_input, outputs=text_output)
    b1_lower.click(lambda x: x.lower(), inputs=text_input, outputs=text_output)
    b1_flip.click(lambda x: x[::-1], inputs=text_input, outputs=text_output)
    b1_en2es.click(en2es, inputs=text_input, outputs=text_output)
    b1_ner.click(ner, inputs=text_input, outputs=text_output)
    
    image_button.click(lambda x: np.fliplr(x), inputs=image_input, outputs=[image_input, image_output])
    url_button.click(img_from_url, inputs=url_input, outputs=image_input)
    det_button.click(img_detect, inputs=[image_input, sli], outputs=image_output)

    

pipe_en2es

demo.launch(share=True)
