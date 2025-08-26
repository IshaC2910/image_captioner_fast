import gradio as gr
from PIL import Image
from inference import Captioner

captioner = Captioner()

def describe(img: Image.Image) -> str:
    if img is None:
        return "Please upload an image."
    return captioner.caption(img)

with gr.Blocks(title="Fast Image Captioner (BLIP)") as demo:
    gr.Markdown("# 🖼️ → ✍️ Fast Image Captioner\nUsing **Salesforce/blip-image-captioning-base** (faster & compact).")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload an image")
            run = gr.Button("Generate caption")
            opts = gr.Slider(5, 40, value=30, step=1, label="Max tokens (length)")
        with gr.Column(scale=1):
            out = gr.Textbox(label="Caption", lines=2)
    def generate_with_len(img, length):
        if img is None:
            return "Please upload an image."
        captioner.gen_kwargs["max_length"] = int(length)
        return captioner.caption(img)

    run.click(generate_with_len, inputs=[inp, opts], outputs=out)
    inp.change(generate_with_len, inputs=[inp, opts], outputs=out)

if __name__ == "__main__":
    demo.launch()
