# Wrapper entrypoint for Hugging Face Spaces (Gradio)
from app_gradio import demo
if __name__ == "__main__":
    demo.launch()
