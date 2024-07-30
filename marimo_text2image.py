import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from diffusers import StableDiffusionPipeline
    from dataclasses import dataclass
    import numpy as np
    from PIL import Image
    return Image, StableDiffusionPipeline, dataclass, mo, np


@app.cell
def __(StableDiffusionPipeline):
    pipeline = StableDiffusionPipeline.from_pretrained("models/t2i/").to("cuda")
    return pipeline,


@app.cell
def __(PIL, dataclass, mo, text):
    @dataclass
    class Prompt():
        text: str
        delete: bool = False

    @dataclass
    class imagegen():
        image: PIL.Image
        prompt: text

    get_prompts, set_prompts = mo.state([])
    get_images, set_images = mo.state([])
    prompt_added, set_prompt_added = mo.state(False)
    return (
        Prompt,
        get_images,
        get_prompts,
        imagegen,
        prompt_added,
        set_images,
        set_prompt_added,
        set_prompts,
    )


@app.cell
def __(mo):
    prompt_entry_box = mo.ui.text(placeholder="prompt")
    return prompt_entry_box,


@app.cell
def __(Prompt, mo, prompt_entry_box, set_prompts):
    def add_prompt():
        if prompt_entry_box.value:
            set_prompts(lambda v: v + [Prompt(prompt_entry_box.value.lower())] if satisfies(prompt_entry_box.value) else v)

    def satisfies(prompt):
        prompt = prompt.strip().lower()
        materials = prompt.split(" ")
        if not len(materials)==3:
            return False
        possible_mats = {"ar", "as", "gs", "gr", "pr", "ps"}
        for m in materials:
            if m not in possible_mats:
                return False
        return True

    def clear_prompts():
        set_prompts(lambda v: [prompt for prompt in v if not prompt.delete])

    add_prompts_button = mo.ui.button(label="add prompt", on_change=lambda _: add_prompt())
    delete_prompt_button = mo.ui.button(label="delete selected prompts", on_change=lambda _: clear_prompts())
    return (
        add_prompt,
        add_prompts_button,
        clear_prompts,
        delete_prompt_button,
        satisfies,
    )


@app.cell
def __(Prompt, get_prompts, mo, set_prompts):
    prompt_list = mo.ui.array([mo.ui.checkbox(value=p.delete, label=p.text) for p in get_prompts()], label="Prompts", 
                              on_change= lambda v: set_prompts(
                                  lambda prompts: [
                                      Prompt(prompt.text, v[i]) for i, prompt in enumerate(prompts)
                                  ]
                              ))
    return prompt_list,


@app.cell
def __(mo):
    possible_mats = {"ar", "as", "gs", "gr", "pr", "ps"}
    mo.md(f"Prompt mora sadržavati tri materijala: [ar, as, gs, gr, pr, ps]. <br/> npr. ar as gs")
    return possible_mats,


@app.cell
def __(
    add_prompts_button,
    delete_prompt_button,
    generate_images_button,
    mo,
    prompt_entry_box,
):
    mo.hstack([prompt_entry_box, add_prompts_button, delete_prompt_button, generate_images_button], justify="start")
    return


@app.cell
def __(mo, prompt_list):
    mo.as_html(prompt_list) if prompt_list.value else mo.md("Nema prompta 😔")
    return


@app.cell
def __(get_prompts, imagegen, mo, pipeline, set_images):
    generate_images_button = mo.ui.button(label="generate", on_change=lambda _: generate())

    def generate():
        images = []
        prompts = [p.text for p in get_prompts()]
        for i in mo.status.progress_bar(range(len(prompts)), title="Generating",
        show_eta=True,
        show_rate=True
        ):
            images.append(pipeline(prompts[i], height=256, width=256, num_inference_steps=100)[0][0])
        set_images([imagegen(im.convert("L"), prompt) for im, prompt in zip(images, prompts)])
    return generate, generate_images_button


@app.cell
def __(get_images, mo, np):
    grid = mo.vstack([[im.prompt, mo.image(np.array(im.image), alt=im.prompt)] for im in get_images()], align="center", justify="space-between")
    grid
    return grid,


if __name__ == "__main__":
    app.run()