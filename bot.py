import discord
from discord.ext import commands
from gradio_client import Client
from PIL import Image
from io import BytesIO
import asyncio
import os
import tempfile

bot = commands.Bot(command_prefix=">", intents=discord.Intents.all())

midjourney_client = Client("mukaist/Midjourney")
dalle_client = Client("mukaist/DALLE-4K")
stable_diffusion_client = Client("stabilityai/stable-diffusion-3-medium")
pixart_client = Client("PixArt-alpha/PixArt-Sigma")
flux_schnell_client = Client("black-forest-labs/FLUX.1-schnell")
flux_dev_client = Client("black-forest-labs/FLUX.1-dev")

async def generate_midjourney_image(prompt: str):
    try:
        result = await asyncio.to_thread(
            midjourney_client.predict,
            prompt=prompt,
            negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
            use_negative_prompt=True,
            style="2560 x 1440",
            seed=0,
            width=1080,
            height=1080,
            guidance_scale=6,
            randomize_seed=True,
            api_name="/run"
        )
        return [item['image'] for item in result[0]], "Midjourney"
    except Exception as e:
        print(f"Error generating Midjourney images: {e}")
        return None, "Midjourney"

async def generate_dalle_image(prompt: str):
    try:
        result = await asyncio.to_thread(
            dalle_client.predict,
            prompt=prompt,
            negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
            use_negative_prompt=True,
            style="3840 x 2160",
            seed=1,
            width=1080,
            height=1080,
            guidance_scale=7,
            randomize_seed=True,
            api_name="/run"
        )
        return [item['image'] for item in result[0]], "DALL-E"
    except Exception as e:
        print(f"Error generating DALL-E images: {e}")
        return None, "DALL-E"

async def generate_stable_diffusion_image(prompt: str):
    try:
        images = []
        for _ in range(1):  # Generate 6 images
            result = await asyncio.to_thread(
                stable_diffusion_client.predict,
                prompt=prompt,
                negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                seed=0,
                randomize_seed=True,
                width=1024,
                height=1024,
                guidance_scale=5,
                num_inference_steps=40,
                api_name="/infer"
            )
            image_path = result[0] if isinstance(result, tuple) else result
            images.append(image_path)
        return images, "Stable Diffusion"
    except Exception as e:
        print(f"Error generating Stable Diffusion images: {e}")
        return None, "Stable Diffusion"

async def generate_pixart_image(prompt: str):
    try:
        result = await asyncio.to_thread(
            pixart_client.predict,
            prompt=prompt,
            negative_prompt="low quality, bad, watermark, ugly",
            style="(No style)",
            use_negative_prompt=False,
            num_imgs=1,
            seed=0,
            width=1080,
            height=1080,
            schedule="DPM-Solver",
            dpms_guidance_scale=4.5,
            sas_guidance_scale=3,
            dpms_inference_steps=30,
            sas_inference_steps=30,
            randomize_seed=True,
            api_name="/run"
        )
        image_info = result[0][0]
        return [image_info['image']], "PixArt-alpha"
    except Exception as e:
        print(f"Error generating PixArt-alpha images: {e}")
        return None, "PixArt-alpha"

async def generate_flux_image(prompt: str, client, model_name):
    try:
        result = await asyncio.to_thread(
            client.predict,
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=30,
            api_name="/infer"
        )
        image_path = result[0] if isinstance(result, tuple) else result
        return [image_path], f"FLUX.1-{model_name}"
    except Exception as e:
        print(f"Error generating FLUX.1-{model_name} image: {e}")
        return None, f"FLUX.1-{model_name}"

async def generate_flux_schnell_image(prompt: str):
    return await generate_flux_image(prompt, flux_schnell_client, "schnell")

async def generate_flux_dev_image(prompt: str):
    return await generate_flux_image(prompt, flux_dev_client, "dev")

@bot.event
async def on_ready():
    print(f"Logged in as: {bot.user.name}!")

@bot.command(name="gen")
async def askai(ctx: commands.Context, *, prompt: str):
    initial_message = await ctx.reply("Generating images from Midjourney, DALL-E, Stable Diffusion, PixArt-alpha, FLUX.1-schnell, and FLUX.1-dev, please wait...")

    tasks = [
        generate_midjourney_image(prompt),
        generate_dalle_image(prompt),
        generate_stable_diffusion_image(prompt),
        generate_pixart_image(prompt),
        generate_flux_schnell_image(prompt),
        generate_flux_dev_image(prompt)
    ]

    all_files = []
    ai_results = []
    temp_files = []

    async def process_result(image_paths, ai_type):
        nonlocal all_files, ai_results, temp_files
        if image_paths:
            for i, image_path in enumerate(image_paths):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                image = Image.open(BytesIO(image_data))

                # Save the image to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                image.save(temp_file.name, format="PNG")
                temp_files.append(temp_file.name)

                file = discord.File(temp_file.name, filename=f'{ai_type}_image_{i+1}.png')
                all_files.append(file)

                if os.path.exists(image_path):
                    os.remove(image_path)

            ai_results.append(ai_type)

            # Update the message with all accumulated images
            current_content = f"Images generated by: {', '.join(ai_results)}"
            new_files = [discord.File(temp_file, filename=f'image_{i+1}.png') for i, temp_file in enumerate(temp_files)]
            await initial_message.edit(content=current_content, attachments=new_files)

    try:
        for completed_task in asyncio.as_completed(tasks):
            image_paths, ai_type = await completed_task
            await process_result(image_paths, ai_type)

        if not all_files:
            await initial_message.edit(content="Sorry, I couldn't generate any images. Please try again.")
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")
