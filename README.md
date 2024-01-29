## SyncDiffusion

Cog wrapper for SyncDiffusion, leveraging the [Stable Diffusion 2.0](https://huggingface.co/stabilityai/stable-diffusion-2-base/tree/main), introduces an innovative approach to generating seamless panoramas. 
Refer to the original [project page](https://syncdiffusion.github.io/), [paper](https://syncdiffusion.github.io/static/syncdiffusion_pdf.pdf) and [repository](https://github.com/KAIST-Geometric-AI-Group/SyncDiffusion) for technical details.


## How to use the API

You need to have Cog and Docker installed to run this model locally. To use Sync Diffusion you need to provide text prompts. You can create horizontal or vertical panoramic immages. The output file will be in .png format. 

To build the docker image with cog and run a prediction:
```bash
cog predict -i prompt="natural landscape in anime style illustration"
```

To start a server and send requests to your locally or remotely deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

Input parameters are as follows: 

- prompt: Provide a descriptive prompt for the image you want to generate. This is the primary driver of the content in the generated panorama.  
- negative_prompt: Use this to specify what you don't want in the image, helping to refine the results.  
- width: Set the width of the output image.  
- height: Set the height of the output image.  
- guidance_scale: Adjusts the scale of the guidance image. Higher values lead to more adherence to the prompt.  
- sync_weight: Determines the weight of the sync diffusion in the image generation process.  
- sync_decay_rate: Sets the weight schduler decay rate of the sync diffusion.   
- sync_freq: Specifies the frequency for the gradient descent of the sync diffusion process.  
- sync_threshold: Defines the maximum number of steps for the sync diffusion.  
- num_inference_steps: Sets the number of inference steps for the diffusion process.  
- stride: Determines the window stride in the latent space for the diffusion.  
- seed: Provides a seed for the sync diffusion to control randomness.  
- loop_closure: Enable or disable the use of loop closure in the panorama image generation.  


## References
```
@article{lee2023syncdiffusion,
    title={SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions}, 
    author={Yuseung Lee and Kunho Kim and Hyunjin Kim and Minhyuk Sung},
    journal={arXiv preprint arXiv:2306.05178},
    year={2023}
}
```