import os
import sys
import time
import random
import string
import gc
import json
import base64
import io

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.parameter import ParameterType

import torch
# Set allocator configuration before any CUDA work to prevent fragmentation
if 'PYTORCH_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage

"""
ROS node for the **Qwen-Image-2.1** text-to-image and image-editing model from the Qwen team.
"""

class Q21ImgNode(Node):
    """
    A ROS node that interfaces with the Qwen-Image-2.1 model for text-to-image
    and image-editing generation.
    """

    def __init__(self):
        """
        Initializes the Q21ImgNode, declaring parameters, creating subscribers/publishers,
        and setting up the image processing bridge.
        """
        super().__init__('tti', namespace=os.environ.get('ROS_NAMESPACE', ''))
        self.get_logger().info("Initializing Qwen-Image-2.1 ROS node...")

        self.bridge = CvBridge()

        # --- ROS Parameter Declaration ---
        self.declare_parameter(
            'repo_id',
            os.environ.get('Q21IMG_REPO_ID', 'Qwen/Qwen-Image-2.1'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The Huggingface repo id to use (only used when model_dir has no local weights).'
            )
        )
        self.declare_parameter(
            'model_dir',
            os.environ.get('Q21IMG_MODEL_DIR', os.path.join(os.getcwd(), 'models')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Directory holding the local weights (model_index.json) or the download cache.'
            )
        )
        self.declare_parameter(
            'device',
            os.environ.get('Q21IMG_DEVICE', 'cuda:0'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The torch device to use.'
            )
        )
        self.declare_parameter(
            'prompt',
            os.environ.get('Q21IMG_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='An initial prompt to generate an image at startup.'
            )
        )
        self.declare_parameter(
            'input_image',
            os.environ.get('Q21IMG_INPUT_IMAGE', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to an input image to condition the generation.'
            )
        )
        self.declare_parameter(
            'once',
            os.environ.get('Q21IMG_ONCE', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, the node will exit after generating the first image.'
            )
        )
        self.declare_parameter(
            'image_path',
            os.environ.get('Q21IMG_IMAGE_PATH', os.path.join('generated_images', 'auto')),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path for saving the output image. If empty, local saving is disabled.'
            )
        )
        self.declare_parameter(
            'seed',
            int(os.environ.get('Q21IMG_SEED', -1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Seed for the random number generator.'
            )
        )
        self.declare_parameter(
            'image_counter_start',
            int(os.environ.get('Q21IMG_IMAGE_COUNTER_START', 9000)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='The starting value for the image filename counter.'
            )
        )
        self.declare_parameter(
            'height',
            int(os.environ.get('Q21IMG_HEIGHT', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Height of the generated image (native: 2048).'
            )
        )
        self.declare_parameter(
            'width',
            int(os.environ.get('Q21IMG_WIDTH', 1024)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Width of the generated image (native: 2048).'
            )
        )
        self.declare_parameter(
            'output_resolution',
            int(os.environ.get('Q21IMG_OUTPUT_RESOLUTION', 0)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Edge length (px) the condition image is resized to before encoding. '
                            '0 = use max(width, height). Only affects image-conditioned runs.'
            )
        )
        self.declare_parameter(
            'num_inference_steps',
            int(os.environ.get('Q21IMG_NUM_STEPS', 40)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of denoising steps.'
            )
        )
        self.declare_parameter(
            'guidance_scale',
            float(os.environ.get('Q21IMG_GUIDANCE', 1.0)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Classifier-free guidance (maps to true_cfg_scale). Only active with > 1 and a negative_prompt.'
            )
        )
        self.declare_parameter(
            'negative_prompt',
            os.environ.get('Q21IMG_NEGATIVE_PROMPT', ''),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Negative prompt, required for guidance_scale > 1.'
            )
        )
        self.declare_parameter(
            'num_images_per_prompt',
            int(os.environ.get('Q21IMG_NUM_IMAGES', 1)),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of images to generate per prompt.'
            )
        )
        self.declare_parameter(
            'frame_id',
            os.environ.get('Q21IMG_FRAME_ID', 'q21img'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The frame_id for the generated image messages.'
            )
        )
        self.declare_parameter(
            'keep_loaded',
            os.environ.get('Q21IMG_KEEP_LOADED', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, keeps the model in memory between prompts.'
            )
        )
        self.declare_parameter(
            'preload',
            os.environ.get('Q21IMG_PRELOAD', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, loads the model at startup instead of on the first prompt.'
            )
        )
        self.declare_parameter(
            'cpu_offload',
            os.environ.get('Q21IMG_CPU_OFFLOAD', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If true, uses model CPU offloading to save VRAM.'
            )
        )
        self.declare_parameter(
            'sequential_offload',
            os.environ.get('Q21IMG_SEQUENTIAL_OFFLOAD', 'false').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='More aggressive offloading (slower) for very tight VRAM.'
            )
        )
        self.declare_parameter(
            'vae_tiling',
            os.environ.get('Q21IMG_VAE_TILING', 'true').lower() in ['1','true'],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Tiled VAE decode to keep the memory peak flat (needed from ~1536 px).'
            )
        )

        # --- Get Parameters ---
        self.repo_id = self.get_parameter('repo_id').value
        self.model_dir = self.get_parameter('model_dir').value
        self.device = self.get_parameter('device').value
        self.image_path = self.get_parameter('image_path').value
        self.prompt = self.get_parameter('prompt').value
        self.seed = self.get_parameter('seed').value
        self.input_image = self.get_parameter('input_image').value
        self.once = self.get_parameter('once').value
        self.image_counter = self.get_parameter('image_counter_start').value
        self.height = self.get_parameter('height').value
        self.width = self.get_parameter('width').value
        self.torch_dtype = torch.bfloat16
        self.pipe = None
        self.current_source = None
        self.current_cpu_offload = None
        self.current_sequential_offload = None
        self.current_vae_tiling = None

        # Ensure model_dir exists
        if self.model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            self.get_logger().info(f"Created model directory: {self.model_dir}")

        # Ensure image_path directory exists if using auto
        if 'auto' in self.image_path:
            out_dir = os.path.dirname(self.image_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        self.get_logger().info(f"Using repo_id: '{self.repo_id}'")
        self.get_logger().info(f"Using device: '{self.device}'")

        # Make the effective weight source visible at startup: the fallback to the
        # Huggingface hub downloads tens of GB, so it must not happen silently.
        if not self.model_dir or not os.path.isfile(os.path.join(self.model_dir, 'model_index.json')):
            self.get_logger().warning(
                f"No local weights (model_index.json) in model_dir '{self.model_dir}' – "
                f"falling back to Huggingface repo '{self.repo_id}', which downloads into that directory. "
                f"Set model_dir (or Q21IMG_MODEL_DIR) to the local weights directory, e.g. ./models."
            )
        else:
            self.get_logger().info(f"Using local weights from: '{self.model_dir}'")

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            String,
            'prompt',
            self.prompt_callback,
            1000)

        # --- Publisher ---
        self.image_pub = self.create_publisher(Image, 'generated_image', 10)

        # --- Handle initial prompt ---
        if self.prompt:
            self.get_logger().info("Processing initial prompt...")
            # Use Timer to avoid blocking constructor/initialization
            # (Humble has no `oneshot` argument, so the timer destroys itself in its callback)
            self._prompt_timer = self.create_timer(0.1, self._initial_prompt_timer_callback)

        # --- Preload the model at startup ---
        if self.get_parameter('preload').value:
            self.get_logger().info("Preloading model at startup...")
            # Use Timer to avoid blocking constructor/initialization
            self._preload_timer = self.create_timer(0.1, self._preload_timer_callback)

        # --- Dynamic Parameter Callback ---
        self._param_callback_handle = self.add_on_set_parameters_callback(self._parameters_callback)

    def _parameters_callback(self, params):
        """
        Callback to dynamically update parameters at runtime.
        """
        for param in params:
            if param.name == 'image_counter_start':
                try:
                    self.image_counter = int(param.value)
                    self.get_logger().info(f"Image counter reset to: {self.image_counter}")
                except (ValueError, TypeError) as e:
                    self.get_logger().error(f"Invalid value for image_counter_start: {param.value} ({e})")
                    return SetParametersResult(successful=False, reason=str(e))
            elif param.name == 'image_path':
                self.image_path = str(param.value)
                if 'auto' in self.image_path:
                    out_dir = os.path.dirname(self.image_path)
                    if out_dir and not os.path.exists(out_dir):
                        os.makedirs(out_dir, exist_ok=True)
                self.get_logger().info(f"Image path updated to: {self.image_path}")
        return SetParametersResult(successful=True)

    def _preload_timer_callback(self):
        """
        One-shot timer callback that loads the pipeline before the first prompt arrives.
        The load-relevant parameters are read here, so a changed RQT setting is picked up.
        """
        self.destroy_timer(self._preload_timer)
        try:
            self._load_pipeline(
                self.get_parameter('cpu_offload').value,
                self.get_parameter('sequential_offload').value,
                self.get_parameter('vae_tiling').value,
            )
            self.get_logger().info("Model preloaded and ready.")
        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("Preload failed: CUDA out of memory. Loading will be retried on the first prompt.")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.get_logger().error(f"Preload failed: {e}")

    def _initial_prompt_timer_callback(self):
        """
        One-shot timer callback to process the initial prompt after the node has initialized.
        """
        self.destroy_timer(self._prompt_timer)
        self.prompt_callback(String(data=self.prompt))

    def _parse_prompt(self, msg_data):
        """
        Parses the incoming prompt message.
        Supports plain text or a JSON string like:
        {"content": "...", "image_url": "..."}
        `image_url` may also be a list of up to ten reference images.
        """
        try:
            data = json.loads(msg_data)
            if isinstance(data, dict):
                if 'content' not in data:
                    self.get_logger().error("JSON prompt missing 'content' key.")
                    return None, None
                return data.get('content'), data.get('image_url')
        except json.JSONDecodeError:
            pass # Not a JSON string, treat as plain text

        return msg_data, None

    def _load_input_image(self, image_input):
        """
        Loads a list of condition images from URLs, local file paths, or base64 data strings.
        """
        if isinstance(image_input, str):
            image_inputs = [image_input]
        elif isinstance(image_input, (list, tuple)):
            image_inputs = list(image_input)
        else:
            self.get_logger().error(f"Unsupported image_url type: {type(image_input).__name__}")
            return None

        images = []
        for entry in image_inputs:
            try:
                if entry.startswith('data:image'):
                    # Handle base64
                    self.get_logger().info("Loading image from base64 data...")
                    header, encoded = entry.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    images.append(PILImage.open(io.BytesIO(image_data)).convert("RGB"))
                    continue

                # Strip file:// prefix if present
                clean_path = entry
                if entry.startswith('file://'):
                    clean_path = entry[7:]

                if not clean_path.startswith(('http://', 'https://')):
                    # Local file handling
                    if not os.path.exists(clean_path):
                        self.get_logger().error(f"Local image file does not exist: {clean_path}")
                        return None
                    self.get_logger().info(f"Loading local image from: {clean_path}")
                    images.append(PILImage.open(clean_path).convert("RGB"))
                else:
                    # Handle URL via diffusers utility
                    self.get_logger().info(f"Loading image from URL: {clean_path}")
                    images.append(load_image(clean_path).convert("RGB"))
            except Exception as e:
                self.get_logger().error(f"Failed to load input image '{entry}': {e}")
                return None

        return images or None

    def _get_image_output_path(self):
        """
        Generates the output path for the generated image.
        If image_path ends with 'auto', generates a unique filename with a counter and random suffix.
        """
        if self.image_path.endswith('auto'):
            directory = os.path.dirname(self.image_path)
            random_chars = string.ascii_letters + string.digits
            random_string = ''.join(random.choice(random_chars) for _ in range(10))
            filename = f"q21img_{self.image_counter:04d}_{random_string}.png"
            return os.path.join(directory, filename)
        else:
            return self.image_path

    def _resolve_model_source(self):
        """
        Returns the source to load from and the cache directory to use.
        Local weights win: if model_dir contains a model_index.json, it is loaded directly
        (no Huggingface cache involved), otherwise repo_id is downloaded into model_dir.
        """
        if self.model_dir and os.path.isfile(os.path.join(self.model_dir, 'model_index.json')):
            return self.model_dir, None
        return self.repo_id, (self.model_dir or None)

    def _load_pipeline(self, cpu_offload, sequential_offload, vae_tiling):
        """
        Loads the pipeline if it is not loaded yet or if a load-relevant parameter changed.
        """
        source, cache_dir = self._resolve_model_source()

        # Reload if not loaded or if configuration changed
        if (self.pipe is not None and
            self.current_source == source and
            self.current_cpu_offload == cpu_offload and
            self.current_sequential_offload == sequential_offload and
            self.current_vae_tiling == vae_tiling):
            self.get_logger().debug("Using cached pipeline.")
            return

        if self.pipe is not None:
            self.get_logger().info("Configuration changed or pipe exists, clearing old pipeline...")
            del self.pipe
            self.pipe = None
            gc.collect()
            torch.cuda.empty_cache()

        start_time = time.time()
        self.get_logger().info(f"Loading Qwen-Image-2.1 pipeline from '{source}'...")

        load_kwargs = {"dtype": self.torch_dtype}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        self.pipe = QwenImage21Pipeline.from_pretrained(source, **load_kwargs)

        if sequential_offload:
            self.get_logger().info("Optimization: Enabling sequential CPU offload (Max VRAM Savings)...")
            self.pipe.enable_sequential_cpu_offload()
        elif cpu_offload:
            self.get_logger().info("Optimization: Enabling model CPU offload...")
            self.pipe.enable_model_cpu_offload()
        else:
            self.get_logger().info(f"Optimization: Moving whole model to {self.device}...")
            self.pipe.to(self.device)

        if vae_tiling:
            # The Qwen VAE has no disable_tiling, so tiling is only ever switched on.
            if hasattr(self.pipe.vae, 'enable_tiling'):
                self.pipe.vae.enable_tiling()
            if hasattr(self.pipe.vae, 'enable_slicing'):
                self.pipe.vae.enable_slicing()
            self.get_logger().info("Optimization: Tiled VAE decode enabled...")

        self.get_logger().info(f"Pipeline loaded in {time.time() - start_time:.2f}s")
        self.current_source = source
        self.current_cpu_offload = cpu_offload
        self.current_sequential_offload = sequential_offload
        self.current_vae_tiling = vae_tiling

    def prompt_callback(self, msg):
        """
        Callback for the 'prompt' subscription.
        Parses the prompt (supports JSON for dynamic ITI), loads or clears the model as needed,
        runs the inference, saves the result, and publishes the image.
        """
        prompt_data = msg.data
        prompt, json_image_url = self._parse_prompt(prompt_data)

        if prompt is None:
            return # Error already logged in _parse_prompt

        self.get_logger().info(f"Received prompt")
        self.get_logger().debug(f"Prompt: {prompt}")

        callback_start = time.time()

        try:
            # --- Model Loading / Caching Logic ---
            # We fetch parameter values dynamically to support RQT reconfiguration
            cpu_offload = self.get_parameter('cpu_offload').value
            sequential_offload = self.get_parameter('sequential_offload').value
            vae_tiling = self.get_parameter('vae_tiling').value

            self._load_pipeline(cpu_offload, sequential_offload, vae_tiling)

            # Seed handling
            current_seed = self.seed
            if current_seed == -1:
                current_seed = random.randint(0, 2**32 - 1)

            # When using sequential_offload, the generator must be on CPU
            # as the pipeline handles device placement during execution.
            gen_device = "cpu" if sequential_offload else self.device
            generator = torch.Generator(device=gen_device).manual_seed(current_seed)

            num_steps = self.get_parameter('num_inference_steps').value
            guidance = self.get_parameter('guidance_scale').value
            negative_prompt = self.get_parameter('negative_prompt').value
            num_images = self.get_parameter('num_images_per_prompt').value
            # `output_resolution` is what the pipeline resizes the condition images to (16x VAE
            # downsampling), so it caps how much detail of the reference image survives.
            output_resolution = self.get_parameter('output_resolution').value or max(self.width, self.height)

            self.get_logger().info(
                f"Inference Params: steps={num_steps}, guidance={guidance:.2f}, num_images={num_images}, "
                f"size={self.width}x{self.height}, output_resolution={output_resolution}, current_seed={current_seed}")

            call_kwargs = {
                "prompt": prompt,
                "height": self.height,
                "width": self.width,
                "output_resolution": output_resolution,
                "true_cfg_scale": guidance,
                "num_inference_steps": num_steps,
                "num_images_per_prompt": num_images,
                "generator": generator
            }

            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt

            # Determine input image
            # Priority: 1. JSON-provided image_url, 2. ROS-parameter input_image
            target_image_input = json_image_url if json_image_url else self.input_image

            if target_image_input:
                input_images = self._load_input_image(target_image_input)
                if input_images:
                    call_kwargs["image"] = input_images
                else:
                    self.get_logger().warning("Image input provided but failed to load. Falling back to text-to-image.")

            self.get_logger().info(f"Generating image...")
            gen_start = time.time()
            output = self.pipe(**call_kwargs)

            for i, image in enumerate(output.images):
                self.get_logger().info(f"Generation finished (Image {i+1}/{len(output.images)}) in {time.time() - gen_start:.2f}s")

                # Save image
                if self.image_path:
                    output_path = self._get_image_output_path()
                    image.save(output_path)
                    self.get_logger().info(f"Image saved to: {output_path}")

                    if self.image_path.endswith('auto'):
                        self.image_counter += 1

                # Publish image as BGR8 (Standard for ROS 2 vision nodes)
                # We convert PIL RGB -> Numpy -> BGR
                image_rgb = np.array(image.convert("RGB"))
                image_bgr = image_rgb[:, :, ::-1].copy() # Swaps R and B channels and ensures memory is contiguous

                image_msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
                image_msg.header.stamp = self.get_clock().now().to_msg()
                image_msg.header.frame_id = self.get_parameter('frame_id').value
                self.image_pub.publish(image_msg)

        except torch.cuda.OutOfMemoryError:
            self.get_logger().error("Error in prompt_callback: CUDA out of memory.")
            self.get_logger().error("TIP: Enable 'sequential_offload' or reduce width/height, or close other GPU apps (browsers, editors).")
            # Clear cache for next attempt
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            self.get_logger().error(f"Error in prompt_callback: {e}")

        finally:
            keep_loaded = self.get_parameter('keep_loaded').value

            if not keep_loaded:
                self.get_logger().info("Cleaning up (keep_loaded=false)...")
                if self.pipe is not None:
                    del self.pipe
                    self.pipe = None
                    self.current_source = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                self.get_logger().debug("Keeping model loaded for next prompt.")

            if self.once:
                self.get_logger().info("'once' is True, exiting node.")
                sys.exit(0)

            total_duration = time.time() - callback_start
            self.get_logger().info(f"DONE: Total callback duration: {total_duration:.2f}s")

def main(args=None):
    """
    Main entry point for the Q21Img ROS 2 node.
    """
    rclpy.init(args=args)
    node = Q21ImgNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
