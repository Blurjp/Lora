"""
CogVideoX backend for text+image to video generation.

CogVideoX is a state-of-the-art open-source text-to-video model from Tsinghua University (THUDM).
Provides near-Sora quality results with reasonable VRAM requirements.

Models:
- CogVideoX-5B: High quality, requires ~16GB VRAM
- CogVideoX-2B: Good quality, requires ~8GB VRAM
- CogVideoX-5B-I2V: Image-to-video variant

Links:
- GitHub: https://github.com/THUDM/CogVideo
- HuggingFace: https://huggingface.co/THUDM/CogVideoX-5b
- Paper: https://arxiv.org/abs/2408.06072
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Callable
import torch
import numpy as np
from PIL import Image
import imageio
from tqdm.auto import tqdm

logger = logging.getLogger("video_service")


class DebugProgressCallback:
    """Callback to log detailed generation progress."""

    def __init__(self, logger):
        self.logger = logger
        self.step_times = []
        self.start_time = None

    def __call__(self, step: int, timestep: int, latents: torch.Tensor):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)

        # Log detailed step info
        self.logger.info(f"🔄 Step {step}/50 | Timestep: {timestep:.2f} | Elapsed: {elapsed:.1f}s")
        self.logger.info(f"   Latent shape: {latents.shape} | Dtype: {latents.dtype} | Device: {latents.device}")

        # Memory info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            self.logger.info(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Estimate remaining time
        if len(self.step_times) > 1:
            avg_step_time = elapsed / (step + 1)
            remaining_steps = 50 - (step + 1)
            eta = avg_step_time * remaining_steps
            self.logger.info(f"   ETA: {eta/60:.1f} minutes")

        self.logger.info("")  # Blank line for readability


class BackendNotAvailable(Exception):
    """Raised when backend cannot be initialized."""
    pass


class CogVideoXBackend:
    """
    CogVideoX backend using Hugging Face diffusers.

    Supports both text-to-video and image-to-video generation.
    """

    def __init__(
        self,
        weights_dir: Path,
        device: str = "cuda",
        low_vram: bool = False
    ):
        """
        Initialize CogVideoX backend.

        Args:
            weights_dir: Path to weights directory
            device: torch device ('cuda' or 'cpu')
            low_vram: Enable low-VRAM optimizations (uses 2B model)
        """
        self.weights_dir = Path(weights_dir) / "cogvideox"
        self.device = device
        self.low_vram = low_vram
        self.pipeline = None
        self.i2v_pipeline = None

        logger.info(f"Initializing CogVideoX backend on {device}")

        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU (extremely slow)")
            self.device = "cpu"

        self._load_pipeline()

    def _load_pipeline(self):
        """
        Load CogVideoX diffusion pipeline.

        Automatically selects model size based on VRAM availability:
        - Low-VRAM: CogVideoX-2B (~8GB)
        - Standard: CogVideoX-5B (~16GB)
        """
        try:
            from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
            from diffusers.utils import export_to_video
        except ImportError as e:
            raise BackendNotAvailable(
                f"Diffusers library not installed or too old: {e}\n"
                f"Install with: pip install diffusers>=0.30.0 transformers accelerate"
            )

        # Select model based on VRAM
        if self.low_vram:
            model_id = "THUDM/CogVideoX-2b"
            i2v_model_id = None  # 2B doesn't have I2V variant
            logger.info("Low-VRAM mode: using CogVideoX-2B")
        else:
            model_id = "THUDM/CogVideoX-5b"
            i2v_model_id = "THUDM/CogVideoX-5b-I2V"
            logger.info("Standard mode: using CogVideoX-5B")

        try:
            # Check for local weights first
            local_path = self.weights_dir / model_id.split("/")[-1]

            # Also check HuggingFace cache
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            cache_model_name = f"models--{model_id.replace('/', '--')}"
            cache_path = cache_dir / cache_model_name

            if local_path.exists():
                logger.info(f"✓ Loading model from local weights: {local_path}")
                model_path = str(local_path)
            elif cache_path.exists():
                # Model is in cache - will load from there automatically
                logger.info(f"✓ Loading model from HuggingFace cache: {cache_path}")
                logger.info("  (Using cached model - no download needed)")
                model_path = model_id
            else:
                logger.info(f"Downloading model from Hugging Face: {model_id}")
                logger.info("  First download may take 10-20 minutes (~20GB)")
                model_path = model_id

            # Load text-to-video pipeline
            # IMPORTANT: Use bfloat16 (not float16) for better stability
            # Official CogVideo repo uses bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                logger.info("Using bfloat16 for better I2V stability")
            elif self.device == "cuda":
                dtype = torch.float16
                logger.info("Using float16 (bfloat16 not supported)")
            else:
                dtype = torch.float32

            self.pipeline = CogVideoXPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )

            # Optimize for memory based on available VRAM
            if self.device == "cuda":
                # Check available VRAM
                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                    # IMPORTANT: RTX 4090 with 24GB VRAM - ALWAYS use full GPU mode
                    # CPU offloading causes hangs (GitHub Issue #545)
                    if total_vram >= 18.0 and not self.low_vram:
                        logger.info(f"✓ Sufficient VRAM ({total_vram:.1f}GB), using full GPU mode")
                        self.pipeline = self.pipeline.to(self.device)
                    else:
                        # Only for GPUs with <18GB VRAM
                        logger.warning(f"Limited VRAM ({total_vram:.1f}GB), enabling CPU offloading")
                        logger.warning("This may cause slow generation or hangs")
                        self.pipeline.enable_model_cpu_offload()
                        self.pipeline.enable_sequential_cpu_offload()
                except Exception as e:
                    logger.error(f"Could not check VRAM: {e}")
                    logger.info("Defaulting to full GPU mode")
                    self.pipeline = self.pipeline.to(self.device)

                # Enable memory optimizations (these don't hurt performance much)
                try:
                    self.pipeline.vae.enable_slicing()
                    self.pipeline.vae.enable_tiling()
                    logger.info("Enabled VAE slicing and tiling")
                except Exception as e:
                    logger.debug(f"Could not enable VAE optimizations: {e}")

                # Try xformers (speeds up attention)
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception:
                    logger.debug("xformers not available, using standard attention")

            # Load image-to-video pipeline with CORRECT configuration
            # Key fixes: bfloat16 dtype + height/width params in call
            if i2v_model_id and not self.low_vram:
                try:
                    local_i2v_path = self.weights_dir / i2v_model_id.split("/")[-1]
                    cache_i2v_name = f"models--{i2v_model_id.replace('/', '--')}"
                    cache_i2v_path = cache_dir / cache_i2v_name

                    if local_i2v_path.exists():
                        logger.info(f"✓ Loading I2V model from local weights: {local_i2v_path}")
                        i2v_path = str(local_i2v_path)
                    elif cache_i2v_path.exists():
                        logger.info(f"✓ Loading I2V model from HuggingFace cache: {cache_i2v_path}")
                        logger.info("  (Using cached model - no download needed)")
                        i2v_path = i2v_model_id
                    else:
                        logger.info(f"Downloading I2V model: {i2v_model_id}")
                        i2v_path = i2v_model_id

                    self.i2v_pipeline = CogVideoXImageToVideoPipeline.from_pretrained(
                        i2v_path,
                        torch_dtype=dtype
                    )

                    if self.device == "cuda":
                        # Apply VRAM-based optimization strategy
                        try:
                            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                            # High VRAM (≥18GB): Use full GPU mode for maximum speed
                            # Low VRAM (<18GB): Use CPU offloading to fit in memory
                            if total_vram >= 18.0 and not self.low_vram:
                                logger.info(f"✓ I2V: Sufficient VRAM ({total_vram:.1f}GB), using full GPU mode")
                                self.i2v_pipeline = self.i2v_pipeline.to(self.device)
                            else:
                                logger.info(f"I2V: Limited VRAM ({total_vram:.1f}GB), enabling CPU offloading")
                                self.i2v_pipeline.enable_sequential_cpu_offload()
                        except Exception as e:
                            logger.error(f"Could not check VRAM for I2V: {e}")
                            logger.info("I2V: Defaulting to full GPU mode")
                            self.i2v_pipeline = self.i2v_pipeline.to(self.device)

                        # Enable VAE optimizations (safe for both modes)
                        try:
                            self.i2v_pipeline.vae.enable_slicing()
                            self.i2v_pipeline.vae.enable_tiling()
                            logger.info("Enabled VAE slicing and tiling for I2V")
                        except Exception as e:
                            logger.debug(f"Could not enable VAE optimizations: {e}")

                    logger.info("✓ I2V pipeline loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load I2V pipeline: {e}")
                    logger.info("Will use text-to-video pipeline only")

            logger.info("CogVideoX backend initialized successfully")

        except Exception as e:
            raise BackendNotAvailable(
                f"Failed to load CogVideoX pipeline: {e}\n"
                f"Try: huggingface-cli download {model_id} --local-dir {self.weights_dir / model_id.split('/')[-1]}"
            )

    def generate_video(
        self,
        prompt: str,
        ref_image_path: Optional[str] = None,
        fps: int = 24,
        num_frames: int = 64,
        width: int = 720,
        height: int = 720,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Generate video from text and optional reference image.

        Args:
            prompt: Text description
            ref_image_path: Optional path to reference image for I2V
            fps: Frames per second (CogVideoX typically generates at 8 fps, will be upsampled)
            num_frames: Total number of frames (CogVideoX generates 49 frames)
            width: Video width (CogVideoX default is 720)
            height: Video height (CogVideoX default is 480)
            seed: Random seed for reproducibility

        Returns:
            Dictionary with generation results
        """
        start_time = time.time()

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            # Use smaller range to avoid int32 overflow
            seed = np.random.randint(0, 2**31 - 1)

        logger.info(
            f"Generating with CogVideoX: {num_frames} frames at {width}x{height}, "
            f"{fps} fps, seed={seed}"
        )

        # CogVideoX model constraints
        # T2V: Flexible resolution (recommend 720x480)
        # I2V: FIXED 480x720 (cannot be changed - model limitation)

        # Check if using I2V
        using_i2v = ref_image_path and self.i2v_pipeline

        if using_i2v:
            # I2V: MUST use model's built-in default (480x720)
            # CogVideoX-5b-I2V does NOT accept custom resolutions
            cogvideo_width = 720   # Fixed by model
            cogvideo_height = 480  # Fixed by model
            logger.info(f"I2V mode: Using model default resolution {cogvideo_width}x{cogvideo_height} (fixed)")
        else:
            # T2V: Standard dimensions (user-configurable)
            if self.low_vram:
                cogvideo_width = min(width, 480)
                cogvideo_height = min(height, 480)
            else:
                cogvideo_width = min(width, 720)
                cogvideo_height = min(height, 480)

            # Ensure dimensions are multiples of 16
            cogvideo_width = (cogvideo_width // 16) * 16
            cogvideo_height = (cogvideo_height // 16) * 16

        # CogVideoX generates 49 frames natively (6 seconds at 8fps)
        cogvideo_frames = 49

        logger.info(
            f"Adjusted to CogVideoX constraints: {cogvideo_width}x{cogvideo_height}, "
            f"{cogvideo_frames} frames"
        )

        try:
            # Handle reference image for I2V
            ref_image = None
            if ref_image_path and self.i2v_pipeline:
                # Validate image path exists and is not empty
                if not ref_image_path or ref_image_path == 'None' or not Path(ref_image_path).exists():
                    logger.warning(f"Invalid reference image path: {ref_image_path}")
                else:
                    try:
                        ref_image = Image.open(ref_image_path).convert("RGB")
                        ref_image = ref_image.resize(
                            (cogvideo_width, cogvideo_height),
                            Image.LANCZOS
                        )
                        logger.info(f"Using I2V pipeline with reference image: {ref_image_path}")
                    except Exception as e:
                        logger.error(f"Failed to load reference image: {e}")
                        logger.info("Falling back to text-to-video mode")

            # Generate video
            logger.info("="*70)
            logger.info("STARTING VIDEO GENERATION")
            logger.info("="*70)

            with torch.no_grad():
                if ref_image and self.i2v_pipeline:
                    # Image-to-video generation with detailed debugging
                    logger.info("🎬 Using I2V (Image-to-Video) pipeline")
                    logger.info(f"📐 Resolution: {cogvideo_width}x{cogvideo_height} (model defaults)")
                    logger.info(f"🖼️  Reference image: {ref_image.size}")
                    logger.info(f"📝 Prompt: {prompt[:100]}...")
                    logger.info(f"🎲 Seed: {seed}")
                    logger.info(f"🎞️  Frames: {cogvideo_frames}")
                    logger.info(f"🔧 Guidance scale: 6.0")
                    logger.info(f"⚙️  Dynamic CFG: True")
                    logger.info("")

                    # Log GPU state before generation
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        reserved = torch.cuda.memory_reserved() / (1024**3)
                        free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - allocated
                        logger.info(f"🖥️  GPU State BEFORE freeing T2V:")
                        logger.info(f"   Allocated: {allocated:.2f}GB")
                        logger.info(f"   Reserved: {reserved:.2f}GB")
                        logger.info(f"   Free: {free:.2f}GB")
                        logger.info("")

                    # CRITICAL FIX: Unload T2V pipeline to free GPU memory for I2V
                    # Both pipelines loaded = ~40GB (impossible on 24GB GPU)
                    if self.pipeline is not None:
                        logger.info("🔄 Unloading T2V pipeline to free GPU memory for I2V...")
                        try:
                            # Move T2V to CPU to free GPU memory
                            self.pipeline = self.pipeline.to("cpu")
                            torch.cuda.empty_cache()

                            if torch.cuda.is_available():
                                allocated_after = torch.cuda.memory_allocated() / (1024**3)
                                freed = allocated - allocated_after
                                logger.info(f"   ✓ Freed {freed:.2f}GB GPU memory")
                                logger.info(f"   GPU now: {allocated_after:.2f}GB allocated")
                                logger.info("")
                        except Exception as e:
                            logger.warning(f"Could not move T2V to CPU: {e}")

                    logger.info("🚀 Calling I2V pipeline...")
                    logger.info("⏳ Generation starting (will log progress every 5 steps)")
                    logger.info("")

                    def log_i2v_step(pipe, step: int, timestep: int, callback_kwargs: Dict[str, torch.Tensor]):
                        """
                        Emit periodic progress updates from the I2V denoising loop.

                        Args:
                            pipe: Active pipeline instance (unused but required by diffusers API)
                            step: Zero-based index of the current scheduler step
                            timestep: Scheduler timestep value
                            callback_kwargs: Tensors requested via callback_on_step_end_tensor_inputs
                        """
                        latents = callback_kwargs.get("latents")

                        if step == 0:
                            logger.info("\U0001f680 I2V denoising loop started")

                        if step % 5 == 0 or step >= 48:
                            logger.info(
                                "   \U0001f504 Step %02d/50 | timestep=%s | latents=%s | device=%s",
                                step + 1,
                                timestep,
                                tuple(latents.shape) if latents is not None else "unknown",
                                latents.device if latents is not None else "unknown",
                            )

                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / (1024**3)
                                reserved = torch.cuda.memory_reserved() / (1024**3)
                                logger.info(
                                    "      GPU memory: %.2fGB allocated / %.2fGB reserved",
                                    allocated,
                                    reserved
                                )

                        return {}

                    gen_start = time.time()

                    video = self.i2v_pipeline(
                        prompt=prompt,
                        image=ref_image,
                        num_videos_per_prompt=1,
                        num_inference_steps=50,
                        num_frames=cogvideo_frames,
                        # height/width NOT passed - I2V uses fixed 720x480 resolution
                        guidance_scale=6.0,
                        use_dynamic_cfg=True,
                        generator=generator,
                        callback_on_step_end=log_i2v_step,
                        callback_on_step_end_tensor_inputs=["latents"],
                    ).frames[0]

                    gen_elapsed = time.time() - gen_start
                    logger.info(f"✅ I2V generation completed in {gen_elapsed:.1f}s")
                    logger.info("")

                    # Reload T2V pipeline back to GPU for future T2V requests
                    if self.pipeline is not None and self.device == "cuda":
                        logger.info("🔄 Reloading T2V pipeline to GPU for future use...")
                        try:
                            self.pipeline = self.pipeline.to(self.device)
                            logger.info("   ✓ T2V pipeline reloaded to GPU")
                            logger.info("")
                        except Exception as e:
                            logger.warning(f"Could not reload T2V to GPU: {e}")
                else:
                    # Text-to-video generation
                    if ref_image_path:
                        logger.warning("I2V pipeline not available, using T2V mode")

                    video = self.pipeline(
                        prompt=prompt,
                        num_videos_per_prompt=1,
                        num_inference_steps=50,
                        num_frames=cogvideo_frames,
                        guidance_scale=6.0,
                        generator=generator,
                        height=cogvideo_height,
                        width=cogvideo_width,
                        use_dynamic_cfg=True,
                    ).frames[0]

            # video is a list of PIL Images or numpy arrays
            # Convert to numpy array if needed
            if isinstance(video, list) and isinstance(video[0], Image.Image):
                frames = np.array([np.array(frame) for frame in video])
            elif isinstance(video, torch.Tensor):
                frames = video.cpu().numpy()
                if frames.ndim == 4:  # (T, C, H, W)
                    frames = frames.transpose(0, 2, 3, 1)  # (T, H, W, C)
            else:
                frames = video

            # Ensure uint8 format
            if frames.max() <= 1.0:
                frames = (frames * 255).astype(np.uint8)
            else:
                frames = frames.astype(np.uint8)

            # Generate output path
            from app.utils import generate_output_filename
            from app.settings import settings

            output_filename = generate_output_filename(prompt, "mp4")
            output_path = settings.OUTPUTS_DIR / output_filename

            # CogVideoX native fps is ~8, but we can export at higher fps for smoother playback
            export_fps = 8  # Native generation FPS

            # Save video with imageio
            imageio.mimwrite(
                str(output_path),
                frames,
                fps=export_fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p'
            )

            elapsed = time.time() - start_time
            logger.info(f"Video saved to {output_path} in {elapsed:.2f}s")

            return {
                "mp4_path": str(output_path),
                "frames": len(frames),
                "fps": export_fps,
                "width": cogvideo_width,
                "height": cogvideo_height,
                "seed": seed,
                "backend": "cogvideox",
                "elapsed_time": round(elapsed, 2)
            }

        except Exception as e:
            logger.error(f"CogVideoX generation failed: {e}", exc_info=True)
            raise

    def __del__(self):
        """Cleanup GPU memory on deletion."""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline

        if hasattr(self, 'i2v_pipeline') and self.i2v_pipeline is not None:
            del self.i2v_pipeline

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
