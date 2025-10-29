# CogVideoX Quick Start Guide

This guide will help you get started with **CogVideoX**, the best quality video generation backend.

## What is CogVideoX?

CogVideoX is a state-of-the-art text-to-video generation model from Tsinghua University (THUDM). It produces near-Sora quality results and is currently the best open-source option available.

**Key Features:**
- ✨ Exceptional video quality (best among open-source models)
- 🎬 Generates ~6 second videos (49 frames at 8fps)
- 🖼️ Support for image-to-video (I2V) generation
- 📐 Native resolution: 720×480 (16:9) or 720×720 (1:1)
- 🔬 Published research with reproducible results

## Hardware Requirements

### CogVideoX-5B (Best Quality)
- **GPU**: 16GB+ VRAM (RTX 3090, RTX 4080, RTX 4090, A5000, A6000)
- **RAM**: 32GB system RAM recommended
- **Storage**: ~20GB for model weights
- **Generation Time**: 3-7 minutes per video

### CogVideoX-2B (Low-VRAM)
- **GPU**: 8GB+ VRAM (RTX 3060 Ti, RTX 3070, RTX 4060 Ti)
- **RAM**: 16GB system RAM
- **Storage**: ~10GB for model weights
- **Generation Time**: 2-4 minutes per video

## Installation

### 1. Install Dependencies

```bash
cd local_video_service

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies (includes diffusers 0.30.3+ for CogVideoX)
pip install -r requirements.txt
```

### 2. Configure for CogVideoX

The service is pre-configured to use CogVideoX by default! Just verify `.env`:

```bash
# Copy example config (if you haven't already)
cp .env.example .env

# Should show:
# PREFERRED_BACKEND=cogvideox
# LOW_VRAM=false
```

For low-VRAM GPUs (8GB), set:
```env
LOW_VRAM=true
```

### 3. Download Model Weights (Optional but Recommended)

Models auto-download on first use, but pre-downloading avoids waiting:

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# For 16GB+ VRAM (best quality)
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b

# For image-to-video support
huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir ./weights/cogvideox/CogVideoX-5b-I2V

# OR for 8GB VRAM (low-VRAM mode)
huggingface-cli download THUDM/CogVideoX-2b --local-dir ./weights/cogvideox/CogVideoX-2b
```

Download takes 10-20 minutes depending on your internet speed.

## Usage

### Start the Service

```bash
# Quick start
python start.py

# Or with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Web UI

1. Open http://localhost:8000
2. You'll see **CogVideoX** pre-selected in the backend dropdown
3. Enter your prompt (example: "A cat playing with a ball of yarn in slow motion")
4. Adjust settings:
   - FPS: 8 (native CogVideoX rate)
   - Frames: 49 (native, generates ~6 seconds)
   - Resolution: 720×480 or 720×720
5. Check "lawful use" agreement
6. Click "Generate Video"
7. Wait 3-7 minutes (longer on first run due to model loading)

### Example Prompts

**Good prompts for CogVideoX:**

```
"A golden retriever running through a field of flowers at sunset, cinematic lighting"

"Ocean waves crashing on a rocky shore, slow motion, dramatic sky"

"A steaming cup of coffee on a wooden table, morning light, shallow depth of field"

"City street at night with neon signs reflecting on wet pavement, rain, bokeh"

"Astronaut floating in space with Earth in background, realistic, 4K quality"
```

**Tips:**
- Be descriptive and specific
- Mention camera movement/style (slow motion, cinematic, etc.)
- Include lighting details (sunset, morning light, dramatic, etc.)
- Add quality markers (4K, realistic, detailed, etc.)

### Image-to-Video (I2V)

If you downloaded CogVideoX-5B-I2V:

1. Upload a reference image in the web UI
2. Enter a prompt describing the desired motion/action
3. Generate!

Example I2V prompt:
```
"The camera slowly zooms into the scene"
"The subject turns to look at the camera"
"Gentle wind blowing through the hair"
```

## API Usage

### Generate with CogVideoX

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=A serene lake at sunset with mountains in the background" \
  -F "fps=8" \
  -F "num_frames=49" \
  -F "width=720" \
  -F "height=480" \
  -F "seed=42" \
  -F "prefer_backend=cogvideox" \
  -F "lawful_use_consent=true"
```

### With Reference Image

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=Camera zooms into this beautiful landscape" \
  -F "image=@/path/to/image.jpg" \
  -F "prefer_backend=cogvideox" \
  -F "lawful_use_consent=true"
```

## Troubleshooting

### Out of Memory (OOM)

**Error:** "CUDA out of memory"

**Solutions:**
1. Enable low-VRAM mode:
   ```bash
   # In .env file
   LOW_VRAM=true
   ```
2. Close other GPU applications (Chrome, games, etc.)
3. Use CogVideoX-2B instead of 5B
4. Restart the service to clear VRAM cache

### Model Download Fails

**Error:** "Failed to download model"

**Solutions:**
1. Check internet connection
2. Verify you have ~20GB free disk space
3. Try manual download:
   ```bash
   huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b
   ```
4. If behind firewall/proxy, configure git:
   ```bash
   git config --global http.proxy http://proxy:port
   ```

### Slow Generation

**Issue:** Taking longer than 10 minutes

**Causes & Fixes:**
- **First run**: Model loading + potential download (normal, 10-15 min)
- **CPU fallback**: Check if CUDA is available with `nvidia-smi`
- **Low system RAM**: Close other applications
- **Disk swapping**: Ensure 32GB+ system RAM for CogVideoX-5B

### Quality Issues

**Issue:** Video quality not as expected

**Tips:**
- Use specific, detailed prompts
- Mention cinematography terms (shallow depth of field, bokeh, etc.)
- Try different seeds (add `seed=` parameter)
- Ensure you're using CogVideoX-5B, not 2B or other backends
- Check backend in logs: should say "Successfully initialized cogvideox backend"

## Performance Tips

### Optimal Settings

For **RTX 4090 (24GB)**:
```
Backend: CogVideoX-5B
FPS: 8
Frames: 49
Resolution: 720×480
Generation Time: ~3-5 minutes
```

For **RTX 3060 Ti (8GB)**:
```
Backend: CogVideoX-2B (LOW_VRAM=true)
FPS: 8
Frames: 49
Resolution: 480×480
Generation Time: ~2-4 minutes
```

### Speed Up Generation

1. **Use CogVideoX-2B** if quality is acceptable
2. **Keep service running** between generations (models stay loaded)
3. **Batch requests** via API instead of restarting
4. **Pre-download models** to avoid first-run delays

## Comparison with Other Backends

| Feature | CogVideoX-5B | CogVideoX-2B | Mochi | Open-Sora |
|---------|--------------|--------------|-------|-----------|
| Quality | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| Speed | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| VRAM | 16GB | 8GB | 6-8GB | 10-15GB |
| Video Length | 6 sec | 6 sec | 2-3 sec | 2-4 sec |
| I2V Support | ✅ | ❌ | ⚠️ Limited | ⚠️ Limited |

## Resources

- **Paper**: [CogVideoX: Text-to-Video Diffusion Models](https://arxiv.org/abs/2408.06072)
- **GitHub**: https://github.com/THUDM/CogVideo
- **HuggingFace**: https://huggingface.co/THUDM
- **Demo Videos**: https://huggingface.co/THUDM/CogVideoX-5b (see model card)

## Example Workflow

Complete example from start to finish:

```bash
# 1. Setup
cd local_video_service
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download model (optional, ~20min)
huggingface-cli download THUDM/CogVideoX-5b --local-dir ./weights/cogvideox/CogVideoX-5b

# 3. Start service
python start.py

# 4. Generate video (in new terminal)
curl -X POST http://localhost:8000/generate \
  -F "prompt=A majestic eagle soaring over mountain peaks at golden hour" \
  -F "seed=42" \
  -F "lawful_use_consent=true"

# 5. Check result
# Video will be in ./outputs/
```

## FAQ

**Q: Is CogVideoX-5B really better than Stable Video Diffusion?**
A: Yes, CogVideoX-5B produces significantly better results with more coherent motion and higher visual fidelity.

**Q: Can I use CogVideoX commercially?**
A: Yes, it's Apache 2.0 licensed. Check the license file for full terms.

**Q: Why is the first generation so slow?**
A: Model needs to be loaded (5-10 min) and possibly downloaded (10-20 min). Subsequent generations are much faster (3-7 min).

**Q: Can I generate longer videos?**
A: CogVideoX generates 49 frames (~6 sec). For longer videos, use video interpolation tools like RIFE or generate multiple clips and stitch them.

**Q: Does it work on AMD GPUs?**
A: Not officially supported. PyTorch ROCm might work but is untested.

---

**Ready to create amazing videos!** 🎬✨
