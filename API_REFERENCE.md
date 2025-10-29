# API Reference

Quick reference for the Local Video Generation Service REST API.

## Base URL

```
http://localhost:8000
```

## Endpoints

### 1. Web UI

```http
GET /
```

Returns the HTML web interface.

**Response**: HTML page

---

### 2. Generate Video

```http
POST /generate
```

Generate a video from text and optional image.

**Content-Type**: `multipart/form-data`

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Text description (3-1000 chars) |
| `image` | file | No | - | Reference image (JPEG, PNG) |
| `fps` | integer | No | 24 | Frames per second (1-60) |
| `num_frames` | integer | No | 64 | Number of frames (8-240) |
| `width` | integer | No | 720 | Video width (256-1920, multiple of 8) |
| `height` | integer | No | 720 | Video height (256-1080, multiple of 8) |
| `seed` | integer | No | random | Random seed (0 to 2^32-1) |
| `prefer_backend` | string | No | auto | Backend: "opensora", "mochi", or empty |
| `low_vram` | boolean | No | false | Enable low-VRAM optimizations |
| `lawful_use_consent` | boolean | Yes | - | User agreement (must be true) |

**Example Request (curl)**:

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=A serene lake at sunset with mountains in the background" \
  -F "fps=24" \
  -F "num_frames=48" \
  -F "width=512" \
  -F "height=512" \
  -F "seed=42" \
  -F "low_vram=false" \
  -F "lawful_use_consent=true"
```

**Example Request (with image)**:

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=Zoom into this beautiful scene" \
  -F "image=@/path/to/image.jpg" \
  -F "fps=24" \
  -F "num_frames=32" \
  -F "lawful_use_consent=true"
```

**Response (Success)**:

```json
{
  "success": true,
  "video_url": "/outputs/20250127_143022_serene_lake.mp4",
  "mp4_path": "/path/to/outputs/20250127_143022_serene_lake.mp4",
  "frames": 48,
  "fps": 24,
  "width": 512,
  "height": 512,
  "seed": 42,
  "backend": "mochi",
  "elapsed_time": 127.45,
  "error": null
}
```

**Response (Error)**:

```json
{
  "success": false,
  "video_url": null,
  "error": "Insufficient VRAM: mochi needs ~6.0GB, but only 4.2GB available."
}
```

---

### 3. Download Video

```http
GET /outputs/{filename}
```

Download a generated video file.

**Parameters**:
- `filename`: Video filename (from generation response)

**Example**:

```bash
curl http://localhost:8000/outputs/20250127_143022_serene_lake.mp4 \
  -o my_video.mp4
```

**Response**: MP4 video file

---

### 4. System Information

```http
GET /system-info
```

Get current system resource information.

**Response**:

```json
{
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 4090",
  "total_vram_gb": 24.0,
  "free_vram_gb": 18.5,
  "used_vram_gb": 2.3,
  "cpu_percent": 15.2,
  "ram_available_gb": 28.4,
  "ram_total_gb": 64.0,
  "backend": "mochi"
}
```

---

### 5. Health Check

```http
GET /health
```

Check service health and backend status.

**Response**:

```json
{
  "status": "healthy",
  "backend": "mochi",
  "cuda_available": true,
  "message": "Service operational with mochi backend"
}
```

**Status Values**:
- `healthy`: Service fully operational
- `degraded`: Service running with limitations
- `unhealthy`: Service cannot initialize

---

## Python Client Example

```python
import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def generate_video(
    prompt: str,
    image_path: str = None,
    fps: int = 24,
    num_frames: int = 64,
    width: int = 720,
    height: int = 720,
    seed: int = None,
    low_vram: bool = False
):
    """Generate a video using the API."""

    # Prepare form data
    data = {
        "prompt": prompt,
        "fps": fps,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "low_vram": str(low_vram).lower(),
        "lawful_use_consent": "true"
    }

    if seed is not None:
        data["seed"] = seed

    files = {}
    if image_path:
        files["image"] = open(image_path, "rb")

    # Make request
    response = requests.post(
        f"{API_URL}/generate",
        data=data,
        files=files
    )

    result = response.json()

    # Close file if opened
    if files:
        files["image"].close()

    return result


# Example usage
if __name__ == "__main__":
    # Generate video
    result = generate_video(
        prompt="A cat playing with a ball of yarn",
        fps=24,
        num_frames=48,
        width=512,
        height=512,
        seed=42,
        low_vram=False
    )

    if result["success"]:
        print(f"✅ Video generated: {result['video_url']}")
        print(f"   Backend: {result['backend']}")
        print(f"   Time: {result['elapsed_time']}s")

        # Download video
        video_url = f"{API_URL}{result['video_url']}"
        video_response = requests.get(video_url)

        output_path = Path("my_generated_video.mp4")
        with open(output_path, "wb") as f:
            f.write(video_response.content)

        print(f"   Saved to: {output_path}")
    else:
        print(f"❌ Error: {result['error']}")
```

---

## JavaScript Client Example

```javascript
// generate-video.js

async function generateVideo(config) {
    const formData = new FormData();

    // Add required fields
    formData.append('prompt', config.prompt);
    formData.append('lawful_use_consent', 'true');

    // Add optional fields
    if (config.image) formData.append('image', config.image);
    if (config.fps) formData.append('fps', config.fps);
    if (config.num_frames) formData.append('num_frames', config.num_frames);
    if (config.width) formData.append('width', config.width);
    if (config.height) formData.append('height', config.height);
    if (config.seed) formData.append('seed', config.seed);
    if (config.low_vram) formData.append('low_vram', config.low_vram);

    try {
        const response = await fetch('http://localhost:8000/generate', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            console.log('✅ Video generated:', result.video_url);
            return result;
        } else {
            console.error('❌ Generation failed:', result.error);
            throw new Error(result.error);
        }
    } catch (error) {
        console.error('❌ Request failed:', error);
        throw error;
    }
}

// Example usage
const config = {
    prompt: 'A beautiful sunset over the ocean',
    fps: 24,
    num_frames: 48,
    width: 512,
    height: 512,
    seed: 42,
    low_vram: false
};

generateVideo(config)
    .then(result => {
        console.log('Backend:', result.backend);
        console.log('Time:', result.elapsed_time, 'seconds');
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

---

## Error Codes

| HTTP Code | Meaning |
|-----------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 403 | Forbidden (lawful use consent not given) |
| 404 | Not Found (video file doesn't exist) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (backend initialization failed) |

---

## Rate Limiting

No rate limiting is implemented for local use. For production deployment, consider adding rate limiting middleware.

---

## Tips

1. **Start Small**: Test with 256×256, 16 frames first
2. **Check VRAM**: Use `/system-info` before large generations
3. **Use Seeds**: For reproducible results
4. **Low-VRAM Mode**: Enable if you get OOM errors
5. **Async Processing**: For multiple videos, queue requests

---

## OpenAPI Documentation

Interactive API docs are available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide an interactive interface to test the API directly in your browser.
