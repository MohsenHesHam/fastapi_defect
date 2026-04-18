# Surface Defect Detection FastAPI Service

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the model files in the same directory:
   - `defect_model.keras`
   - `class_names.json`

3. Run the service:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## API Usage

### POST /predict
Upload an image file to get defect detection results.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "class": "scratches",
  "confidence": 95.2,
  "defect_percentage": 12.4,
  "bbox": [10, 20, 50, 60]
}
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Upload model files to your repository or use persistent disk

## Laravel Integration

In your Laravel controller:

```php
use Illuminate\Support\Facades\Http;

public function detectDefect(Request $request)
{
    $image = $request->file('image');

    $response = Http::attach(
        'file', file_get_contents($image->getRealPath()), $image->getClientOriginalName()
    )->post('http://your-fastapi-url:8000/predict');

    $result = $response->json();

    // Merge with other data if needed
    return response()->json($result);
}
```

This ensures compatibility with Flutter and React frontends.