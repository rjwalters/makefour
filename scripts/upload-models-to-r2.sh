#!/bin/bash
# Upload trained neural network models to Cloudflare R2
#
# Prerequisites:
#   1. wrangler CLI installed and authenticated
#   2. R2 bucket created: wrangler r2 bucket create makefour-models
#   3. Public access configured (custom domain or r2.dev subdomain)
#
# Usage: ./scripts/upload-models-to-r2.sh

set -e

BUCKET_NAME="makefour-models"
MODELS_DIR="training/models"

echo "Uploading models to R2 bucket: $BUCKET_NAME"

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found: $MODELS_DIR"
    exit 1
fi

# Upload each ONNX model
for model in "$MODELS_DIR"/*.onnx; do
    if [ -f "$model" ]; then
        filename=$(basename "$model")
        echo "Uploading $filename..."
        wrangler r2 object put "$BUCKET_NAME/$filename" --file="$model" --content-type="application/octet-stream" --remote
        echo "  Uploaded: $filename"
    fi
done

echo ""
echo "Upload complete!"
echo ""
echo "Next steps:"
echo "  1. Configure public access for the bucket in Cloudflare dashboard"
echo "  2. Set up custom domain: models.makefour.app"
echo "     Or use r2.dev public URL"
echo ""
echo "Models will be accessible at:"
echo "  https://models.makefour.app/<model-name>.onnx"
