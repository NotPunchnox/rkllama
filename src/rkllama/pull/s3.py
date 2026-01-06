"""S3/Cloud storage model pull handler."""

import os
from collections.abc import AsyncGenerator
from urllib.parse import urlparse

import aiofiles

from rkllama.pull.base import PullHandler, PullProgress


class S3PullHandler(PullHandler):
    """Handler for pulling models from S3-compatible storage."""

    def validate_source(self, source: str) -> bool:
        """Validate S3 URI format."""
        # Support formats:
        # - s3://bucket/path/to/model.rkllm
        # - https://bucket.s3.amazonaws.com/path/to/model.rkllm
        # - https://s3.region.amazonaws.com/bucket/path/to/model.rkllm

        try:
            parsed = urlparse(source)

            # Direct S3 URI
            if parsed.scheme == "s3":
                if not parsed.netloc:  # bucket name
                    return False
                path = parsed.path.lower()
                return path.endswith(".rkllm") or path.endswith(".rknn")

            # HTTPS S3 URLs
            if parsed.scheme == "https" and "s3" in parsed.netloc and "amazonaws.com" in parsed.netloc:
                path = parsed.path.lower()
                return path.endswith(".rkllm") or path.endswith(".rknn")

            return False
        except Exception:
            return False

    def parse_s3_uri(self, source: str) -> tuple[str, str, str]:
        """
        Parse S3 URI into components.

        Args:
            source: S3 URI (s3://bucket/key or https://...)

        Returns:
            Tuple of (bucket, key, filename)
        """
        parsed = urlparse(source)

        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        else:
            # HTTPS URL
            if ".s3." in parsed.netloc:
                # bucket.s3.region.amazonaws.com format
                bucket = parsed.netloc.split(".s3.")[0]
                key = parsed.path.lstrip("/")
            else:
                # s3.region.amazonaws.com/bucket format
                parts = parsed.path.lstrip("/").split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""

        filename = os.path.basename(key)
        return bucket, key, filename

    def suggest_model_name(self, source: str) -> str:
        """Suggest a model name from the S3 URI."""
        _, _, filename = self.parse_s3_uri(source)
        name = os.path.splitext(filename)[0]
        return name or "s3-model"

    async def pull(
        self,
        source: str,
        model_name: str,
        models_path: str,
    ) -> AsyncGenerator[PullProgress, None]:
        """Pull a model from S3-compatible storage."""
        if not self.validate_source(source):
            yield PullProgress(
                status="error",
                error=f"Invalid S3 URI: {source}. Expected: s3://bucket/path/model.rkllm",
            )
            return

        try:
            # Try to import aioboto3 (optional dependency)
            import aioboto3
        except ImportError:
            yield PullProgress(
                status="error",
                error="S3 support requires the 's3' optional dependency. Install with: pip install rkllama[s3]",
            )
            return

        bucket, key, filename = self.parse_s3_uri(source)

        # Use suggested name if model_name not specified
        if not model_name:
            model_name = self.suggest_model_name(source)

        try:
            yield PullProgress(status="pulling manifest")

            # Create model directory
            model_dir = os.path.join(models_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            local_path = os.path.join(model_dir, filename)

            session = aioboto3.Session()

            async with session.client("s3") as s3:
                # Get object metadata for size
                try:
                    head = await s3.head_object(Bucket=bucket, Key=key)
                    total_size = head.get("ContentLength", 0)
                except Exception as e:
                    yield PullProgress(
                        status="error",
                        error=f"Failed to get S3 object metadata: {e}",
                    )
                    return

                # Create Modelfile
                self.create_modelfile(
                    models_path=models_path,
                    model_name=model_name,
                    from_value=filename,
                    huggingface_path=source,  # Store S3 URI
                )

                yield PullProgress(
                    status="downloading",
                    completed=0,
                    total=total_size,
                )

                # Download with progress
                downloaded = 0

                response = await s3.get_object(Bucket=bucket, Key=key)
                stream = response["Body"]

                async with aiofiles.open(local_path, "wb") as f:
                    # Read in chunks
                    while True:
                        chunk = await stream.read(8192)
                        if not chunk:
                            break
                        await f.write(chunk)
                        downloaded += len(chunk)

                        # Yield progress every ~1MB
                        if downloaded % (1024 * 1024) < 8192:
                            yield PullProgress(
                                status="downloading",
                                completed=downloaded,
                                total=total_size,
                            )

            yield PullProgress(
                status="success",
                completed=total_size,
                total=total_size,
            )

        except Exception as e:
            # Clean up partial download
            local_path = os.path.join(models_path, model_name, filename)
            if os.path.exists(local_path):
                os.remove(local_path)

            yield PullProgress(
                status="error",
                error=str(e),
            )
