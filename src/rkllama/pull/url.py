"""Direct URL model pull handler."""

import os
import re
from collections.abc import AsyncGenerator
from urllib.parse import unquote, urlparse

import aiofiles
import aiohttp

from rkllama.pull.base import PullHandler, PullProgress


class URLPullHandler(PullHandler):
    """Handler for pulling models from direct URLs."""

    def validate_source(self, source: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(source)
            # Must have scheme (http/https) and netloc (domain)
            if result.scheme not in ("http", "https"):
                return False
            if not result.netloc:
                return False
            # Should end with a model file extension
            path = result.path.lower()
            return path.endswith(".rkllm") or path.endswith(".rknn")
        except Exception:
            return False

    def extract_filename(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        return os.path.basename(path)

    def suggest_model_name(self, url: str) -> str:
        """Suggest a model name from the URL."""
        filename = self.extract_filename(url)
        # Remove extension
        name = os.path.splitext(filename)[0]
        # Clean up common URL patterns
        name = re.sub(r"[-_]v?\d+\.\d+\.\d+", "", name)  # Remove version numbers
        name = re.sub(r"[-_]+(rkllm|rknn)$", "", name, flags=re.IGNORECASE)
        return name or "downloaded-model"

    async def pull(
        self,
        source: str,
        model_name: str,
        models_path: str,
    ) -> AsyncGenerator[PullProgress, None]:
        """Pull a model from a direct URL."""
        if not self.validate_source(source):
            yield PullProgress(
                status="error",
                error=f"Invalid URL: {source}. Must be http(s):// URL ending with .rkllm or .rknn",
            )
            return

        filename = self.extract_filename(source)

        # Use suggested name if model_name not specified
        if not model_name:
            model_name = self.suggest_model_name(source)

        try:
            yield PullProgress(status="pulling manifest")

            # Create model directory
            model_dir = os.path.join(models_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            local_path = os.path.join(model_dir, filename)

            async with aiohttp.ClientSession() as session, session.get(source) as response:
                if response.status != 200:
                    yield PullProgress(
                        status="error",
                        error=f"HTTP error {response.status} from {source}",
                    )
                    return

                # Try to get content length
                total_size = response.content_length or 0

                # Create Modelfile (use URL as huggingface_path for reference)
                self.create_modelfile(
                    models_path=models_path,
                    model_name=model_name,
                    from_value=filename,
                    huggingface_path=source,  # Store source URL
                )

                yield PullProgress(
                    status="downloading",
                    completed=0,
                    total=total_size,
                )

                # Download with progress
                downloaded = 0
                chunk_size = 8192

                async with aiofiles.open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        # Yield progress every ~1MB
                        if downloaded % (1024 * 1024) < chunk_size:
                            yield PullProgress(
                                status="downloading",
                                completed=downloaded,
                                total=total_size,
                            )

            yield PullProgress(
                status="success",
                completed=downloaded,
                total=total_size if total_size > 0 else downloaded,
            )

        except aiohttp.ClientError as e:
            # Clean up partial download
            if "local_path" in dir() and os.path.exists(local_path):
                os.remove(local_path)

            yield PullProgress(
                status="error",
                error=f"Network error: {e}",
            )

        except Exception as e:
            # Clean up partial download
            if "local_path" in dir() and os.path.exists(local_path):
                os.remove(local_path)

            yield PullProgress(
                status="error",
                error=str(e),
            )
