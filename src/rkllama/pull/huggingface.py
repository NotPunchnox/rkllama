"""HuggingFace model pull handler."""

import os
from collections.abc import AsyncGenerator

import aiofiles
import aiohttp
from huggingface_hub import HfFileSystem, hf_hub_url

from rkllama.pull.base import PullHandler, PullProgress


class HuggingFacePullHandler(PullHandler):
    """Handler for pulling models from HuggingFace Hub."""

    def validate_source(self, source: str) -> bool:
        """Validate HuggingFace source format: owner/repo/file.rkllm"""
        # Format: owner/repo/file.rkllm
        parts = source.split("/")
        if len(parts) < 3:
            return False

        # Check if the last part looks like a file
        return parts[-1].endswith((".rkllm", ".rknn"))

    def parse_source(self, source: str) -> tuple[str, str, str]:
        """
        Parse source into components.

        Args:
            source: HuggingFace source string (owner/repo/file.rkllm)

        Returns:
            Tuple of (repo_id, filename, suggested_model_name)
        """
        parts = source.split("/")
        filename = parts[-1]
        repo_id = "/".join(parts[:2])

        # Suggested model name from filename without extension
        model_name = os.path.splitext(filename)[0]

        return repo_id, filename, model_name

    async def pull(
        self,
        source: str,
        model_name: str,
        models_path: str,
    ) -> AsyncGenerator[PullProgress, None]:
        """Pull a model from HuggingFace Hub."""
        if not self.validate_source(source):
            yield PullProgress(
                status="error",
                error=f"Invalid HuggingFace source format: {source}. Expected: owner/repo/file.rkllm",
            )
            return

        repo_id, filename, suggested_name = self.parse_source(source)

        # Use suggested name if model_name not specified
        if not model_name:
            model_name = suggested_name

        try:
            # Get file info
            yield PullProgress(status="pulling manifest")

            fs = HfFileSystem()
            file_info = fs.info(f"{repo_id}/{filename}")
            total_size = file_info["size"]

            if total_size == 0:
                yield PullProgress(
                    status="error",
                    error="Unable to retrieve file size from HuggingFace",
                )
                return

            # Create model directory
            model_dir = os.path.join(models_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            local_path = os.path.join(model_dir, filename)

            # Create Modelfile
            self.create_modelfile(
                models_path=models_path,
                model_name=model_name,
                from_value=filename,
                huggingface_path=repo_id,
            )

            # Get download URL
            url = hf_hub_url(repo_id=repo_id, filename=filename)

            yield PullProgress(
                status="downloading",
                completed=0,
                total=total_size,
            )

            # Download with progress
            downloaded = 0
            chunk_size = 8192

            async with aiohttp.ClientSession() as session, session.get(url) as response:
                if response.status != 200:
                    yield PullProgress(
                        status="error",
                        error=f"HTTP error {response.status} from HuggingFace",
                    )
                    return

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
