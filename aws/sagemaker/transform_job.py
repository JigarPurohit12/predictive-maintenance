"""Launch a SageMaker batch transform.

    python -m aws.sagemaker.transform_job --dry-run

Batch transform, not an endpoint. It starts, scores the fleet, writes to S3 and
stops, and it bills only for the minutes it ran. A persistent endpoint would
bill 24 hours a day to answer a question asked once every three, and forgetting
to delete one is the single most common way a portfolio project produces a
surprise bill.

``SplitType: None`` is deliberate: each input file arrives as one payload with
its column names intact, which is what makes the feature-hash assertion in
``serving/inference.py`` possible. Splitting by line would strip the header and
leave the container guessing what column 7 is — precisely the failure this
project is built to prevent. A fleet-scale batch is a few hundred rows, so
there is nothing to gain from splitting anyway.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from config import Settings, configure_logging, get_settings

LOGGER = logging.getLogger(__name__)

MAX_PAYLOAD_MB = 6
MAX_RUNTIME_SECONDS = 1_800


def job_name(prefix: str = "pdm-transform", now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{prefix}-{now:%Y%m%d-%H%M%S}"


def build_request(
    settings: Settings,
    *,
    model_name: str | None = None,
    name: str | None = None,
    as_of: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The CreateTransformJob payload. Pure — no network, no credentials."""
    settings.require_s3_bucket()
    partition = (as_of or now or datetime.now()).strftime("%Y-%m-%d")

    return {
        "TransformJobName": name or job_name(now=now),
        "ModelName": model_name or settings.mlflow_model_name,
        "MaxPayloadInMB": MAX_PAYLOAD_MB,
        "BatchStrategy": "SingleRecord",
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": settings.s3_uri(settings.s3_features_prefix, f"scoring/dt={partition}"),
                }
            },
            "ContentType": "application/json",
            # See the module docstring: splitting strips the header, and the
            # header is what the feature-hash check reads.
            "SplitType": "None",
            "CompressionType": "None",
        },
        "TransformOutput": {
            "S3OutputPath": settings.s3_uri(settings.s3_scores_prefix, f"dt={partition}"),
            "Accept": "application/jsonlines",
            "AssembleWith": "Line",
        },
        "TransformResources": {
            "InstanceType": settings.sagemaker_transform_instance,
            "InstanceCount": 1,
        },
        "Environment": {
            "PREDICTION_HORIZON_HOURS": str(settings.prediction_horizon_hours),
            "FEATURE_CADENCE_HOURS": str(settings.feature_cadence_hours),
        },
        "Tags": [
            {"Key": "project", "Value": "pdm-platform"},
            {"Key": "phase", "Value": "4"},
        ],
    }


def build_model_request(
    settings: Settings,
    *,
    model_artifact_s3_uri: str,
    image: str,
    name: str | None = None,
) -> dict[str, Any]:
    """CreateModel — the artifact a transform job points at."""
    return {
        "ModelName": name or settings.mlflow_model_name,
        "ExecutionRoleArn": settings.require_sagemaker_role_arn(),
        "PrimaryContainer": {
            "Image": image,
            "ModelDataUrl": model_artifact_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "serving/inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_artifact_s3_uri,
            },
        },
        "Tags": [{"Key": "project", "Value": "pdm-platform"}],
    }


def launch(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit the transform. ``dry_run`` returns the request without calling AWS."""
    settings = settings or get_settings()
    request = build_request(settings, **kwargs)
    if dry_run:
        LOGGER.info("Dry run: not calling CreateTransformJob.")
        return request

    if client is None:
        import boto3

        client = boto3.client("sagemaker", region_name=settings.aws_region)
    response = client.create_transform_job(**request)
    LOGGER.info("Launched transform job %s.", request["TransformJobName"])
    return {"TransformJobName": request["TransformJobName"], **response}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--as-of", type=datetime.fromisoformat, default=None)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    request = launch(settings, dry_run=args.dry_run, as_of=args.as_of, model_name=args.model_name)
    print(json.dumps(request, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
