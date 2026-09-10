"""Launch a SageMaker training job in script mode.

    python -m aws.sagemaker.train_job --dry-run     # print the request, spend nothing

Script mode runs ``training/train.py`` **unchanged**. If the training code had
to be modified to work on SageMaker, the abstraction would be wrong: the
container is a place to run the code, not a variant of it.

Everything here is built as a plain boto3 request rather than through the
sagemaker SDK. Two reasons: the request payload is then an ordinary dictionary
that a test can assert on without mocking a fluent builder, and the repository
does not take a large dependency to make four API calls.

Nothing in this module creates an endpoint. See ``aws/infra_notes.md``.
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

#: Prebuilt SageMaker XGBoost image. Region-specific; the account number is
#: AWS's, not yours, and differs per region — check the current URI before a
#: real run rather than trusting this constant.
DEFAULT_IMAGE_ACCOUNTS = {
    "us-east-1": "683313688378",
    "us-west-2": "246618743249",
    "eu-west-1": "141502667606",
}
XGBOOST_VERSION = "1.7-1"

#: Training runs on one instance and finishes in minutes on this dataset. Stop
#: it running for a day if something hangs.
MAX_RUNTIME_SECONDS = 3_600


def image_uri(settings: Settings) -> str:
    account = DEFAULT_IMAGE_ACCOUNTS.get(settings.aws_region)
    if account is None:
        raise ValueError(
            f"No XGBoost image account recorded for {settings.aws_region}. Look it up in the "
            f"SageMaker docs and add it to DEFAULT_IMAGE_ACCOUNTS."
        )
    return f"{account}.dkr.ecr.{settings.aws_region}.amazonaws.com/sagemaker-xgboost:{XGBOOST_VERSION}"


def job_name(prefix: str = "pdm-train", now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{prefix}-{now:%Y%m%d-%H%M%S}"


def build_request(
    settings: Settings,
    *,
    name: str | None = None,
    source_s3_uri: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The CreateTrainingJob payload. Pure — no network, no credentials."""
    bucket = settings.require_s3_bucket()
    role = settings.require_sagemaker_role_arn()
    name = name or job_name(now=now)

    return {
        "TrainingJobName": name,
        "AlgorithmSpecification": {
            "TrainingImage": image_uri(settings),
            "TrainingInputMode": "File",
        },
        "RoleArn": role,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": settings.s3_uri(settings.s3_features_prefix, "matrix"),
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "application/x-parquet",
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/models/"},
        "ResourceConfig": {
            "InstanceType": settings.sagemaker_training_instance,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},
        "HyperParameters": {
            # Script mode: the entry point is training/train.py, unmodified.
            "sagemaker_program": "training/train.py",
            "sagemaker_submit_directory": source_s3_uri or f"s3://{bucket}/code/source.tar.gz",
            # The problem definition travels with the job so a run is
            # reproducible from its own record, not from whatever config.py
            # happened to say that week.
            "prediction_horizon_hours": str(settings.prediction_horizon_hours),
            "feature_cadence_hours": str(settings.feature_cadence_hours),
            "post_failure_exclusion_hours": str(settings.post_failure_exclusion_hours),
            "gap_hours": str(settings.gap_hours),
            "window_sizes_hours": ",".join(str(size) for size in settings.window_sizes_hours),
            "cost_false_alarm": str(settings.cost_false_alarm),
            "cost_missed_failure": str(settings.cost_missed_failure),
        },
        "Tags": [
            {"Key": "project", "Value": "pdm-platform"},
            {"Key": "phase", "Value": "4"},
        ],
    }


def launch(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Submit the job. ``dry_run`` returns the request without calling AWS."""
    settings = settings or get_settings()
    request = build_request(settings, **kwargs)
    if dry_run:
        LOGGER.info("Dry run: not calling CreateTrainingJob.")
        return request

    if client is None:
        import boto3

        client = boto3.client("sagemaker", region_name=settings.aws_region)
    response = client.create_training_job(**request)
    LOGGER.info("Launched training job %s.", request["TrainingJobName"])
    return {"TrainingJobName": request["TrainingJobName"], **response}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the request and exit")
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings)
    print(json.dumps(launch(settings, dry_run=args.dry_run, name=args.name), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
