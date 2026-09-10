"""The AWS job launchers and the Glue transforms.

Nothing here calls AWS. The launchers are split so the request payload is a
plain dictionary a test can assert on, and the assertions that matter are the
ones about money and about correctness:

* no code path creates an endpoint
* the transform does not split its input, because splitting strips the header
  the feature-hash check reads
* the Glue job's plausibility gates match the ones in ``data_layer/schemas.py``
"""

from __future__ import annotations

import inspect
from datetime import datetime

import pytest

from aws.glue_jobs import etl_telemetry
from aws.sagemaker import train_job, transform_job
from config import MissingCredentialError, Settings
from data_layer.schemas import SENSOR_RANGES

NOW = datetime(2015, 11, 1, 9, 30, 0)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        s3_bucket="pdm-platform",
        aws_region="us-east-1",
        glue_role_arn="arn:aws:iam::123456789012:role/pdm-glue-role",
        sagemaker_role_arn="arn:aws:iam::123456789012:role/pdm-sagemaker-role",
    )


class FakeSageMaker:
    def __init__(self) -> None:
        self.training: list[dict] = []
        self.transforms: list[dict] = []

    def create_training_job(self, **kwargs):
        self.training.append(kwargs)
        return {"TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/x"}

    def create_transform_job(self, **kwargs):
        self.transforms.append(kwargs)
        return {"TransformJobArn": "arn:aws:sagemaker:us-east-1:123:transform-job/x"}


# --- Training job ----------------------------------------------------------


def test_the_training_request_points_at_the_feature_matrix(settings: Settings) -> None:
    request = train_job.build_request(settings, now=NOW)
    channel = request["InputDataConfig"][0]
    assert channel["DataSource"]["S3DataSource"]["S3Uri"] == "s3://pdm-platform/features/matrix"
    assert request["ResourceConfig"]["InstanceCount"] == 1


def test_script_mode_runs_the_training_module_unchanged(settings: Settings) -> None:
    """If the training code had to be modified to run on SageMaker, the
    abstraction would be wrong."""
    request = train_job.build_request(settings, now=NOW)
    assert request["HyperParameters"]["sagemaker_program"] == "training/train.py"


def test_the_problem_definition_travels_with_the_job(settings: Settings) -> None:
    """So a run is reproducible from its own record, not from whatever config.py
    happened to say that week."""
    parameters = train_job.build_request(settings, now=NOW)["HyperParameters"]
    assert parameters["prediction_horizon_hours"] == "24"
    assert parameters["feature_cadence_hours"] == "3"
    assert parameters["post_failure_exclusion_hours"] == "24"
    assert parameters["gap_hours"] == "24"
    assert parameters["window_sizes_hours"] == "3,12,24"


def test_a_runtime_ceiling_is_always_set(settings: Settings) -> None:
    """An unbounded training job that hangs bills until somebody notices."""
    request = train_job.build_request(settings, now=NOW)
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] <= 3_600


def test_the_job_name_is_timestamped(settings: Settings) -> None:
    assert train_job.build_request(settings, now=NOW)["TrainingJobName"] == "pdm-train-20151101-093000"


def test_launching_without_a_bucket_fails_before_any_api_call(settings: Settings) -> None:
    bucketless = settings.model_copy(update={"s3_bucket": None})
    with pytest.raises(MissingCredentialError, match="S3_BUCKET"):
        train_job.build_request(bucketless, now=NOW)


def test_launching_without_a_role_fails_before_any_api_call(settings: Settings) -> None:
    roleless = settings.model_copy(update={"sagemaker_role_arn": None})
    with pytest.raises(MissingCredentialError, match="SAGEMAKER_ROLE_ARN"):
        train_job.build_request(roleless, now=NOW)


def test_a_dry_run_spends_nothing(settings: Settings) -> None:
    client = FakeSageMaker()
    request = train_job.launch(settings, client=client, dry_run=True, now=NOW)
    assert client.training == []
    assert request["TrainingJobName"].startswith("pdm-train-")


def test_launching_sends_exactly_the_built_request(settings: Settings) -> None:
    client = FakeSageMaker()
    train_job.launch(settings, client=client, now=NOW)
    assert len(client.training) == 1
    assert client.training[0] == train_job.build_request(settings, now=NOW)


def test_an_unmapped_region_is_reported_rather_than_guessed(settings: Settings) -> None:
    elsewhere = settings.model_copy(update={"aws_region": "ap-south-2"})
    with pytest.raises(ValueError, match="ap-south-2"):
        train_job.image_uri(elsewhere)


# --- Transform job ---------------------------------------------------------


def test_the_transform_does_not_split_its_input(settings: Settings) -> None:
    """Splitting by line strips the header, and the header is what the
    feature-hash check in serving/inference.py reads."""
    request = transform_job.build_request(settings, now=NOW)
    assert request["TransformInput"]["SplitType"] == "None"
    assert request["TransformInput"]["ContentType"] == "application/json"


def test_scores_are_written_to_a_dated_partition(settings: Settings) -> None:
    request = transform_job.build_request(settings, as_of=datetime(2015, 11, 3))
    assert request["TransformOutput"]["S3OutputPath"] == "s3://pdm-platform/scores/dt=2015-11-03"


def test_the_transform_runs_on_one_small_instance(settings: Settings) -> None:
    resources = transform_job.build_request(settings, now=NOW)["TransformResources"]
    assert resources["InstanceCount"] == 1
    assert resources["InstanceType"] == "ml.m5.large"


def test_launching_a_transform_sends_the_built_request(settings: Settings) -> None:
    client = FakeSageMaker()
    transform_job.launch(settings, client=client, now=NOW)
    assert client.transforms == [transform_job.build_request(settings, now=NOW)]


def test_the_model_request_points_at_the_inference_handlers(settings: Settings) -> None:
    request = transform_job.build_model_request(
        settings, model_artifact_s3_uri="s3://pdm-platform/models/model.tar.gz", image="123.dkr.ecr/x:1"
    )
    assert request["PrimaryContainer"]["Environment"]["SAGEMAKER_PROGRAM"] == "serving/inference.py"
    assert request["ExecutionRoleArn"].endswith("pdm-sagemaker-role")


# --- No endpoints, anywhere ------------------------------------------------


@pytest.mark.parametrize("module", [train_job, transform_job])
def test_no_launcher_can_create_an_endpoint(module) -> None:
    """The single most common way a portfolio project produces a surprise bill.
    There is no code path here that creates one, and this test is what keeps
    that true as the module grows."""
    source = inspect.getsource(module)
    assert "create_endpoint" not in source
    assert "EndpointConfig" not in source


def test_the_scoring_driver_cannot_create_an_endpoint() -> None:
    from serving import batch_score

    source = inspect.getsource(batch_score)
    assert "create_endpoint" not in source
    assert "invoke_endpoint" not in source


# --- Glue ------------------------------------------------------------------


def test_the_glue_gates_match_the_ingest_gates() -> None:
    """A Glue job ships as a single file with no access to the repository, so
    the ranges are duplicated as literals. This is what keeps them in step."""
    assert etl_telemetry.SENSOR_RANGES == SENSOR_RANGES


def test_the_glue_job_computes_no_features() -> None:
    """Feature engineering lives in ONE module. A second Spark implementation of
    the same rolling windows is exactly the skew this repo is arranged to
    prevent."""
    source = inspect.getsource(etl_telemetry)
    for forbidden in ("rangeBetween", "rowsBetween", "_mean_3h", "lag(", "Window.orderBy"):
        assert forbidden not in source, f"{forbidden} suggests feature logic leaking into the Glue job"


def test_the_glue_module_imports_without_the_glue_runtime() -> None:
    """awsglue only exists inside the Glue container. Importing this module
    anywhere else has to work, or none of it is testable."""
    assert callable(etl_telemetry.curate_telemetry)
    assert callable(etl_telemetry.run)


def test_unrecognised_glue_arguments_do_not_fail_a_paid_run() -> None:
    args = etl_telemetry.parse_args(
        ["--raw", "s3://b/raw", "--curated", "s3://b/curated", "--extra-thing", "x"]
    )
    assert args.raw == "s3://b/raw"
    assert args.curated == "s3://b/curated"


def test_glue_arguments_are_required() -> None:
    with pytest.raises(SystemExit):
        etl_telemetry.parse_args([])
