# AWS resources, and how to tear every one of them down

Phases 1–3 need none of this. Read the cost notes before creating anything.

## Before you create a single resource

1. **Set a budget alert at $20.** Billing → Budgets → Create budget → Cost
   budget → $20 monthly, alert at 80%. Do this first; it is the only thing here
   that protects you from the mistakes below.
2. Confirm current pricing on the AWS pricing pages. Rates change and vary by
   region; nothing in this file is a quote.

## What gets created

| Resource | Name | Purpose | Cost shape |
|---|---|---|---|
| S3 bucket | `pdm-platform` | raw, curated, features, scores, model artifacts | Pennies at this size |
| IAM role | `pdm-glue-role` | Glue service principal | Free |
| IAM role | `pdm-sagemaker-role` | SageMaker service principal | Free |
| Glue job | `pdm-etl-telemetry` | raw → curated parquet | **Per DPU-hour, with a minimum charge per run** |
| Glue database | `pdm_catalog` | Data Catalog | Free below 1M objects |
| SageMaker training job | `pdm-train-*` | trains the model | Per second, one `ml.m5.xlarge` |
| SageMaker model | `pdm-xgb` | artifact + inference image | Free to hold |
| SageMaker transform job | `pdm-transform-*` | scores the fleet | Per second, one `ml.m5.large` |
| EventBridge rule | `pdm-nightly-scoring` | triggers the scoring chain | Effectively free |
| SNS topic | `pdm-work-orders` | work order delivery | Effectively free |

**There is no endpoint in this table and there must never be one.** A batch
transform starts, scores, writes and stops. An endpoint bills 24 hours a day to
answer a question asked once every three, and leaving one running is the single
most common way a portfolio project produces a surprise bill. `serving/batch_score.py`
has no code path that creates one.

## IAM

Two roles, because AWS services assume roles — they do not carry keys. **Never
put a long-lived access key inside a Glue job or a SageMaker container.**

`pdm-glue-role` — trust policy on `glue.amazonaws.com`:
- `AWSGlueServiceRole` (managed)
- `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` on
  `arn:aws:s3:::pdm-platform` and `arn:aws:s3:::pdm-platform/*` — that bucket only

`pdm-sagemaker-role` — trust policy on `sagemaker.amazonaws.com`:
- the same scoped S3 statement
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
- `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`

One IAM **user** with programmatic access for local development. Nothing else
needs a key.

## Bucket layout

```
s3://pdm-platform/
├── raw/           telemetry/dt=YYYY-MM-DD/, errors/, maint/, failures/, machines/
├── curated/       same tables, cleaned and deduplicated by the Glue job
├── features/      matrix/split=train|val|test/, manifest.json
├── scores/        dt=YYYY-MM-DD/
├── models/        SageMaker training output
└── code/          source.tar.gz for script mode
```

The local `data/` tree mirrors this exactly, so phase 4 swaps the root and not
the shape.

## The scoring chain

EventBridge (cron) → Glue `pdm-etl-telemetry` → SageMaker transform → SNS.

`serving/alerts.py` turns scores above the threshold into work orders and
publishes them. Subscribe an email address to `pdm-work-orders` to see them.

## Cost control, in order of how much it matters

1. **Never leave an endpoint running.** Check the SageMaker console before you
   close your laptop. This is number one for a reason.
2. **Budget alert at $20**, set before anything else exists.
3. **Develop locally.** Phases 1–3 need no AWS at all; DuckDB stands in for
   Athena while you write the SQL, and pandas handles this dataset comfortably.
4. **Glue bills a minimum duration per run.** Do not iterate on ETL logic by
   re-running Glue jobs. The transforms in `etl_telemetry.py` are pure functions
   of a Spark DataFrame precisely so you can develop them against a local
   session and pay for one run when it works.
5. **Partition properly.** Athena bills per TB scanned; an unpartitioned scan of
   a year of telemetry costs more than every other line here combined.
6. **Delete everything when the project is done.** Keep the code and
   `results/*.json` — the numbers are the evidence, the infrastructure is not.

## Teardown

Run in this order. Jobs first, then the model, then storage, then identity.

```bash
REGION=us-east-1
BUCKET=pdm-platform

# 1. Confirm no endpoint exists. This should print nothing at all.
aws sagemaker list-endpoints --region "$REGION" --query 'Endpoints[].EndpointName'

# 2. Any endpoint that does exist, delete it and its config now.
#    aws sagemaker delete-endpoint --endpoint-name <name> --region "$REGION"
#    aws sagemaker delete-endpoint-config --endpoint-config-name <name> --region "$REGION"

# 3. Scheduled trigger.
aws events remove-targets --rule pdm-nightly-scoring --ids 1 --region "$REGION"
aws events delete-rule --name pdm-nightly-scoring --region "$REGION"

# 4. SageMaker model. Training and transform jobs are not deletable and cost
#    nothing once finished; they age out of the console on their own.
aws sagemaker delete-model --model-name pdm-xgb --region "$REGION"

# 5. Glue.
aws glue delete-job --job-name pdm-etl-telemetry --region "$REGION"
aws glue delete-database --name pdm_catalog --region "$REGION"

# 6. SNS.
aws sns delete-topic --topic-arn "arn:aws:sns:$REGION:<account-id>:pdm-work-orders"

# 7. S3. Empties the bucket first; nothing here is recoverable afterwards.
aws s3 rm "s3://$BUCKET" --recursive
aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION"

# 8. IAM, last: the roles are what the steps above needed to run.
for ROLE in pdm-glue-role pdm-sagemaker-role; do
  for POLICY in $(aws iam list-attached-role-policies --role-name "$ROLE" \
      --query 'AttachedPolicies[].PolicyArn' --output text); do
    aws iam detach-role-policy --role-name "$ROLE" --policy-arn "$POLICY"
  done
  for POLICY in $(aws iam list-role-policies --role-name "$ROLE" \
      --query 'PolicyNames' --output text); do
    aws iam delete-role-policy --role-name "$ROLE" --policy-name "$POLICY"
  done
  aws iam delete-role --role-name "$ROLE"
done
```

Then check the bill two days later. Charges lag, and an empty console is not the
same as a zero invoice.
